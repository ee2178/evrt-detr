#!/usr/bin/env python3
"""
Distill EvRT-DETR backbone outputs after applying LRD to the student wrapper.

What it does:
1) Load teacher wrapper via evlearn.eval.eval.load_model
2) Load fresh student wrapper (same model_dir/epoch), copy weights from teacher
3) Apply LRD to student wrapper (expects wrapper)
4) Distill student backbone to match teacher backbone output distributions:
   - backbone returns list of 3 multi-resolution feature maps
   - loss = sum_i KL_hist(teacher_i || student_i)
5) Save distilled student backbone state_dict to a checkpoint file

Run example:
PYTHONPATH=$PWD python scripts/distillation/distill_backbone.py \
  models/gen1_backup/video_evrtdetr_presnet50 \
  --device cuda --split train --steps 2000 --apply-lrd --ratio 0.5 \
  --outdir models/gen1_backup/video_evrtdetr_presnet50/distilled \
  --save-name backbone_rr50_distilled.pt
"""

import os
import argparse
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Repo imports (you already use these in other scripts)
from evlearn.eval.eval import load_model, load_eval_dset
from scripts.quant_trt.real_lrd_resnet import apply_lrd_to_model

# ----------------------------
# Wrapper/_nets helpers
# ----------------------------
def _nets_items(nets) -> Iterable[Tuple[str, nn.Module]]:
    """
    Iterate (name, module) pairs from wrapper._nets which may be:
      - dict
      - NamedDict-like with _fields
      - attribute container
    """
    if isinstance(nets, dict):
        for k, v in nets.items():
            if isinstance(v, nn.Module):
                yield k, v
        return

    if hasattr(nets, "_fields") and isinstance(nets._fields, dict):
        for k, v in nets._fields.items():
            if isinstance(v, nn.Module):
                yield k, v
        return

    # fallback: try dir(...) attributes
    for k in dir(nets):
        if k.startswith("_"):
            continue
        v = getattr(nets, k, None)
        if isinstance(v, nn.Module):
            yield k, v


def get_net(wrapper, key: str) -> nn.Module:
    nets = wrapper._nets
    if isinstance(nets, dict) and key in nets:
        return nets[key]
    if hasattr(nets, "_fields") and key in nets._fields:
        return nets._fields[key]
    if hasattr(nets, key):
        return getattr(nets, key)
    # last try: indexing
    try:
        return nets[key]
    except Exception:
        avail = [k for k, _ in _nets_items(nets)]
        raise KeyError(f"Could not find net '{key}' in wrapper._nets. Available: {avail}")


def freeze_all_nets(wrapper) -> None:
    for _, net in _nets_items(wrapper._nets):
        for p in net.parameters():
            p.requires_grad_(False)


# ----------------------------
# Batch -> backbone input extraction
# ---------------------------

def extract_backbone_input(batch: Any, device: torch.device, t_index: int = 0) -> torch.Tensor:
    """
    EVRT-DETR 'video' batch format (from your breakpoint):
      batch["video"] is a list:
        v[0]: frames Tensor (T, B, C, H, W) = (21, 1, 20, 256, 320)
        v[1]: is_new_frame Tensor (T, B, 2)
        v[2]: labels (python objects)

    We distill backbone on a single frame: x = frames[t_index] with shape (B,C,H,W).
    """
    if isinstance(batch, dict) and "video" in batch:
        v = batch["video"]
        if not isinstance(v, list) or len(v) < 1:
            raise TypeError(f"Expected batch['video'] to be a list, got {type(v)} len={len(v) if hasattr(v,'__len__') else '??'}")

        frames = v[0]
        if not torch.is_tensor(frames) or frames.dim() != 5:
            raise TypeError(f"Expected v[0] to be Tensor (T,B,C,H,W), got {type(frames)} shape={getattr(frames,'shape',None)}")

        T = frames.shape[0]
        if t_index < 0 or t_index >= T:
            raise ValueError(f"t_index {t_index} out of range for T={T}")

        x = frames[t_index].to(device)  # (B,C,H,W)
        return x

    # Fallbacks if you later use a different loader
    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, (tuple, list)) and len(batch) > 0 and torch.is_tensor(batch[0]):
        x = batch[0].to(device)
        if x.dim() == 5:
            x = x[0]  # (B,C,H,W) if (T,B,C,H,W)
        return x

    raise TypeError(f"Unsupported batch type for extracting backbone input: {type(batch)}")


# ----------------------------
# KL loss on distributions of feature maps
# ----------------------------
def feat_to_channel_samples(x: torch.Tensor) -> torch.Tensor:
    """
    (B, C, H, W) -> (C, B*H*W)
    """
    if x.dim() != 4:
        raise ValueError(f"Expected feature map (B,C,H,W), got {tuple(x.shape)}")
    b, c, h, w = x.shape
    return x.permute(1, 0, 2, 3).contiguous().view(c, -1)

def channel_energy_kl(t: torch.Tensor, s: torch.Tensor, temp: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    # (B,C,H,W) -> energy per channel (B,C)
    tE = (t.float() ** 2).mean(dim=(2,3))
    sE = (s.float() ** 2).mean(dim=(2,3))

    # turn into distributions
    p = F.softmax(tE / temp, dim=1)
    q = F.softmax(sE / temp, dim=1)

    # KL(p || q), averaged over batch
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=1).mean()


# ----------------------------
# Args
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # model
    ap.add_argument("model_dir", help="Path like models/gen1_backup/video_evrtdetr_presnet50")
    ap.add_argument("--epoch", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--save-name", default="backbone_distilled.pt")

    # data
    ap.add_argument("--data-name", default="video")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data-path", default=None)

    # distill
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--q", type=float, default=0.99)
    ap.add_argument("--log-every", type=int, default=20)

    # backbone selection within _nets
    ap.add_argument("--backbone-key", default="backbone",
                    help="Key name in wrapper._nets for the backbone module")

    # act-bits control (requires your patched backbone forward signature)
    ap.add_argument("--disable-act-bits", action="store_true",
                    help="Call backbone(x, apply_act_bits=False) during distill")

    # LRD controls (passed through to your LRD util)
    ap.add_argument("--apply-lrd", action="store_true")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--ratio-mode", default="param")
    ap.add_argument("--scheme", type=int, default=2)
    ap.add_argument("--skip-1x1", action="store_true")
    ap.add_argument("--skip-if-no-mac-reduction", action="store_true")
    ap.add_argument("--skip-if-rank-gt-out", action="store_true")
    ap.add_argument(
        '--whitelist-file',
        default=None,
        dest='whitelist_file',
        help='whitelist file: layer_name,method(svd1/svd2/tucker/orig),rank_ratio',
        type=str,
    )

    # checkpoint options
    ap.add_argument("--save-full-wrapper", action="store_true",
                    help="Also save student wrapper full state_dict (in addition to backbone-only)")

    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    cmd = parse_args()
    torch.manual_seed(cmd.seed)

    if cmd.device != "cuda" and torch.cuda.is_available():
        print("[WARN] device is not cuda but CUDA is available; using specified device anyway.")
    device = torch.device(cmd.device)

    # 1) Load teacher wrapper
    args_t, teacher = load_model(cmd.model_dir, epoch=cmd.epoch, device=cmd.device)
    teacher.eval()

    # 2) Load fresh student wrapper and copy weights from teacher (wrapper-safe)
    args_s, student = load_model(cmd.model_dir, epoch=cmd.epoch, device=cmd.device)
    student.eval()

    # Copy all underlying nets weights (safest to keep wrapper consistency)
    # If you ONLY want backbone weights copied, comment this block and use backbone-only copy below.
    t_nets = dict(_nets_items(teacher._nets))
    s_nets = dict(_nets_items(student._nets))
    common = sorted(set(t_nets.keys()) & set(s_nets.keys()))
    if not common:
        raise RuntimeError("No common nets found between teacher and student wrapper._nets.")

    for k in common:
        s_nets[k].load_state_dict(t_nets[k].state_dict(), strict=True)

    print(f"[INIT] Copied weights for {len(common)} subnets: {common}")

    # 3) Apply LRD to student wrapper (expects wrapper)
    whitelist_map = None
    if cmd.whitelist_file:
        whitelist_map = {}
        with open(cmd.whitelist_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    raise ValueError(f"Invalid whitelist line: {line}")
                name, method, ratio_text = parts[:3]
                method = method.lower()
                if method not in ("svd1", "svd2", "tucker", "orig"):
                    raise ValueError(f"Invalid method in whitelist: {method}")
                ratio_val = float(ratio_text) if method != "orig" else 1.0
                whitelist_map[name] = (method, ratio_val)
                if not name.startswith("backbone."):
                    whitelist_map[f"backbone.{name}"] = (method, ratio_val)

    if whitelist_map is not None:
        print(
            f"Applying LRD with whitelist: {cmd.whitelist_file} "
            f"(entries={len(whitelist_map)})"
        )
    else:
        print(f"Applying LRD with ratio: {cmd.ratio} and scheme: {cmd.scheme}")

    apply_lrd_to_model(
        student,
        cmd.ratio,
        cmd.ratio_mode,
        cmd.device,
        cmd.scheme,
        skip_1x1=cmd.skip_1x1,
        skip_if_no_mac_reduction=cmd.skip_if_no_mac_reduction,
        skip_if_rank_gt_out=cmd.skip_if_rank_gt_out,
        whitelist_map=whitelist_map,
    )

    t_backbone = get_net(teacher, cmd.backbone_key).to(device)
    s_backbone = get_net(student, cmd.backbone_key).to(device)

    # 4) Build dataloader (reuse eval config style like your other scripts)
    data_config = args_t.config.data.eval[cmd.data_name]
    data_config.batch_size = cmd.batch_size
    data_config.workers = cmd.workers
    if cmd.data_path is not None:
        data_config.dataset["path"] = cmd.data_path
    args_t.config.data.eval = {cmd.data_name: data_config}
    dl = load_eval_dset(args_t, split=cmd.split)

    # 5) Freeze all student nets, unfreeze only backbone
    freeze_all_nets(student)
    for p in s_backbone.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [p for p in s_backbone.parameters() if p.requires_grad],
        lr=cmd.lr,
        weight_decay=cmd.weight_decay,
    )

    # 6) Distillation loop
    teacher.eval()
    student.eval()
    
    # SET TEACHER BACKBONE TO 8 BIT
    # s_backbone._net.act_bits = 8

    def freeze_bn(m):
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    s_backbone.train()
    # Freeze batch norm layers
    s_backbone.apply(freeze_bn)

    it = iter(dl)
    for step in range(cmd.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        x = extract_backbone_input(batch, device)

        opt.zero_grad(set_to_none=True)

        # teacher forward: no grads
        with torch.inference_mode():
            if cmd.disable_act_bits:
                t_feats: List[torch.Tensor] = t_backbone(x, apply_act_bits=False)
            else:
                t_feats = t_backbone(x)

        # student forward: grads
        if cmd.disable_act_bits:
            s_feats: List[torch.Tensor] = s_backbone(x, apply_act_bits=False)
        else:
            s_feats = s_backbone(x)

        if len(t_feats) != 3 or len(s_feats) != 3:
            raise RuntimeError(f"Expected 3 backbone outputs, got {len(t_feats)} and {len(s_feats)}")
        weights = [1.0 0.5 0.25]
        loss =sum(w*channel_energy_kl(t_f, s_f, temp=2.0) for w, t_f, s_f in zip(weights, t_feats, s_feats))
        loss.backward()
        opt.step()

        if step % cmd.log_every == 0:
            print(f"[DISTILL] step={step} loss={float(loss):.6f}", flush=True)

    # 7) Save distilled backbone checkpoint
    outdir = os.path.abspath(cmd.outdir or cmd.model_dir)
    os.makedirs(outdir, exist_ok=True)
    save_path = os.path.join(outdir, cmd.save_name)

    payload = {
        "backbone_key": cmd.backbone_key,
        "backbone_state_dict": s_backbone.state_dict(),
        "meta": {
            "model_dir": cmd.model_dir,
            "epoch": cmd.epoch,
            "apply_lrd": cmd.apply_lrd,
            "ratio": cmd.ratio,
            "ratio_mode": cmd.ratio_mode,
            "scheme": cmd.scheme,
            "skip_1x1": cmd.skip_1x1,
            "skip_if_no_mac_reduction": cmd.skip_if_no_mac_reduction,
            "steps": cmd.steps,
            "lr": cmd.lr,
            "weight_decay": cmd.weight_decay,
            "bins": cmd.bins,
            "q": cmd.q,
            "split": cmd.split,
            "batch_size": cmd.batch_size,
            "disable_act_bits": cmd.disable_act_bits,
        },
    }

    if cmd.save_full_wrapper:
        # Save full student wrapper weights too (can be large)
        payload["student_full_state_dict"] = {k: v.state_dict() for k, v in _nets_items(student._nets)}

    torch.save(payload, save_path)
    print(f"[SAVE] wrote distilled backbone checkpoint to: {save_path}", flush=True)


if __name__ == "__main__":
    main()

