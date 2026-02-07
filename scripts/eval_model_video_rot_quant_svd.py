import argparse
import os

import torch
import torch.nn as nn
import pandas as pd

from evlearn.train.train import eval_epoch
from evlearn.eval.eval   import load_model, load_eval_dset

from mha_utils import replace_mha_with_custom_mha
import quant_utils
import hadamard_utils
import gptq_utils
import utils
import rotation_utils
import svd_utils_refactor

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Evaluate model metrics')

    parser.add_argument(
        'model',
        metavar  = 'MODEL',
        help     = 'model directory',
        type     = str,
    )

    parser.add_argument(
        '-e', '--epoch',
        default  = None,
        dest     = 'epoch',
        help     = 'epoch',
        type     = int,
    )

    parser.add_argument(
        '--device',
        choices  = [ 'cuda', 'cpu' ],
        default  = 'cuda',
        dest     = 'device',
        help     = 'device to use for evaluation (cuda/cpu)',
        type     = str,
    )

    parser.add_argument(
        '--data-name',
        default  = None,
        dest     = 'data_name',
        help     = 'name of the dataset to use',
        type     = str,
    )

    parser.add_argument(
        '--data-path',
        default  = None,
        dest     = 'data_path',
        help     = 'path to the new dataset to evaluate',
        type     = str,
    )

    parser.add_argument(
        '--split',
        default  = 'test',
        dest     = 'split',
        help     = 'dataset split',
        type     = str,
    )

    parser.add_argument(
        '--steps',
        default  = None,
        dest     = 'steps',
        help     = 'steps for evaluation',
        type     = int,
    )

    parser.add_argument(
        '--batch-size',
        default  = None,
        dest     = 'batch_size',
        help     = 'batch size for evaluation',
        type     = int,
    )

    parser.add_argument(
        '--workers',
        default  = None,
        dest     = 'workers',
        help     = 'number of workers to use for evaluation',
        type     = int,
    )

    parser.add_argument(
        '--abit',
        default  = 16,
        dest     = 'abit',
        help     = 'activation bits for quantization (linear for now)',
        type     = int,
    )

    parser.add_argument(
        '--wbit',
        default  = 16,
        dest     = 'wbit',
        help     = 'weight bits for quantization (linear for now)',
        type     = int,
    )

    parser.add_argument(
        '--decoder_layer_only',
        default  = False,
        dest     = 'decoder_layer_only',
        help     = 'only rotate the decoder layers',
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--fp32_had',
        default  = False,
        dest     = 'fp32_had',
        help     = 'use fp32 hadamard matrix for rotation',
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--is_exact_had',
        default  = False,
        dest     = 'is_exact_had',
        help     = 'use exact hadamard matrix for rotation',
        action=argparse.BooleanOptionalAction
    )

    
    parser.add_argument(
        '--rotate',
        default  = False,
        dest     = 'rotate',
        help     = 'rotate the model',
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--rank_ratio',
        default  = 1.0,
        dest     = 'rank_ratio',
        help     = 'rank ratio for SVD',
        type     = float,
    )

    parser.add_argument(
        '--seed',
        default  = 0,
        dest     = 'seed',
        help     = 'seed for SVD',
        type     = int,
    )
    parser.add_argument(
        '--act_bits',
        default = None,
        dest    = 'act_bits',
        help    = 'number of bits for activation',
        type    = int
    )

    return parser.parse_args()

def make_eval_directory(model, savedir, mkdir = True):
    result = os.path.join(savedir, 'evals')

    if model.current_epoch is None:
        result = os.path.join(result, 'final')
    else:
        result = os.path.join(result, f'epoch_{model.current_epoch}')

    if mkdir:
        os.makedirs(result, exist_ok = True)

    return result

def save_metrics(evaldir, data_name, data_path, split, steps, metrics):
    # pylint: disable=too-many-arguments
    fname = f'metrics_data({data_name})'

    # if data_path is not None:
    #     fname += f'_path({data_path})'

    fname += f'_split({split})'

    if steps is not None:
        fname += f'_nb({steps})'

    fname += '.csv'
    path   = os.path.join(evaldir, fname)

    df = pd.Series(metrics).to_frame().T
    df.to_csv(path, index = False)

def eval_single_dataset(
    model, args, data_name, data_config, split, steps, evaldir, batch_size,
    workers, data_path
):
    # pylint: disable=too-many-arguments
    if batch_size is not None:
        data_config.batch_size = batch_size

    if workers is not None:
        data_config.workers = workers

    if data_path is not None:
        data_config.dataset['path'] = data_path

    args.config.data.eval = { data_name : data_config }
    dl = load_eval_dset(args, split = split)

    metrics = eval_epoch(
        dl, model,
        title           = f'Evaluation: {data_name}',
        steps_per_epoch = steps
    )

    save_metrics(evaldir, data_name, data_path, split, steps, metrics.get())



def profile_input_activations(
    model: nn.Module,
    *,
    name_contains=(),   # filter by module name substring(s); set to () to log all "fpn_blocks",
    module_types=(nn.Conv2d, nn.Linear),
    to_cpu=True,
    detach=True,
    keep_last=True,                 # True: keep only last activation per layer; False: append list
):
    """
    Returns:
      activations: dict[name -> tensor] if keep_last else dict[name -> list[tensor]]
      handles: list of hook handles (call .remove() when done)
    """
    activations = {}  # name -> tensor or list[tensor]
    handles = []

    def want(name, m):
        if module_types and not isinstance(m, module_types):
            return False
        if name_contains:
            return any(s in name for s in name_contains)
        return True

    def make_hook(name):
        def hook(mod, inputs):
            x = inputs[0]
            if isinstance(x, (tuple, list)):
                x = x[0]
            if not torch.is_tensor(x):
                return
            if detach:
                x = x.detach()
            if to_cpu:
                x = x.to("cpu")
            if keep_last:
                activations[name] = x
            else:
                activations.setdefault(name, []).append(x)
        return hook

    for name, m in model.named_modules():
        if want(name, m):
            handles.append(m.register_forward_pre_hook(make_hook(name)))

    return activations, handles


import torch
import torch.nn as nn

import torch
import torch.nn as nn

def profile_input_activations_all(
    model: nn.Module,
    *,
    name_contains=(),
    module_types=(nn.Conv2d, nn.Linear, nn.MultiheadAttention),
    to_cpu=True,
    detach=True,
    keep_last=True,
):
    activations = {}
    handles = []

    def want(name, m):
        if module_types and not isinstance(m, module_types):
            return False
        if name_contains:
            return any(s in name for s in name_contains)
        return True

    def _save(key, x):
        if x is None or not torch.is_tensor(x):
            return
        if detach:
            x = x.detach()
        if to_cpu:
            x = x.to("cpu")
        if keep_last:
            activations[key] = x
        else:
            activations.setdefault(key, []).append(x)

    def make_default_hook(name):
        def hook(mod, inputs):
            x = inputs[0] if len(inputs) > 0 else None
            if isinstance(x, (tuple, list)):
                x = x[0] if len(x) > 0 else None
            _save(name, x)
        return hook

    def make_mha_qkv_hook(name):
        def hook(mod, inputs):
            # (query, key, value, key_padding_mask, need_weights, attn_mask, ...)
            q = inputs[0] if len(inputs) > 0 else None
            _save(name + ".q_in", q)
        return hook

    def make_outproj_in_hook(mha_name):
        def hook(mod, inputs):
            x = inputs[0] if len(inputs) > 0 else None
            breakpoint()
            _save(mha_name + ".out_proj_in", x)
        return hook

    for name, m in model.named_modules():
        if not want(name, m):
            continue

        # 1) Special-case MHA: log q/k/v input + out_proj input
        if isinstance(m, nn.MultiheadAttention):
            handles.append(m.register_forward_pre_hook(make_mha_qkv_hook(name)))

            # breakpoint()

            # out_proj is a child module; hook its input as well
            # This covers NonDynamicallyQuantizableLinear, whether or not you included nn.Linear in module_types.
            if hasattr(m, "out_proj") and isinstance(m.out_proj, nn.Module):
                handles.append(m.out_proj.register_forward_pre_hook(make_outproj_in_hook(name)))

            continue

        # 2) Default behavior for Conv2d / Linear (and anything else in module_types)
        handles.append(m.register_forward_pre_hook(make_default_hook(name)))

    return activations, handles

def main():
    cmdargs = parse_cmdargs()

    args, model = load_model(
        cmdargs.model, epoch = cmdargs.epoch, device = cmdargs.device, act_bits = cmdargs.act_bits
    )
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    def count_params1(module):
        return sum(p.numel() for p in module.parameters())
    def count_params_mb(module, dtype_bytes=2, trainable_only=True):
        params = module.parameters()
        if trainable_only:
            params = [p for p in params if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        # return n_params * dtype_bytes / (1024 ** 2)
        return {
        "params_M": n_params / 1e6,
        "size_MB": n_params * dtype_bytes / (1024 ** 2),
        }
    # breakpoint()
    # fp16 2byte
    # print(model._nets)
    # print(model._nets['backbone'])
    # print(model._nets['encoder'])
    # print(model._nets['decoder'])
    # print(model._nets['temp_enc']) 
    # in fp16
    # count_params_mb(model._nets['backbone']) #  21.37158203125         Mb   'params_M': 11.204864
    # count_params_mb(model._nets['encoder'])  #  9.47021484375          Mb   'params_M': 4.96512
    # count_params_mb(model._nets['decoder'])  #  {'params_M': 7.306218, 'size_MB': 13.935504913330078}
    # count_params_mb(model._nets['decoder'].decoder) 'params_M': 5.975232, 'size_MB': 11.3968505859375
    # count_params_mb(model._nets['temp_enc']) #  27.38671875            Mb   'params_M': 14.358528
    # model._nets['encoder'].encoder[0].layers[0].self_attn.named_parameters().keys()
    # model._nets['encoder'].encoder[0].layers[0].self_attn.in_proj_weight

    # count_params_mb(model._nets['backbone'], 2, False) #  21.37158203125         Mb   'params_M': 11.204864   
    # count_params_mb(model._nets['encoder'], 2, False)  #  9.47021484375          Mb   'params_M': 4.96512   
        # count_params_mb(model._nets['encoder'].encoder, 2, False)       #  1.50634765625          Mb     'params_M': 0.78976   
        # count_params_mb(model._nets['encoder'].fpn_blocks, 2, False)    #  2.509765625            Mb     'params_M': 1.31584,      
        # count_params_mb(model._nets['encoder'].pan_blocks, 2, False)    #  2.509765625            Mb     'params_M': 1.31584     
    # count_params_mb(model._nets['decoder'], 2, False)  #  7.4753265380859375     Mb   'params_M': 3.919224,
    # count_params_mb(model._nets['temp_enc'], 2, False) #  27.38671875            Mb   'params_M': 14.358528
    # breakpoint()
    #### construct conv2d activation input 
    replace_mha_with_custom_mha(model._nets['encoder'])
    # print(model._nets['encoder'])
    # breakpoint()
    replace_mha_with_custom_mha(model._nets['decoder'])
    # print(model._nets['decoder'])
    # breakpoint()
    name_to_profile = 'decoder'
    # profile_input_activations
    # activations, handles = profile_input_activations_all(model._nets[name_to_profile], keep_last=False) # Conv2d Linear # , module_types=(nn.Linear,)
    
    ##### quantizaiton config
    #### adding rotation things before this

    if cmdargs.rank_ratio < 1.0:
        svd_utils_refactor.svd_lm_setup(model._nets['decoder'], cmdargs)
        # breakpoint()
    if cmdargs.rotate:
        ### add Blinear only for decoder
        if cmdargs.rank_ratio < 1.0:
            Q, Q1 = rotation_utils.rotate_evrtdetr_decoder_svd(model._nets['decoder'], cmdargs)
        else:
            Q, Q1 = rotation_utils.rotate_evrtdetr_decoder(model._nets['decoder'], cmdargs)

    if cmdargs.decoder_layer_only:
        quant_utils.add_actquant(model._nets['decoder'].decoder)
    else:
        quant_utils.add_actquant(model._nets['decoder'])
    qlayers = quant_utils.find_qlayers(model._nets['decoder'], layers=[quant_utils.ActQuantWrapper])
    # breakpoint()
    # qlayers.keys()
    if cmdargs.rotate:
        for name in qlayers:
            if cmdargs.rank_ratio < 1.0 and 'ALinear' in name:
                continue
            if 'dec_' in name:
                continue # pass decoder heads/ query heads for now
            if 'query_' in name:
                had_K, K = hadamard_utils.get_hadK(model._nets['decoder'].hidden_dim)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = cmdargs.fp32_had
            elif 'linear2' in name:
                # ffn2
                qlayers[name].online_random_had = True
                qlayers[name].had_K = Q1
                qlayers[name].K = 1
                qlayers[name].fp32_had = cmdargs.fp32_had
            else:
                if cmdargs.is_exact_had:
                    had_K, K = hadamard_utils.get_hadK(model._nets['decoder'].hidden_dim)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = cmdargs.fp32_had
                else:
                    qlayers[name].online_random_had = True
                    qlayers[name].had_K = Q
                    qlayers[name].K = 1
                    qlayers[name].fp32_had = cmdargs.fp32_had



    if cmdargs.wbit < 16:
        print(f'Quantizing weights to {cmdargs.wbit} bits')
        cmdargs.w_asym = False
        cmdargs.w_clip = True
        # sym quant, channel wise, rtn for now
        quantizers = gptq_utils.rtn_fwrdvit(model._nets['decoder'], utils.get_dev(), cmdargs)
    if cmdargs.abit < 16:
        print(f'Quantizing activations to {cmdargs.abit} bits')
        for name in qlayers:
            layer_input_bits = cmdargs.abit
            layer_groupsize = -1
            layer_a_sym = True
            layer_a_clip = 1.0
            if '_head' in name and 'dec_' in name:
                layer_input_bits = 16
            qlayers[name].quantizer.configure(bits=layer_input_bits, 
                                              groupsize=layer_groupsize, 
                                              sym=layer_a_sym, 
                                              clip_ratio=layer_a_clip)
    model._nets['decoder'] = model._nets['decoder'].to(utils.get_dev())
    print(model._nets['decoder'])
    # breakpoint()


    data_config_dict = args.config.data.eval
    assert isinstance(data_config_dict, dict)
    rotation_mode = 'h' if cmdargs.is_exact_had else 'rh'
    if not cmdargs.rotate:
        rotation_mode = ''

    evaldir = make_eval_directory(model, cmdargs.model + f"/svd_{cmdargs.rank_ratio}{'dec_only' if cmdargs.decoder_layer_only else 'wo_dec_head'}v1/mha_replace_{'rot' if cmdargs.rotate else 'no_rot'}_{rotation_mode}/w{cmdargs.wbit}a{cmdargs.abit}")# + /mha_replace

    if cmdargs.data_name is not None:
        datasets = [ cmdargs.data_name ]
    else:
        datasets = list(sorted(data_config_dict.keys()))
    
    try:
        with torch.inference_mode():
            for name in datasets:
                eval_single_dataset(
                    model, args, name, data_config_dict[name],
                    cmdargs.split, cmdargs.steps, evaldir, cmdargs.batch_size,
                    cmdargs.workers, cmdargs.data_path
                )
    except Exception as e:
        print(e)
        # for h in handles:
        #     h.remove()
        # torch.save(activations, f'/scratch/yw6594/cf/vlm/quant/evrt-detr/scripts/model/outlier/act_{name_to_profile}_all.pth')
        print(f'Saved activations to /scratch/yw6594/cf/vlm/quant/evrt-detr/scripts/model/outlier/act_{name_to_profile}_all.pth') # _linear

if __name__ == '__main__':
    main()
