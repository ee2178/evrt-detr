import logging
import torch
import os

from evlearn.bundled.leanbase.torch.funcs import seed_everything

from evlearn.config import Args
from evlearn.models import construct_model

LOGGER = logging.getLogger('evlearn.eval')

def enable_lrd_in_backbone_config(config):
    """
    Ensure backbone config has LRD enabled.
    Safe to call multiple times.
    """
    if not hasattr(config, "backbone"):
        return

    if not hasattr(config.backbone, "lrd"):
        config.backbone.lrd = {}

    config.backbone.lrd["enabled"] = True



def load_model(
    savedir, epoch, device = 'cuda', dtype = torch.float32, act_bits = None
):
    args = Args.load(savedir)
    # 🔑 detect LRD from checkpoint
    ckpt_path = os.path.join(savedir, "net_backbone.pth")
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        if any(".lrd_conv1." in k for k in sd.keys()):
            enable_lrd_in_backbone_config(args.config)
            LOGGER.info("Detected LRD backbone — enabling LRD in config")

    LOGGER.info('Starting evaluation: ')
    LOGGER.info(args.config.to_json(indent = 4))

    seed_everything(args.config.seed)

    model = construct_model(
        args.config, device, dtype, init_train = False, savedir = args.savedir, act_bits=act_bits
    )

    if epoch == -1:
        epoch = model.find_last_checkpoint_epoch()

    if epoch is None:
        model.load(None)
    elif epoch > 0:
        model.load(epoch)

    model.eval()

    return (args, model)

def load_eval_dset(args, split = 'test'):
    # pylint: disable=import-outside-toplevel
    from evlearn.data.data import construct_data_loader
    dl = construct_data_loader(args.config.data.eval, split = split)
    return dl

