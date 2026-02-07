from evlearn.bundled.leanbase.base.funcs  import extract_name_kwargs

from .frame_detection_rtdetr       import FrameDetectionRTDETR
from .frame_detection_yolox        import FrameDetectionYoloX
from .vcf_detection_evrtdetr       import VCFDetectionEvRTDETR

MODELS_DICT = {
    'frame-detection-rtdetr'       : FrameDetectionRTDETR,
    'frame-detection-yolox'        : FrameDetectionYoloX,
    'vcf-detection-evrtdetr'       : VCFDetectionEvRTDETR,
}

def select_model(model, **kwargs):
    name, model_kwargs = extract_name_kwargs(model)

    if name not in MODELS_DICT:
        raise ValueError(f"Unknown model: '{name}'")

    return MODELS_DICT[name](**kwargs, **model_kwargs)
    return model

def construct_model(config, device, dtype, init_train, savedir, act_bits):
    """
    Construct model normally.
    Backbone will enable LRD based on config.backbone.lrd.
    """

    backbone_cfg = config.nets["backbone"]["model"]

    lrd_cfg = backbone_cfg.get("lrd", {})
    lrd_enabled = lrd_cfg.get("enabled", False)

    # No injection into config.model!
    model = select_model(
        config.model,
        config     = config,
        device     = device,
        dtype      = dtype,
        init_train = init_train,
        savedir    = savedir,
    )

    # Retroactively define backbone activation bits if that input exists:
    if act_bits:
        model._nets.backbone._net.act_bits = act_bits
    return model

