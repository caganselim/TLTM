from detectron2.utils.registry import Registry
OBJ_ENCODER_REGISTRY = Registry("OBJ_ENCODER")
OBJ_ENCODER_REGISTRY.__doc__ = """

Registry for the OBJ_ENCODER Module, which encodes objects from object masks.
The registered object will be called with `obj(cfg)`.
The call should return a `nn.Module` object.

"""

def build_obj_encoder(cfg):
    """
    Build an object encoderr from `cfg.MODEL.OBJ_ENCODER.NAME`.
    """
    name = cfg.MODEL.OBJ_ENCODER.NAME
    return OBJ_ENCODER_REGISTRY.get(name)(cfg)