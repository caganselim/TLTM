from detectron2.utils.registry import Registry
RESFUSER_REGISTRY = Registry("RESFUSER")
RESFUSER_REGISTRY.__doc__ = """

Registry for the RESFUSER Module.

"""

def build_resfuser(cfg):
    """
    Build an object encoderr from `cfg.MODEL.OBJ_ENCODER.NAME`.
    """
    name = cfg.MODEL.RESFUSER.NAME
    return RESFUSER_REGISTRY.get(name)(cfg)