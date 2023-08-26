from detectron2.utils.registry import Registry
MATCHER_REGISTRY = Registry("Matcher")
MATCHER_REGISTRY.__doc__ = """

Registry for the Matcher module, which relates objects from feature vectors.
The registered object will be called with `obj(cfg)`.
The call should return a `nn.Module` object.
"""

def build_matcher(cfg):
    """
    Build a GNN from `cfg.MODEL.GNN.NAME`.
    """

    name = cfg.MODEL.MATCHER.NAME

    return MATCHER_REGISTRY.get(name)(cfg)