from detectron2.utils.registry import Registry
GNN_REGISTRY = Registry("GNN")
GNN_REGISTRY.__doc__ = """

Registry for the GNN Module, which relates objects to each other.
The registered object will be called with `obj(cfg)`.
The call should return a `nn.Module` object.

"""

def build_gnn(cfg):
    """
    Build an object encoderr from `cfg.MODEL.OBJ_ENCODER.NAME`.
    """
    name = cfg.MODEL.GNN.NAME
    return GNN_REGISTRY.get(name)(cfg)