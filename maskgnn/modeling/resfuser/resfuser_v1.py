import torch.nn
from torch import nn
from maskgnn.modeling.resfuser.build import RESFUSER_REGISTRY

@RESFUSER_REGISTRY.register()
class ResFuserV1(nn.Module):

    """ResFuserV1"""

    def __init__(self, cfg):

        super(ResFuserV1, self).__init__()
        fmap_dim = cfg.MODEL.RESFUSER.FMAP_DIM
        hidden_dim = cfg.MODEL.RESFUSER.HIDDEN_DIM

        self.cn1 = nn.Conv2d(fmap_dim*2, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.cn2 = nn.Conv2d(hidden_dim, fmap_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU()

    def forward(self, f_ref, f_cur):

        f_cat = torch.cat([f_ref, f_cur], dim=1)
        h = self.act(self.cn1(f_cat))
        f_delta = self.cn2(h)

        return f_cur + f_delta


