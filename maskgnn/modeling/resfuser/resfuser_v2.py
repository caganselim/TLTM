import torch.nn
from torch import nn
from maskgnn.modeling.resfuser.build import RESFUSER_REGISTRY

@RESFUSER_REGISTRY.register()
class ResFuserV2(nn.Module):

    """ResFuserV2"""

    def __init__(self, cfg):

        super(ResFuserV2, self).__init__()

        fmap_dim = cfg.MODEL.RESFUSER.FMAP_DIM
        hidden_dim = cfg.MODEL.RESFUSER.HIDDEN_DIM
        
        self.cn1 = nn.Conv2d(fmap_dim * 2, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.cn2 = nn.Conv2d(hidden_dim, fmap_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.cn3 = nn.Conv2d(fmap_dim * 2, fmap_dim * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU()

    def forward(self, f_0, f_1):
        
        f_cat = torch.cat([f_0, f_1], dim=1)
        
        h = self.act(self.cn3(f_cat))
        h = self.act(self.cn3(h))
        h = self.act(self.cn1(h))
        f_1_delta = self.cn2(h)

        return f_1 + f_1_delta