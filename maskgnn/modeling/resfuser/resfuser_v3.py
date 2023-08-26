import torch.nn
from torch import nn
from spatial_correlation_sampler import SpatialCorrelationSampler
from maskgnn.modeling.resfuser.build import RESFUSER_REGISTRY

@RESFUSER_REGISTRY.register()
class ResFuserV3(nn.Module):

    """ResFuserV3"""

    def __init__(self, cfg):

        super(ResFuserV3, self).__init__()

        fmap_dim = cfg.MODEL.RESFUSER.FMAP_DIM * 3
        hidden_dim = cfg.MODEL.RESFUSER.HIDDEN_DIM
        output_dim = cfg.MODEL.RESFUSER.FMAP_DIM
        
        self.cn1 = nn.Conv2d(fmap_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.cn2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.correlator = SpatialCorrelationSampler(kernel_size=1, patch_size=16, stride=1, padding=0, dilation=1, dilation_patch=1)
        self.act = nn.ReLU()

    def forward(self, f_0, f_1):

        f_correlated = self.correlator(f_0, f_1)
        b, ph, pw, h, w = f_correlated.size()
        f_correlated = f_correlated.view(b, ph * pw, h, w)
        
        f_cat = torch.cat([f_0, f_correlated, f_1], dim=1)
        
        h = self.act(self.cn1(f_cat))
        f_1_delta = self.cn2(h)

        return f_1 + f_1_delta