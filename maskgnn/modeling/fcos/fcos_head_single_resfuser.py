from typing import List
from detectron2.layers import ShapeSpec
from ..resfuser import build_resfuser
INF = 100000000
from .fcos_head import FCOSHead

class FCOSHeadSingleResfuser(FCOSHead):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__(cfg, input_shape)
        self.resfuser_0 = build_resfuser(cfg)
        self.resfuser_1 = build_resfuser(cfg)
        self.resfuser_2 = build_resfuser(cfg)
        self.resfuser_3 = build_resfuser(cfg)
        self.resfuser_4 = build_resfuser(cfg)

    def residual_fuse(self, x_0, x_1):

        for idx,(f_0, f_1) in enumerate(zip(x_0, x_1)):
            f_1_encoded = self.__getattr__(f"resfuser_{idx}")(f_ref=f_0, f_cur=f_1)
            x_1[idx] = f_1_encoded

        return x_1


    def forward(self, x_0, x_1):

        """
        In training, we need to get both outputs. In test time, it is not required.
        """

        x_1 = self.residual_fuse(x_0, x_1)

        return self.forward_single(x_1)