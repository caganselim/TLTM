from typing import List, Dict
from detectron2.layers import ShapeSpec
from ..resfuser import build_resfuser
from .fcos_head import FCOSHead

class FCOSHeadDoubleResfuser(FCOSHead):
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

            f_0_encoded= self.__getattr__(f"resfuser_{idx}")(f_ref=f_1, f_cur=f_0)
            f_1_encoded= self.__getattr__(f"resfuser_{idx}")(f_ref=f_0, f_cur=f_1)
            x_0[idx] = f_0_encoded
            x_1[idx] = f_1_encoded

        return x_0, x_1


    def forward(self, x_0, x_1):

        """
        In training, we need to get both outputs. In test time, it is not required.
        """

        x_0, x_1 = self.residual_fuse(x_0, x_1)

        if self.training:

            return  self.forward_single(x_0), self.forward_single(x_1)

        else:

            return self.forward_single(x_1)