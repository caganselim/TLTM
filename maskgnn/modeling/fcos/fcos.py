import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from maskgnn.layers import DFConv2d, IOULoss
from .fcos_head import FCOSHead
from .fcos_head_double_resfuser import FCOSHeadDoubleResfuser
from .fcos_head_single_resfuser import FCOSHeadSingleResfuser
from .fcos_outputs_single import FCOSOutputsWithoutTracking

__all__ = ["FCOS"]

INF = 100000000
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOS(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features          = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides          = cfg.MODEL.FCOS.FPN_STRIDES
        self.focal_loss_alpha     = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma     = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample        = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides              = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius               = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test  = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_train   = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test    = cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.nms_thresh           = cfg.MODEL.FCOS.NMS_TH
        self.post_nms_topk_train  = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test   = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        self.thresh_with_ctr      = cfg.MODEL.FCOS.THRESH_WITH_CTR
        self.mask_on              = cfg.MODEL.MASK_ON #ywlee

        # fmt: on
        self.iou_loss = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

        self.resfuser_mode = cfg.MODEL.RESFUSER.MODE

        if self.resfuser_mode == "double":
            self.fcos_head = FCOSHeadDoubleResfuser(cfg, [input_shape[f] for f in self.in_features])
        elif self.resfuser_mode == "single":
            self.fcos_head = FCOSHeadSingleResfuser(cfg, [input_shape[f] for f in self.in_features])
        else:
            # OFF case
            self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])

    def get_proposals_and_losses(self, images, locations, output_dict, gt_instances):

        pre_nms_thresh = self.pre_nms_thresh_train
        pre_nms_topk = self.pre_nms_topk_train
        post_nms_topk = self.post_nms_topk_train

        outputs = FCOSOutputsWithoutTracking(images, locations, output_dict['logits'], output_dict['bbox_reg'],
                                               output_dict['ctrness'], self.focal_loss_alpha, self.focal_loss_gamma,
                                               self.iou_loss, self.center_sample, self.sizes_of_interest, self.strides,
                                               self.radius, self.fcos_head.num_classes, pre_nms_thresh, pre_nms_topk,
                                               self.nms_thresh, post_nms_topk, self.thresh_with_ctr, gt_instances)

        losses = outputs.losses()
        proposals = outputs.predict_proposals()

        return proposals, losses

    def forward_train(self, images_0,  features_0, images_1=None, features_1=None, gt_instances_0=None, gt_instances_1=None):

        features_0_list = [features_0[f] for f in self.in_features]
        features_1_list = [features_1[f] for f in self.in_features]

        locations = self.compute_locations(features_0_list)  # meshgrid

        if self.resfuser_mode == "single":

            output_dict_1 = self.fcos_head(features_0_list, features_1_list)
            proposals_1, losses_1 = self.get_proposals_and_losses(images_1, locations, output_dict_1, gt_instances_1)

            return proposals_1, output_dict_1, losses_1

        else:

            output_dict_0, output_dict_1 = self.fcos_head(features_0_list, features_1_list)
            proposals_0, losses_0 = self.get_proposals_and_losses(images_0, locations, output_dict_0, gt_instances_0)
            proposals_1, losses_1 = self.get_proposals_and_losses(images_1, locations, output_dict_1, gt_instances_1)

            for key in losses_0.keys():
                losses_0[key] += losses_1[key]

            return proposals_0, proposals_1, output_dict_0, output_dict_1, losses_0



    def forward_test(self, images_0,  features_0, images_1=None, features_1=None, gt_instances_0=None, gt_instances_1=None):

        pre_nms_thresh = self.pre_nms_thresh_test
        pre_nms_topk = self.pre_nms_topk_test
        post_nms_topk = self.post_nms_topk_test

        if self.resfuser_mode == "off":


            """
            Resfuser mode off.
            """

            features_1_list = [features_1[f] for f in self.in_features]
            locations = self.compute_locations(features_1_list) # meshgrid
            output_dict = self.fcos_head(features_1_list)
            outputs = FCOSOutputsWithoutTracking(images_1, locations, output_dict['logits'], output_dict['bbox_reg'],
                                                   output_dict['ctrness'], self.focal_loss_alpha, self.focal_loss_gamma,
                                                   self.iou_loss, self.center_sample, self.sizes_of_interest, self.strides,
                                                   self.radius, self.fcos_head.num_classes, pre_nms_thresh, pre_nms_topk,
                                                   self.nms_thresh, post_nms_topk, self.thresh_with_ctr, gt_instances_0)


        else:


            features_0_list = [features_0[f] for f in self.in_features]
            features_1_list = [features_1[f] for f in self.in_features]
            locations = self.compute_locations(features_0_list) # meshgrid
            output_dict = self.fcos_head(features_0_list, features_1_list)

            outputs = FCOSOutputsWithoutTracking(images_1, locations, output_dict['logits'], output_dict['bbox_reg'],
                                                   output_dict['ctrness'], self.focal_loss_alpha, self.focal_loss_gamma,
                                                   self.iou_loss, self.center_sample, self.sizes_of_interest, self.strides,
                                                   self.radius, self.fcos_head.num_classes, pre_nms_thresh, pre_nms_topk,
                                                   self.nms_thresh, post_nms_topk, self.thresh_with_ctr, gt_instances_0)

        proposals = outputs.predict_proposals()

        return proposals, outputs


    def forward(self, images_0,  features_0, images_1=None, features_1=None, gt_instances_0=None, gt_instances_1=None):

        if self.training:
            return self.forward_train(images_0,  features_0, images_1=images_1, features_1=features_1, gt_instances_0=gt_instances_0, gt_instances_1=gt_instances_1)
        else:
            return self.forward_test(images_0,  features_0, images_1=images_1, features_1=features_1, gt_instances_0=gt_instances_0, gt_instances_1=gt_instances_1)


    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(h, w, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)

        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

