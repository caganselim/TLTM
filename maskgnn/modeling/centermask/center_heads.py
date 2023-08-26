# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList
from detectron2.utils.events import get_event_storage
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.layers import ShapeSpec

from .mask_head import build_mask_head, mask_rcnn_loss, mask_rcnn_inference
from .maskiou_head import build_maskiou_head, mask_iou_loss, mask_iou_inference
from .pooler import ROIPooler

__all__ = ["CenterROIHeads"]


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )


    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
            detected instances. Returned during inference only; may be [] during training.

            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class CenterROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches  masks directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(CenterROIHeads, self).__init__(cfg, input_shape)
        self._init_mask_head(cfg)
        self._init_mask_iou_head(cfg)


    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        assign_crit = cfg.MODEL.ROI_MASK_HEAD.ASSIGN_CRITERION

        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            assign_crit=assign_crit,
        )

        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_mask_iou_head(self, cfg):
        # fmt: off
        self.maskiou_on = cfg.MODEL.MASKIOU_ON
        if not self.maskiou_on:
            return
        in_channels = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.maskiou_weight = cfg.MODEL.MASKIOU_LOSS_WEIGHT

        # fmt : on

        self.maskiou_head = build_maskiou_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """

        del images


        if self.training:

            # Process masks
            losses, mask_features, selected_mask, labels, maskiou_targets, proposals = self._forward_mask(features, proposals)
            # losses, proposals = self._forward_mask(features, proposals)


            losses.update(self._forward_maskiou(mask_features, proposals, selected_mask, labels, maskiou_targets))

            return proposals, losses

        else:

            # During inference cascaded prediction is used: the mask heads are only
            # applied to the top scoring box detections.

            pred_instances = self.forward_with_given_boxes(features, proposals)


            return pred_instances, {}

    def forward_with_given_boxes(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_maskiou(instances[0].get('mask_features'), instances)

        return instances


    def _forward_mask(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        # features are received from p3, p4, p5
        features = [features[f] for f in self.in_features]

        if self.training:

            # The loss is only defined on positive proposals.
            # Select fg proposals selects where a gt is assigned to an instance.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)

            # From now on, proposals are fg.
            # proposal_boxes = [x.proposal_boxes for x in proposals]
            num_proposals_per_im = [len(proposals_per_im) for proposals_per_im in proposals]

            # mask_features => [K, 256, POOL_RESOLUTION, POOL_RESOLUTION]

            pooled_feats = self.mask_pooler(features, proposals)

            # mask logits => [K, C, POOL_RESOLUTION*2, POOL_RESOLUTION*2] - K: #masks, C: #classes
            mask_logits = self.mask_head(pooled_feats)

            loss, selected_mask, labels, maskiou_targets = mask_rcnn_loss(mask_logits, proposals, True)

            mask_rcnn_inference(mask_logits, proposals)

            # now proposals become pred_instances.

            return {"loss_mask": loss}, pooled_feats, selected_mask, labels, maskiou_targets, proposals

        else:

            mask_features = self.mask_pooler(features, instances, target_key="pred_boxes")
            mask_logits = self.mask_head(mask_features)
            instances[0].set('mask_features', mask_features)

            mask_rcnn_inference(mask_logits, instances)

            return instances

    def _forward_maskiou(self, mask_features, instances, selected_mask=None, labels=None, maskiou_targets=None):
        """
        Forward logic of the mask iou prediction branch.
        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, calibrate instances' scores.
        """

        if not self.maskiou_on:
            return {} if self.training else instances

        if self.training:

            pred_maskiou = self.maskiou_head(mask_features, selected_mask)

            return {"loss_maskiou": mask_iou_loss(labels, pred_maskiou, maskiou_targets, self.maskiou_weight)}

        else:

            selected_mask = torch.cat([i.pred_masks for i in instances], 0)
            if selected_mask.shape[0] == 0:
                return instances
            pred_maskiou = self.maskiou_head(mask_features, selected_mask)
            mask_iou_inference(instances, pred_maskiou)

            return instances
