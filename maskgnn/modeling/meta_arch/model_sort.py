# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torchvision
from detectron2.modeling.sampling import subsample_labels
from detectron2.utils.events import get_event_storage

from torch import nn
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes, pairwise_iou

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads

# New imports
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from maskgnn.modeling.centermask import ROIPooler
from maskgnn.modeling.gnn import build_gnn
from maskgnn.modeling.matcher.build import build_matcher
from maskgnn.modeling.meta_arch.deep_sort import nn_matching
from maskgnn.modeling.meta_arch.deep_sort.detection import Detection
from maskgnn.modeling.meta_arch.deep_sort.tracker import Tracker
from maskgnn.modeling.obj_encoder.build import build_obj_encoder
from maskgnn.modeling.meta_arch.model_utils import *
from detectron2.modeling.matcher import Matcher
from maskgnn.modeling.centermask.proposal_utils import add_ground_truth_to_proposals

__all__ = ["CenterMaskGNN"]

def clean_grads(instances):
    for i in range(len(instances)):
        instances[i].pooled_feats = instances[i].pooled_feats.detach()
    return instances

@META_ARCH_REGISTRY.register()
class CenterMaskGNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(self, *, cfg):
        """
        NOTE: this interface is experimental.
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()

        # Network modules
        self.backbone =  build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.obj2obj_gnn = build_gnn(cfg)
        self.matcher = build_matcher(cfg)
        self.obj_encoder = build_obj_encoder(cfg)
        self.proposal_matcher = Matcher(cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                                          cfg.MODEL.ROI_HEADS.IOU_LABELS,
                                          allow_low_quality_matches=False)

        # Meta information
        self.input_format = cfg.INPUT.FORMAT
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        assert (self.pixel_mean.shape == self.pixel_std.shape), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.trn_loader_mode = cfg.DATASETS.TRN_LOADER_MODE
        self.val_loader_mode = cfg.DATASETS.VAL_LOADER_MODE
        self.roi_batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.roi_positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION

        # Obj related settings
        self.resfuser_mode = cfg.MODEL.RESFUSER.MODE
        self.node_pool_src = cfg.MASKGNN.NODE_REP.POOL_SRC
        self.node_frame0_src = cfg.MASKGNN.NODE_REP.FRAME0_SRC
        self.node_frame1_src = cfg.MASKGNN.NODE_REP.FRAME1_SRC
        self.is_gnn_on = cfg.MODEL.GNN.IS_ON

        # Tracking related variables
        self.max_cosine_distance = 0.4
        nn_budget = None
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, nn_budget)

        self.tracker = Tracker(self.obj2obj_gnn)

        self.processing_video = "-1"
        self.is_first = False
        self.id_counter = 0

        # Pooling related settings
        self.feature_strides = {k: v.stride for k, v in  self.backbone.output_shape().items()}
        self.pooler_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.pooler_in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        assign_crit = cfg.MODEL.ROI_MASK_HEAD.ASSIGN_CRITERION

        self.object_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            assign_crit=assign_crit,
        )

        self.forget_mode = cfg.MASKGNN.POSTPROCESSING.FORGET_MODE
        self.num_misses_to_forget = cfg.MASKGNN.POSTPROCESSING.NUM_MISSES_TO_FORGET

    @classmethod
    def from_config(cls, cfg):
        return {"cfg": cfg}

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_key="image"):
        """
        Normalize, pad and batch the input images.
        """
        images = [x[image_key].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        #instances = instance_level_nms(instances)

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            #print(len(results_per_image))
            if len(results_per_image) != 0:

                r = detector_postprocess(results_per_image, height, width)
                num_to_filter = 100
                if len(r) > num_to_filter:
                    print("FILTER")
                    scores  = r.scores
                    idxs = torch.argsort(scores,descending=True)[:num_to_filter]
                    r = r[idxs]

                processed_results.append({"instances": r})
            else:
                print("No dets!")
                processed_results.append({"instances": results_per_image})

        return processed_results


    def _tracker_update(self, dets):

        assert len(dets) == 1  # The batch size should be 1
        debug_dict = None

        det = dets[0]

        masks = det.pred_masks.to("cpu")
        print(masks.shape)

        bboxes = det.pred_boxes.tensor.to("cpu")
        scores = det.scores.to("cpu")
        features = det.obj_feats.to("cpu")
        pred_classes = det.pred_classes

        detections = [Detection(bbox, score, feature, mask, cls) for bbox, score, feature, mask, cls in zip(bboxes, scores, features, masks, pred_classes)]
        self.tracker.predict()
        self.tracker.update(detections)

        tracked_instances = Instances(det.image_size)
        classes_out = []
        bboxes_out = []
        masks_out = []
        ids_out = []

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bboxes_out.append(torch.tensor(track.to_tlbr()).unsqueeze(0))
            ids_out.append(track.track_id)
            masks_out.append(track.mask.unsqueeze(0))
            classes_out.append(track.cls.reshape(1))

        if len(bboxes_out) == 0:

            tracked_instances.set("pred_boxes", Boxes(torch.tensor([])))
            tracked_instances.set("pred_masks", torch.tensor([]))
            tracked_instances.set("tracking_id", torch.tensor([]))
            tracked_instances.set("pred_classes", torch.tensor([]))

        else:

            bboxes_out = torch.cat(bboxes_out)
            masks_out = torch.cat(masks_out)
            ids_out = torch.tensor(ids_out)
            print(classes_out)
            classes_out = torch.cat(classes_out,dim=0)
            # Print shapes
            print(f"{bboxes_out.shape} - {masks_out.shape} - {ids_out.shape}")

            tracked_instances.set("pred_boxes", Boxes(bboxes_out))
            tracked_instances.set("pred_masks", masks_out)
            tracked_instances.set("tracking_id", ids_out)
            tracked_instances.set("pred_classes", classes_out)


        return [tracked_instances], debug_dict

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0

        matched_cnt = (matched_labels).sum().cpu().item() > 0
        assert matched_cnt > 0, f"NO MATCH! - {matched_cnt}"

        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes


        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.roi_batch_size_per_image, self.roi_positive_sample_fraction, self.num_classes
        )

        # print("sampled fg/bg: ", len(sampled_fg_idxs), len(sampled_bg_idxs))

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets, append_proposals=True):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # ywlee for using targets.gt_classes
        # in add_ground_truth_to_proposal()
        # gt_boxes = [x.gt_boxes for x in targets]

        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).


        proposals_ext = add_ground_truth_to_proposals(targets, proposals)
        # print(f"prev: {[len(a) for a in proposals]}, targets:  {[len(a) for a in targets]},  next: {[len(a) for a in proposals_ext]}")

        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []

        # Process each image one by one.

        for proposals_per_image, targets_per_image in zip(proposals_ext, targets):

            has_gt = len(targets_per_image) > 0

            assert has_gt, "No gt case!"

            # Calculates pairwise IoU. Gets N and M bounding boxes, return NxM Tensor.
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)

            # Returns N best matched ground truth index
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)


            # Sample proposals from the networks output.
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals.
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").

            # Get ground truth
            sampled_targets = matched_idxs[sampled_idxs]

            # NOTE: here the indexing waste some compute, because heads
            # like masks, keypoints, etc, will filter the proposals again,
            # (by foreground/background, or number of keypoints in the image, etc)
            # so we essentially index the data twice.

            for (trg_name, trg_value) in targets_per_image.get_fields().items():
                if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                    proposals_per_image.set(trg_name, trg_value[sampled_targets])

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads (TensorBoard)
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def pool_and_encode(self, features, proposals, fcos_outputs=None, gt_on=False):

        target_key = "proposal_boxes" if self.training else "pred_boxes"

        if gt_on:
            target_key = "gt_boxes"

        num_proposals_per_im = [len(prop) for prop in proposals]

        # Then pool the feats for the objects.
        if self.node_pool_src == "backbone":
            # Pool features for masks.
            features = [features[f] for f in self.pooler_in_features]
            mask_feats = self.object_pooler(features, proposals, target_key)
            device = mask_feats.device
            obj_feats_in = mask_feats

        elif self.obj_pooling_src == "cls_ctr":
            assert fcos_outputs is not None, "You need to provide fcos outputs."
            # Pool from the cls and centerness head!
            logits = fcos_outputs["logits"]
            logits = [logits[i] for i in range(3)]
            logits_in = self.object_pooler(logits, proposals)

            ctrness = fcos_outputs["ctrness"]
            ctrness = [ctrness[i] for i in range(3)]
            ctrness_in = self.object_pooler(ctrness, proposals)
            obj_feats_in = logits_in*ctrness_in   # K, 40, 14, 14

        else:
            raise NotImplementedError

        # Encode each object afterwards to a 1-D vector.
        if obj_feats_in.shape[0] == 0:
            obj_feats = torch.empty(0, 512).to(device) # No det case
        else:
            obj_feats = self.obj_encoder(obj_feats_in)

        obj_feat_list = torch.split(obj_feats, num_proposals_per_im)
        for obj_feat, prop in zip(obj_feat_list, proposals):
            prop.set("obj_feats", obj_feat)

        return proposals


    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"

        """

        if not self.training:

            if "file_name_0" in batched_inputs[0]:

                # THIS IS A STANDARD VIDEO DATASET, E.G. YT-VOS, DAVIS.
                current_video_name = batched_inputs[0]['file_name_0'].split('/')[-2]
                width = batched_inputs[0]["width"]
                height = batched_inputs[0]["height"]

                if self.processing_video != current_video_name:
                    print("Resetting tracker... - ", current_video_name)
                    self.is_first = False
                    self.prev_bboxes = None
                    self.prev_obj_feats = None
                    self.prev_det_labels = None
                    self.prev_features = None
                    self.prev_hits = None
                    self.id_counter = 0
                    self.processing_video = current_video_name

            else:

                # Two frame detection case
                width = batched_inputs[0]["width"]
                height = batched_inputs[0]["height"]

                print("Resetting tracker...")


            if self.val_loader_mode == "double":
                return self.inference_double(batched_inputs)
            else:
                return self.inference_single(batched_inputs)

            # END OF INFERENCE PART

        else:
            # Training redirection.
            if self.trn_loader_mode == "double" or self.trn_loader_mode == "joint":
                return self.forward_maskgnn(batched_inputs)
            else:
                return self.forward_single(batched_inputs)


    def forward_maskgnn(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):

        losses = {}
        images_0 = self.preprocess_image(batched_inputs, image_key="image_0")
        images_1 = self.preprocess_image(batched_inputs, image_key="image_1")
        
        gt_instances_0 = [x["instances_0"].to(self.device) for x in batched_inputs]
        gt_instances_1 = [x["instances_1"].to(self.device) for x in batched_inputs]


        features_0, features_1 = self.backbone(images_0.tensor), self.backbone(images_1.tensor)


        if self.resfuser_mode == "single":

            proposals_1, outputs_1, proposal_losses = self.proposal_generator(images_0=images_1,
                                                                              images_1=images_1,
                                                                              features_0=features_1,
                                                                              features_1=features_1,
                                                                              gt_instances_0=gt_instances_0,
                                                                              gt_instances_1=gt_instances_1)


        else:

            proposals_0, proposals_1, \
            output_dict_0, output_dict_1, proposal_losses = self.proposal_generator(images_0=images_1,
                                                                             images_1=images_1,
                                                                             features_0=features_1,
                                                                             features_1=features_1,
                                                                             gt_instances_0=gt_instances_0,
                                                                             gt_instances_1=gt_instances_1)

            # GT and proposals are merged.
            proposals_with_gt_0 = self.label_and_sample_proposals(proposals_0, gt_instances_0, append_proposals=True)
            proposals_with_gt_0, _ = select_foreground_proposals(proposals_with_gt_0, self.num_classes)

        for key in proposal_losses.keys():
            losses[key] = proposal_losses[key]


        # Merge
        proposals_with_gt_1 = self.label_and_sample_proposals(proposals_1, gt_instances_1, append_proposals=True)
        proposals_with_gt_1, _ = select_foreground_proposals(proposals_with_gt_1, self.num_classes)

        if self.is_gnn_on:
            if self.node_frame0_src == "gt":
                gnn_objs_0 = self.pool_and_encode(features_0, gt_instances_0, gt_on=True)
            elif self.node_frame0_src == "proposal":
                gnn_objs_0 = self.pool_and_encode(features_0, proposals_with_gt_0)
            else:
                raise NotImplementedError

            if self.node_frame1_src == "gt":
                gnn_objs_1 = self.pool_and_encode(features_1, gt_instances_1, gt_on=True)
            elif self.node_frame1_src == "proposal":
                gnn_objs_1 = self.pool_and_encode(features_1, proposals_with_gt_1)
            else:
                raise NotImplementedError

            losses["loss/gnn_edge"] = self.obj2obj_gnn.forward_train(gnn_objs_0, gnn_objs_1)["loss_gnn_edge"]

        if self.resfuser_mode == "single":
            _, detector_losses_1 = self.roi_heads(images_1, features_1, proposals_with_gt_1, gt_instances_1)

            for key in detector_losses_1.keys():
                losses[key] =  detector_losses_1[key]

        else:
            _, detector_losses_0 = self.roi_heads(images_0, features_0, proposals_with_gt_0, gt_instances_0)
            _, detector_losses_1 = self.roi_heads(images_1, features_1, proposals_with_gt_1, gt_instances_1)

            for key in detector_losses_1.keys():
                losses[key] = detector_losses_0[key] + detector_losses_1[key]

        return losses


    """
    Inference Functions.
    """

    def inference_single(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]]
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs, image_key="image_0")
        features = self.backbone(images.tensor)
        pref_feats_temp =  copy.deepcopy(features)

        if self.prev_features is None:
            self.prev_features = pref_feats_temp

        proposals, outputs = self.proposal_generator(images_0=None, images_1=images,
                                                     features_0=self.prev_features,features_1=features)


        # Pool and encode
        if self.is_gnn_on:
            proposals = self.pool_and_encode(features, proposals, outputs)

        # Now we have the following keys:
        # dict_keys(['pred_boxes', 'scores', 'pred_classes', 'locations', 'pooled_feats', 'obj_feats'])
        dets, _ = self.roi_heads(images, features, proposals, None)

        dets, debug_dict = self._tracker_update(dets)
        dets = CenterMaskGNN._postprocess(dets, batched_inputs, images.image_sizes)

        dets[0]["debug_dict"] = debug_dict

        self.prev_features = pref_feats_temp


        return dets
