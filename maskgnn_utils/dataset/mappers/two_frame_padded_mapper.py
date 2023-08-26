import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable
import maskgnn_utils.dataset.mappers.map_utils as utils

from detectron2.data import transforms as T

__all__ = ["TwoFrameDatasetMapperTrain" , "TwoFrameDatasetMapperTest"]

class TwoFrameDatasetMapperTrain:

    """
    The callable currently does the following:

    1. Read the images from "file_name_1" and "file_name_2
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):

        """

        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """

        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"

        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):

        augs = utils.build_augmentation(cfg, is_train)

        print("INSIDE FROM CONFIG: AUGS =>>>>>>", augs)

        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:

            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # Load images
        image_0 = utils.read_image(dataset_dict["file_name_0"], format=self.image_format)
        utils.check_image_size(dataset_dict, image_0)

        # Sample a next frame
        next_frame_infos = dataset_dict.pop("next_frame_infos")
        num_next_frames = len(next_frame_infos)
        read_idx = np.random.randint(0, num_next_frames)

        # Then append to the current dict, the rest is the same!
        next_frame_dict = copy.deepcopy(next_frame_infos[read_idx])

        for key in next_frame_dict.keys():
            dataset_dict[key] = next_frame_dict[key]

        image_1 = utils.read_image(dataset_dict["file_name_1"], format=self.image_format)
        utils.check_image_size(dataset_dict, image_1)

        # Process the first image
        aug_input_0 = T.AugInput(image_0, sem_seg=None)
        transforms_0 = self.augmentations(aug_input_0)
        image_0 = aug_input_0.image

        # Process the second image, but this time, use the previous transforms.
        image_1 = transforms_0.apply_image(image_1)
        image_shape = image_0.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image_0"] = torch.as_tensor(np.ascontiguousarray(image_0.transpose(2, 0, 1)))
        dataset_dict["image_1"] = torch.as_tensor(np.ascontiguousarray(image_1.transpose(2, 0, 1)))


        if ("annotations_0" in dataset_dict) and ("annotations_1" in dataset_dict):

            # USER: Modify this if you want to keep them for some reason.

            for anno in dataset_dict["annotations_0"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            for anno in dataset_dict["annotations_1"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)


            # USER: Implement additional transformations if you have other types of data
            annos_0 = [utils.transform_instance_annotations(obj, transforms_0, image_shape,
                                                          keypoint_hflip_indices=self.keypoint_hflip_indices)
                for obj in dataset_dict.pop("annotations_0")
                if obj.get("iscrowd", 0) == 0
            ]

            annos_1 = [utils.transform_instance_annotations(obj, transforms_0, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
                for obj in dataset_dict.pop("annotations_1")
                if obj.get("iscrowd", 0) == 0
            ]

            instances_0 = utils.annotations_to_instances(annos_0, image_shape, mask_format=self.instance_mask_format)
            instances_1 = utils.annotations_to_instances(annos_1, image_shape, mask_format=self.instance_mask_format)

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.

            if self.recompute_boxes:
                instances_0.gt_boxes = instances_0.gt_masks.get_bounding_boxes()
                instances_1.gt_boxes = instances_1.gt_masks.get_bounding_boxes()

            instances_0 = utils.filter_empty_instances(instances_0)
            instances_1 = utils.filter_empty_instances(instances_1)

            # After we're done, calculate matchings etc.
            track_ids_0 = instances_0.gt_track_id
            track_ids_1 = instances_1.gt_track_id

            # Match the targets based on the index.
            matches_0, matches_1 = [], []
            gt_bboxes_0, gt_bboxes_1 = [], []

            if len(instances_0.gt_boxes) != 0:

                for i, id_0 in enumerate(track_ids_0):
                    is_matched = False

                    for j, id_1 in enumerate(track_ids_1):
                        if id_0 == id_1:
                            matches_0.append(i)
                            matches_1.append(j)
                            gt_bboxes_0.append(instances_0.gt_boxes.tensor[i, :])
                            gt_bboxes_1.append(instances_1.gt_boxes.tensor[j, :])
                            is_matched = True

                    if not is_matched:

                        gt_bboxes_0.append(torch.zeros_like(instances_0.gt_boxes.tensor[i, :]))
                        gt_bboxes_1.append(torch.zeros_like(instances_0.gt_boxes.tensor[i, :]))

                gt_bboxes_0 = torch.stack(gt_bboxes_0, dim=0)
                gt_bboxes_1 = torch.stack(gt_bboxes_1, dim=0)


                # Calculate loss target.
                gt_w0 = gt_bboxes_0[:, 2] - gt_bboxes_0[:, 0] + 0.0001
                gt_h0 = gt_bboxes_0[:, 3] - gt_bboxes_0[:, 1] + 0.0001
                gt_w1 = gt_bboxes_1[:, 2] - gt_bboxes_1[:, 0] + 0.0001
                gt_h1 = gt_bboxes_1[:, 3] - gt_bboxes_1[:, 1] + 0.0001

                gt_delta_x = (gt_bboxes_1[:, 0] - gt_bboxes_0[:, 0]) / gt_w0
                gt_delta_y = (gt_bboxes_1[:, 1] - gt_bboxes_0[:, 1]) / gt_h0
                gt_delta_w = torch.log(gt_w1 / gt_w0)
                gt_delta_h = torch.log(gt_h1 / gt_h0)


                assert not gt_delta_x.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)

                assert not gt_delta_y.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)

                assert not gt_delta_w.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)

                assert not gt_delta_h.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)


                gt_deltas = torch.stack([gt_delta_x, gt_delta_y, gt_delta_w, gt_delta_h], dim=1)
                if len(gt_deltas) > 0:
                    instances_0.gt_deltas = gt_deltas

            # Saving instances
            dataset_dict["instances_0"] = instances_0
            dataset_dict["instances_1"] = instances_1
            dataset_dict["matches_0"] = matches_0
            dataset_dict["matches_1"] = matches_1


        else:
            print("ERR: " , dataset_dict.keys() , " - ", next_frame_dict)

        return dataset_dict



class TwoFrameDatasetMapperTest:

    """
    The callable currently does the following:

    1. Read the images from "file_name_1" and "file_name_2
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):

        """

        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """

        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"

        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):

        augs = utils.build_augmentation(cfg, is_train)

        print("INSIDE FROM CONFIG: AUGS =>>>>>>", augs)

        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:

            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # Load images
        image_0 = utils.read_image(dataset_dict["file_name_0"], format=self.image_format)
        utils.check_image_size(dataset_dict, image_0)

        # Process the first image
        aug_input_0 = T.AugInput(image_0, sem_seg=None)
        transforms_0 = self.augmentations(aug_input_0)
        image_0 = aug_input_0.image
        image_shape = image_0.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image_0"] = torch.as_tensor(np.ascontiguousarray(image_0.transpose(2, 0, 1)))

        if "file_name_1" in dataset_dict:
            image_1 = utils.read_image(dataset_dict["file_name_1"], format=self.image_format)
            utils.check_image_size(dataset_dict, image_1)

            # Process the second image, but this time, use the previous transforms.
            image_1 = transforms_0.apply_image(image_1)
            dataset_dict["image_1"] = torch.as_tensor(np.ascontiguousarray(image_1.transpose(2, 0, 1)))

        is_processed = False
        if "annotations_0" in dataset_dict:
            is_processed = True
            # USER: Modify this if you want to keep them for some reason.

            for anno in dataset_dict["annotations_0"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos_0 = [utils.transform_instance_annotations(obj, transforms_0, image_shape,
                                                          keypoint_hflip_indices=self.keypoint_hflip_indices)
                for obj in dataset_dict.pop("annotations_0")
                if obj.get("iscrowd", 0) == 0
            ]

            instances_0 = utils.annotations_to_instances(annos_0, image_shape, mask_format=self.instance_mask_format)

            if self.recompute_boxes:
                instances_0.gt_boxes = instances_0.gt_masks.get_bounding_boxes()
            instances_0 = utils.filter_empty_instances(instances_0)


        if is_processed and "annotations_1" in dataset_dict:

            for anno in dataset_dict["annotations_1"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            annos_1 = [utils.transform_instance_annotations(obj, transforms_0, image_shape,
                                                          keypoint_hflip_indices=self.keypoint_hflip_indices)
                for obj in dataset_dict.pop("annotations_1")
                if obj.get("iscrowd", 0) == 0
            ]

            instances_1 = utils.annotations_to_instances(annos_1, image_shape, mask_format=self.instance_mask_format)


            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.

            if self.recompute_boxes:
                instances_1.gt_boxes = instances_1.gt_masks.get_bounding_boxes()
            instances_1 = utils.filter_empty_instances(instances_1)

            # After we're done, calculate matchings etc.
            track_ids_0 = instances_0.gt_track_id
            track_ids_1 = instances_1.gt_track_id

            # Match the targets based on the index.
            matches_0, matches_1 = [], []
            gt_bboxes_0, gt_bboxes_1 = [], []


            for i, id_0 in enumerate(track_ids_0):
                is_matched = False

                for j, id_1 in enumerate(track_ids_1):
                    if id_0 == id_1:
                        matches_0.append(i)
                        matches_1.append(j)
                        gt_bboxes_0.append(instances_0.gt_boxes.tensor[i, :])
                        gt_bboxes_1.append(instances_1.gt_boxes.tensor[j, :])
                        is_matched = True

                if not is_matched:

                    gt_bboxes_0.append(torch.zeros_like(instances_0.gt_boxes.tensor[i, :]))
                    gt_bboxes_1.append(torch.zeros_like(instances_0.gt_boxes.tensor[i, :]))


            if len(gt_bboxes_0) == 0:

                dataset_dict["instances_1"] = instances_1

            else:
                gt_bboxes_0 = torch.stack(gt_bboxes_0, dim=0)
                gt_bboxes_1 = torch.stack(gt_bboxes_1, dim=0)


                # Calculate loss target.
                gt_w0 = gt_bboxes_0[:, 2] - gt_bboxes_0[:, 0] + 0.0001
                gt_h0 = gt_bboxes_0[:, 3] - gt_bboxes_0[:, 1] + 0.0001
                gt_w1 = gt_bboxes_1[:, 2] - gt_bboxes_1[:, 0] + 0.0001
                gt_h1 = gt_bboxes_1[:, 3] - gt_bboxes_1[:, 1] + 0.0001

                gt_delta_x = (gt_bboxes_1[:, 0] - gt_bboxes_0[:, 0]) / gt_w0
                gt_delta_y = (gt_bboxes_1[:, 1] - gt_bboxes_0[:, 1]) / gt_h0
                gt_delta_w = torch.log(gt_w1 / gt_w0)
                gt_delta_h = torch.log(gt_h1 / gt_h0)


                assert not gt_delta_x.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)

                assert not gt_delta_y.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)

                assert not gt_delta_w.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)

                assert not gt_delta_h.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)


                gt_deltas = torch.stack([gt_delta_x, gt_delta_y, gt_delta_w, gt_delta_h], dim=1)
                instances_0.gt_deltas = gt_deltas

                dataset_dict["instances_1"] = instances_1
                dataset_dict["matches_0"] = matches_0
                dataset_dict["matches_1"] = matches_1

        # Saving instances
        dataset_dict["instances_0"] = instances_0

        return dataset_dict

