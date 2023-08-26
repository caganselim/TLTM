import copy
import logging
from typing import List, Optional, Union

from detectron2.config import configurable
import maskgnn_utils.dataset.mappers.map_utils as utils
from detectron2.data import transforms as T
from maskgnn_utils.dataset.mappers.imgaug_backend_rle import *
import random


__all__ = ["JointMapper"]

class JointMapper:

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
        instance_mask_format: str = "bitmask",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        coco_motion_params,
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
        self.coco_motion_params     = coco_motion_params
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

        coco_motion_params = cfg.DATASETS.COCO_MOTION_AUG


        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "coco_motion_params": coco_motion_params,
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
        if "video_id" in dataset_dict:
            dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

            #print(f'{dataset_dict["file_name_0"]} - {dataset_dict["file_name_1"]}')

            # Load images
            image_0 = utils.read_image(dataset_dict["file_name_0"], format=self.image_format)
            utils.check_image_size(dataset_dict, image_0)


            if "file_name_1" not in dataset_dict.keys():

                # Padded sampling case.

                next_frame_dict = random.choice(dataset_dict["next_frame_dicts"])
                dataset_dict["file_name_1"] = next_frame_dict["file_name_1"]
                dataset_dict["annotations_1"] = next_frame_dict["annotations_1"]


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

            if "annotations_0" in dataset_dict and "annotations_1" in dataset_dict:

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
                annos_0 = [utils.transform_instance_annotations(obj, transforms_0, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
                    for obj in dataset_dict.pop("annotations_0")
                    if obj.get("iscrowd", 0) == 0
                ]

                annos_1 = [utils.transform_instance_annotations(obj, transforms_0, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
                    for obj in dataset_dict.pop("annotations_1")
                    if obj.get("iscrowd", 0) == 0
                ]

                instances_0 = utils.annotations_to_instances(annos_0, image_shape, mask_format=self.instance_mask_format)
                instances_1 = utils.annotations_to_instances(annos_1, image_shape, mask_format=self.instance_mask_format)

            dataset_dict["instances_0"] = instances_0
            dataset_dict["instances_1"] = instances_1

        else:
            # Create the augmenter.
            augmenter = ImageToSeqAugmenter(perspective=self.coco_motion_params.PERSPECTIVE,
                                            affine=self.coco_motion_params.AFFINE,
                                            motion_blur=self.coco_motion_params.MOTION_BLUR,
                                            brightness_range=self.coco_motion_params.BRIGHTNESS_RANGE,
                                            hue_saturation_range=self.coco_motion_params.HUE_SATURATION_RANGE,
                                            perspective_magnitude=self.coco_motion_params.PERSPECTIVE_MAGNITUDE,
                                            scale_range=self.coco_motion_params.SCALE_RANGE,
                                            translate_range={"x": self.coco_motion_params.TRANSLATE_RANGE_X,
                                                            "y": self.coco_motion_params.TRANSLATE_RANGE_Y},
                                            rotation_range=self.coco_motion_params.ROTATION_RANGE,
                                            motion_blur_kernel_sizes=self.coco_motion_params.MOTION_BLUR_KERNEL_SIZES,
                                            motion_blur_prob=self.coco_motion_params.MOTION_BLUR_PROB,
                                            identity_mode=self.coco_motion_params.IDENTITY_MODE,
                                            seed_override=self.coco_motion_params.SEED_OVERRIDE)

            invalid_motion_aug = False

            dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
            annos_0 = dataset_dict['annotations']

            # Load image
            image_0 = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            utils.check_image_size(dataset_dict, image_0)

            # Hallucinate motion.
            image_1, annos_1 = augmenter(image=image_0, objs=copy.deepcopy(annos_0))

            if not self.is_train:
                dataset_dict["debug_image_0"] = image_0
                dataset_dict["debug_image_1"] = image_1


            # Process the first image
            aug_input_0 = T.AugInput(image_0, sem_seg=None)
            transforms_0 = self.augmentations(aug_input_0)
            image_0 = aug_input_0.image

            image_1 = transforms_0.apply_image(image_1)

            # Make a copy.
            image_shape = image_0.shape[:2]  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.

            if "annotations" in dataset_dict:

                # Process the first one.
                annos_0 = [utils.transform_instance_annotations(obj, transforms_0, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
                        for obj in annos_0
                        if obj.get("iscrowd", 0) == 0]


                # Process the first one.
                annos_1 = [utils.transform_instance_annotations(obj, transforms_0, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
                        for obj in annos_1
                        if obj.get("iscrowd", 0) == 0]


                instances_0 = utils.annotations_to_instances(annos_0, image_shape, mask_format=self.instance_mask_format)
                instances_1 = utils.annotations_to_instances(annos_1, image_shape, mask_format=self.instance_mask_format)
                instances_1 = utils.filter_empty_instances(instances_1, by_box=True, by_mask=False)

                if len(instances_1) == 0:
                    invalid_motion_aug = True


                # After transforms such as cropping are applied, the bounding box may no longer
                # tightly bound the object. As an example, imagine a triangle object
                # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                # the intersection of original bounding box and the cropping box.

                # Saving instances
                dataset_dict["instances_0"] = instances_0

                if invalid_motion_aug:
                    print("HIT INVALID MOTION AUG: ", dataset_dict["file_name"])
                    dataset_dict["instances_1"] = copy.deepcopy(instances_0)
                else:
                    dataset_dict["instances_1"] = instances_1


            dataset_dict["image_0"] = torch.as_tensor(np.ascontiguousarray(image_0.transpose(2, 0, 1)))
            if invalid_motion_aug:
                # Use the same frame
                dataset_dict["image_1"] = torch.as_tensor(np.ascontiguousarray(image_0.transpose(2, 0, 1)))
            else:
                dataset_dict["image_1"] = torch.as_tensor(np.ascontiguousarray(image_1.transpose(2, 0, 1)))


        return dataset_dict