import imgaug
import imgaug.augmenters as iaa
import numpy as np

from datetime import datetime

import torch
from detectron2.structures import BitMasks, Boxes, BoxMode
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import pycocotools.mask as mask_util

class ImageToSeqAugmenter(object):
    def __init__(self, perspective=False,affine=True, motion_blur=True,
                 brightness_range=(-50, 50), hue_saturation_range=(-15, 15), perspective_magnitude=0.12,
                 scale_range=1.0, translate_range={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, rotation_range=(-20, 20),
                 motion_blur_kernel_sizes=(7, 9), motion_blur_prob=0.5, identity_mode=False, seed_override=None):


        self.identity_mode = identity_mode
        self.seed_override = seed_override

        if self.seed_override is None:
            seed = int(datetime.now().strftime('%M%S%f')[-8:])
        else:
            seed = self.seed_override

        imgaug.seed(seed)


        self.basic_augmenter = iaa.SomeOf((1, None), [
                iaa.Add(brightness_range),
                iaa.AddToHueAndSaturation(hue_saturation_range)
            ]
        )

        transforms = []
        if perspective:
            transforms.append(iaa.PerspectiveTransform(perspective_magnitude))
        if affine:
            transforms.append(iaa.Affine(scale=scale_range,
                                         translate_percent=translate_range,
                                         rotate=rotation_range,
                                         order=1,  # cv2.INTER_LINEAR
                                         backend='auto'))
        transforms = iaa.Sequential(transforms)
        transforms = [transforms]

        if motion_blur:
            blur = iaa.Sometimes(motion_blur_prob, iaa.OneOf(
                [
                    iaa.MotionBlur(ksize)
                    for ksize in motion_blur_kernel_sizes
                ]
            ))
            transforms.append(blur)

        self.frame_shift_augmenter = iaa.Sequential(transforms)

    @staticmethod
    def condense_masks(instance_masks):
        condensed_mask = np.zeros_like(instance_masks[0], dtype=np.int8)
        for instance_id, mask in enumerate(instance_masks, 1):
            condensed_mask = np.where(mask, instance_id, condensed_mask)

        return condensed_mask

    @staticmethod
    def expand_masks(condensed_mask, num_instances):
        return [(condensed_mask == instance_id).astype(np.uint8) for instance_id in range(1, num_instances + 1)]

    def __call__(self, image, objs=None):

        if self.identity_mode:

            return image, objs


        det_augmenter = self.frame_shift_augmenter.to_deterministic()

        if objs:

            bbox_list = []
            masks_np, is_binary_mask = [], []

            for obj in objs:

                box_in = obj["bbox"]

                # Save bounding-boxes
                box = imgaug.augmentables.BoundingBox(x1=box_in[0],
                                                      y1=box_in[1],
                                                      x2=box_in[0] + box_in[2],
                                                      y2=box_in[1] + box_in[3])
                bbox_list.append(box)

                mask = mask_util.decode(obj["segmentation"])
                masks_np.append(mask.astype(np.bool))
                is_binary_mask.append(True)

            num_instances = len(masks_np)
            masks_np = SegmentationMapsOnImage(self.condense_masks(masks_np), shape=image.shape[:2])
            bbox_list_out = imgaug.augmentables.BoundingBoxesOnImage(bbox_list, shape=image.shape[:2])

            # to keep track of which points in the augmented image are padded zeros, we augment an additional all-ones
            # array. The problem is that imgaug will apply one augmentation to the image and associated mask, but a
            # different augmentation to this array. To prevent this, we manually seed the rng in imgaug before both
            # function calls to ensure that the same augmentation is applied to both.

            seed = int(datetime.now().strftime('%M%S%f')[-8:])
            imgaug.seed(seed)
            aug_image, aug_masks, aug_bboxes = det_augmenter(image=self.basic_augmenter(image=image),
                                                 segmentation_maps=masks_np, bounding_boxes=bbox_list_out)

            aug_bboxes_list = [[bbox.x1, bbox.y1, bbox.x2-bbox.x1, bbox.y2-bbox.y1] for bbox in aug_bboxes]


            imgaug.seed(seed)
            invalid_pts_mask = det_augmenter(image=np.ones(image.shape[:2] + (1,), np.uint8)).squeeze(2)

            aug_masks = self.expand_masks(aug_masks.get_arr(), num_instances)

            #aug_masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in aug_masks]))


            for i in range(len(objs)):

                objs[i]["segmentation"] = aug_masks[i]
                objs[i]["bbox"] = aug_bboxes_list[i]

            return aug_image, objs #, invalid_pts_mask == 0

        else:

            aug_image = det_augmenter(image=image)

            return aug_image, objs