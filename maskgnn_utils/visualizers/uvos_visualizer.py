import logging
import numpy as np
from enum import Enum, unique
import cv2

import pycocotools.mask as mask_util
import torch
from PIL import Image, ImageDraw
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

logger = logging.getLogger(__name__)
__all__ = [ "UVOSVisualizer"]


class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (height, width), m.shape
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox


class UVOSVisualizer:

    """
    Visualizer that draws data about segmentation to a blank PNG.
    """

    def __init__(self, height, width):
        """
        Args:
            img_size: width,height
            metadata (Metadata): image metadata.
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """


        self.img = Image.new("P", (width, height))

        self.height = height
        self.width = width

        # Set palette colors
        palette = np.loadtxt("./maskgnn_utils/colors.txt",dtype=np.uint8).reshape(-1,3)
        self.img.putpalette(palette)
        self.img_drawer = ImageDraw.Draw(self.img)

        self.cpu_device = torch.device("cpu")

    def draw_preds_with_tracking_ids(self, predictions, tracking_ids):
        if predictions.has("pred_masks"):

            masks = np.asarray(predictions.pred_masks)
            #masks = predictions.pred_masks

            masks = [GenericMask(x, self.height, self.width) for x in masks]
        else:
            return self.img

        self.overlay_instances(masks=masks, tracking_ids=tracking_ids)

        return self.img


    def draw_gt_with_tracking_ids(self, predictions, tracking_ids):

        if predictions.has("gt_masks"):
            #masks = np.asarray(predictions.gt_masks)
            masks = predictions.gt_masks
            masks = [GenericMask(x, self.height, self.width) for x in masks]
        else:
            return self.img

        self.overlay_instances(masks=masks, tracking_ids=tracking_ids)

        return self.img


    def overlay_instances(self, *, masks=None,tracking_ids=None):
        """
        Args:
            boxes (Boxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,

            labels (list[str]): the text to be displayed for each instance.

            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.

            tracking_ids (list[int]): a list of integers, where each corresponds to a tracking id.

        Returns:
            output (PIL.Image): image object with visualizations.
        """

        num_instances = 0
        if masks is not None:
            masks = self._convert_masks(masks)
            num_instances = len(masks)

        if num_instances == 0:
            return self.img

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if masks is not None:
            areas = np.asarray([x.area() for x in masks])

        # if areas is not None:
        #     # reversed => sorted_idxs = np.argsort(areas).tolist()
        #     sorted_idxs = np.argsort(areas).tolist()
        #     # Re-order overlapped instances in descending order.
        #     masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
        #     tracking_ids = [tracking_ids[idx] for idx in sorted_idxs] if tracking_ids is not None else None

        for i in range(num_instances):
            tracking_id = tracking_ids[i]
            if masks is not None:
                for segment in masks[i].polygons:
                    seg = segment.reshape(-1,2)
                    xy = [(c[0], c[1]) for c in seg]
                    self.img_drawer.polygon(xy, fill=int(tracking_id))

        return self.img


    """
    Primitive drawing functions:
    """

    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.numpy()
        else:
            return np.asarray(boxes)

    def _convert_masks(self, masks_or_polygons):
        """
        Convert different format of masks or polygons to a tuple of masks and polygons.

        Returns:
            list[GenericMask]:
        """

        m = masks_or_polygons
        if isinstance(m, PolygonMasks):
            m = m.polygons
        if isinstance(m, BitMasks):
            m = m.tensor.numpy()
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        ret = []
        for x in m:
            if isinstance(x, GenericMask):
                ret.append(x)
            else:
                ret.append(GenericMask(x, self.height, self.width))
        return ret

    def get_output(self):
        """
        Returns:
            output (VisImage): the image output containing the visualizations added
            to the image.
        """
        return self.img
