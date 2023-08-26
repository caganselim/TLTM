import logging
import numpy as np
from enum import Enum, unique
import cv2
import torch
from PIL import Image, ImageDraw
from detectron2.structures import BitMasks, Boxes, PolygonMasks, RotatedBoxes
from detectron2.utils.visualizer import GenericMask

logger = logging.getLogger(__name__)
__all__ = [ "DebugVisualizer"]

class DebugVisualizer:

    """
    Visualizer that draws data about segmentation to a blank PNG.
    """

    def __init__(self, base_img):

        """
        Args:
            img_size: width,height
            metadata (Metadata): image metadata.
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """

        self.base_img = Image.fromarray((base_img*255.).astype(np.uint8)).convert("RGBA")
        self.base_img.putalpha(255)

        width, height = self.base_img.size
        self.height = height
        self.width = width
        self.mask_img = Image.new("P", (width, height))

        # Set palette colors
        palette = np.loadtxt("./maskgnn_utils/colors.txt",dtype=np.uint8).reshape(-1,3)
        self.mask_img.putpalette(palette)
        self.mask_img._drawer = ImageDraw.Draw(self.mask_img)
        self.cpu_device = torch.device("cpu")

    def get_blended_img(self):

        # new_mask_img = self.mask_img.copy().convert('RGBA')
        # new_mask_img.putalpha(128)

        print("hellow!")

        return self.mask_img#Image.alpha_composite(self.base_img, new_mask_img)


    def draw_preds_with_tracking_ids(self, predictions, tracking_ids):

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.height, self.width) for x in masks]
        else:
            return self.get_blended_img()
        self.overlay_instances(masks=masks, tracking_ids=tracking_ids)

        return self.get_blended_img()


    def draw_gt_with_tracking_ids(self, predictions, tracking_ids):

        if predictions.has("gt_masks"):
            masks = predictions.gt_masks
            masks = [GenericMask(x, self.height, self.width) for x in masks]
        else:
            return self.get_blended_img()

        self.overlay_instances(masks=masks, tracking_ids=tracking_ids)

        return self.get_blended_img()


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
            return self.get_blended_img()

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            # reversed => sorted_idxs = np.argsort(areas).tolist()
            sorted_idxs = np.argsort(areas).tolist()
            # Re-order overlapped instances in descending order.
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            tracking_ids = [tracking_ids[idx] for idx in sorted_idxs] if tracking_ids is not None else None

        for i in range(num_instances):
            tracking_id = tracking_ids[i]
            if masks is not None:
                for segment in masks[i].polygons:
                    seg = segment.reshape(-1,2)
                    xy = [(c[0], c[1]) for c in seg]
                    self.mask_img._drawer.polygon(xy, fill=int(tracking_id))

        return self.get_blended_img()


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
        return self.get_blended_img()
