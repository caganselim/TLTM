import copy
import logging
import os
from collections import OrderedDict

# For the visualization purposes
import cv2
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.transforms import ScaleTransform
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Instances
from maskgnn_utils.visualizers.uvos_visualizer import UVOSVisualizer

class UVOSWriter(DatasetEvaluator):
    """
    Evaluate tracking performance for two consecutive frames
    """

    def __init__(self,
                 dataset_name,
                 distributed=True,
                 output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.

                2. "results.json" a json file which includes results.

        """

        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self.dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)
        self.untracked_idx = 255
        self.frame_cnt = 0
        self.save_gt = True

    def save_visuals(self, filename, gt=None, preds=None, gt_ids=None, pred_ids=None):
        """
        This function visualizes the matchings between ground truth
        instances with the predicted instances for a single object only.

        :param filename: (str) path to the image
        :param gt: (Instances)
        :param preds: (Instances)
        :param gt_ids: (list[int])
        :param pred_ids: (list[int])
        :return:
        """
        #self._logger.warning(f"[UVOSWriter] Saving visualizations - {filename}.")
        tokens = filename.split('/')
        frame, video = tokens[-1].split('.')[0], tokens[-2]

        video_save_dir = os.path.join(self._output_dir, video)
        if not os.path.exists(video_save_dir):
            os.mkdir(video_save_dir)

        img = cv2.imread(filename)
        height, width = img.shape[0], img.shape[1]

        if self.save_gt and (gt_ids is not None) or (gt is not None):
            gt_uvos_visualizer = UVOSVisualizer(width=width, height=height)
            gt_vis = gt_uvos_visualizer.draw_gt_with_tracking_ids(gt.to("cpu"), gt_ids)
            gt_vis.save(f'{video_save_dir}/{frame}.png')

        if (preds is not None) and (pred_ids is not None):

            pred_uvos_visualizer = UVOSVisualizer(width=width, height=height)
            pred_vis = pred_uvos_visualizer.draw_preds_with_tracking_ids(preds.to("cpu"), pred_ids)
            pred_vis.save(f'{video_save_dir}/{frame}.png')

    def process(self, inputs, outputs):
        """
        Args:
            inputs: Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of thr model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """

        self.frame_cnt += 1

        for input, output in zip(inputs, outputs):

            filename = input['file_name'] if 'file_name' in input else input['file_name_0']
            # Read these two images.
            # dict_keys(['pred_boxes', 'scores', 'pred_classes', 'locations', 'pred_deltas',
            # 'mask_features', 'pred_masks', 'mask_scores', 'current_states', 'next_states', 'tracking_id'])

            pred_instances = output['instances']


            if len(pred_instances) > 20 or torch.sum(pred_instances.tracking_id > 19):
                self._logger.warning(f"[UVOSWriter] More than 20 dets in frame: {filename}")
                keep = pred_instances.tracking_id < 20
                pred_instances = pred_instances[keep]


            # Maybe there is a neat way of doing this.

            tracking_ids = pred_instances.tracking_id
            tracking_ids = [id + 1 for id in tracking_ids]

            self.save_visuals(filename, preds=pred_instances, pred_ids=tracking_ids)


    def evaluate(self, img_ids=None):

        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset

        """
        if self.frame_cnt == 0:
            self._logger.warning("[UVOSWriter] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        self._logger.warning("[UVOSWriter] Completed the writing process.")

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
