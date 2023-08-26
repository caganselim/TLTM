import copy
import logging
import os
from collections import OrderedDict
import pycocotools.mask as rletools

# For the visualization purposes
import cv2
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.transforms import ScaleTransform
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Instances
from maskgnn_utils.visualizers.uvos_visualizer import UVOSVisualizer

class KITTIMOTSWriter(DatasetEvaluator):

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

            self._logger.warning(f"[KITTIMOTSWriter] Processing frame: {self.frame_cnt}")

            # Read these two images.
            # dict_keys(['pred_boxes', 'scores', 'pred_classes', 'locations', 'pred_deltas',
            # 'mask_features', 'pred_masks', 'mask_scores', 'current_states', 'next_states', 'tracking_id'])

            pred_instances = output['instances']
            tracking_ids = pred_instances.tracking_id
            tracking_ids = [id + 1 for id in tracking_ids]

            width = input["width"]
            height = input["height"]


            filename = input['file_name'] if 'file_name' in input else input['file_name_0']
            tokens = filename.split('/')
            frame, video = tokens[-1].split('.')[0], tokens[-2]
            video_out_pth = f'{self._output_dir}/{video}.txt'
            if not os.path.isfile(video_out_pth):
                with open(video_out_pth, 'w') as f:
                    f.write('')

            file1 = open(video_out_pth, "a")  # append mode

            for i in range(len(pred_instances)):

                pred = pred_instances[i]

                # time_frame id class_id img_height img_width rle

                class_id = int(pred.pred_classes.item())

                if class_id == 0:
                    # person => to mots
                    class_id = 2
                elif class_id:
                    # person => to mots
                    class_id = 1

                tracking_id = pred.tracking_id

                id = int(class_id*1000 + tracking_id[0])

                pred_mask = np.array(pred.pred_masks.cpu().numpy()).transpose(1,2,0)

                rle = rletools.encode(np.asfortranarray(pred_mask))[0]["counts"].decode("ascii")

                det = f"{int(frame)} {id} {class_id} {height} {width} {rle}"

                file1.write(f"{det}\n")

            file1.close()


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
