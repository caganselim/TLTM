import copy
import json
import logging
import os
from collections import OrderedDict

# For the visualization purposes
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from pycocotools import mask
from zipfile import ZipFile

class YTVISWriter(DatasetEvaluator):

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
        self.frame_cnt = 0
        self.save_gt = True
        self.results = []

        # Temps
        self.current_video_id = -1
        self.preds = {}

        # Container for sequence lengths
        self.vid_cnts = {}

    def process(self, inputs, outputs):
        """
        Args:
            inputs: Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of thr model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """


        for input, output in zip(inputs, outputs):

            # Read these two images.
            # dict_keys(['pred_boxes', 'scores', 'pred_classes', 'locations', 'pred_deltas',
            # 'mask_features', 'pred_masks', 'mask_scores', 'current_states', 'next_states', 'tracking_id'])
            # Update

            video_id = input['video_id']
            preds = output['instances']

            # Initialize detection container if it doesn't exist.
            if not video_id in self.preds.keys():
                # Init an empty dets dict
                self.preds[video_id] = {}
                self.frame_cnt = 0


            # Process the preds.
            for i in range(len(preds)):
                pred = preds[i]
                tracking_id = int(pred.tracking_id.item())

                if not tracking_id in self.preds[video_id]:

                    # Alloc space for that
                    self.preds[video_id][tracking_id] = {}

                # Construct the record

                # Apply RLE compression.
                seg = np.asfortranarray(pred.pred_masks.cpu().numpy().transpose(1, 2, 0))
                compressed_rle = mask.encode(seg)[0]
                compressed_rle["counts"] = compressed_rle["counts"].decode('ascii')

                category_id = int(pred.pred_classes) + 1
                score = pred.scores[0].cpu()

                # Save it.
                record = {}
                record["category_id"] = category_id
                record["score"] = score
                record["rle"] = compressed_rle
                self.preds[video_id][tracking_id][self.frame_cnt] = record

            self.frame_cnt += 1

            # Container for sequence lengths
            self.vid_cnts[video_id] = self.frame_cnt

    def evaluate(self, img_ids=None):

        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset

        """
        if self.frame_cnt == 0:
            self._logger.warning("[YTVISWriter] Did not receive valid predictions.")
            return {}

        print("Preds: ", self.preds.keys())
        print("Vid cnts: ", self.vid_cnts)

        # Prep output json.
        output_json = []
        for video_id in self.vid_cnts.keys():
            self._logger.warning(f"[YTVISWriter] Dumping results for: {video_id}")

            print("===" , self.preds[video_id].keys())
            for track_id in self.preds[video_id].keys():

                preds_entire_vid = self.preds[video_id][track_id]
                seq_len = self.vid_cnts[video_id]

                segs = []
                scores = []
                category_id = []

                for i in range(seq_len):

                    if i in preds_entire_vid.keys():

                        pred = preds_entire_vid[i]
                        segs.append(pred["rle"])
                        scores.append(pred["score"])
                        category_id.append(pred["category_id"])
                    else:
                        segs.append(None)

                rec = {}
                rec["video_id"] = video_id
                rec["score"] = float(torch.mean(torch.tensor(scores)).item())
                rec["category_id"] = int(torch.mode(torch.tensor(category_id))[0])
                rec["segmentations"] = segs
                output_json.append(rec)


        json_save_pth = os.path.join(self._output_dir, 'results.json')
        with open(json_save_pth, 'w') as json_file:
            json.dump(output_json, json_file)


        # Copy so the caller can do whatever with results
        return copy.deepcopy(OrderedDict())
