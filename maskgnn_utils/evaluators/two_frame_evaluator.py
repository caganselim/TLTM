import copy
import logging
from collections import OrderedDict
import torch

from detectron2.data import MetadataCatalog
from detectron2.structures import pairwise_iou, Instances
from detectron2.evaluation.evaluator import DatasetEvaluator
from maskgnn_utils.visualizers.instance_visualizer import InstanceVisualizer

from detectron2.data.transforms import ScaleTransform

# For the visualization purposes
import cv2
import matplotlib.pyplot as plt
import numpy as np


def resize_gt(results, output_height, output_width):
    if isinstance(output_width, torch.Tensor):
        output_width_tmp = output_width.float()
    else:
        output_width_tmp = output_width

    if isinstance(output_height, torch.Tensor):
        output_height_tmp = output_height.float()
    else:
        output_height_tmp = output_height

    prev_size_x = results.image_size[1]
    prev_size_y = results.image_size[0]
    scale_x, scale_y = (output_width_tmp / results.image_size[1], output_height_tmp / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("gt_boxes"):
        output_boxes = results.gt_boxes
        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)
        results = results[output_boxes.nonempty()]

    if results.has('gt_masks'):

        segm = results.gt_masks  # is a polygonmask object.
        # TODO this works flying mnist only! catched a bug
        rescale = ScaleTransform(prev_size_y, prev_size_x, 512, 768)

        for idx, polygon in enumerate(segm.polygons):
            polygons_temp = [np.asarray(p).reshape(-1, 2) for p in polygon]
            segm.polygons[idx] = [p.reshape(-1) for p in rescale.apply_polygons(polygons_temp)]

    return results


class TrackingDeltaEvaluator(DatasetEvaluator):
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

        self.debug_mode = False
        self.frame_cnt = 0

        """
        This is used to match GT instances with the predicted instances.
        If the IoU is higher than 0.2, a GT instance **can** be assigned
        with a predicted instance. 
        """

        self.matching_threshold = 0.2
        self.visualization_on = True

        self.track_error = 0
        self.degree_error = 0



    def reset(self):

        self.track_error = 0
        self.track_cnt = 0.0001

    def match_instances_with_gt(self, gt_instances, pred_instances):

        """
        This function matches ground-truth instances that are presented in the dataset
        with the predicted instances comes from the Mask-GNN output.
        Returns the lists of GT IDs & Pred IDs, which is given in the

        ground truth images (are shown by the palette).
        :return:
        """

        # Extract gt boxes.
        gt_boxes = gt_instances.gt_boxes

        # Move predicted boxes to cpu
        pred_boxes = pred_instances.pred_boxes
        pred_boxes.tensor = pred_boxes.tensor.cpu()

        """
        Calculates pairwise IoU.
        Gets N => len(pred_boxes) and M => len(gt_boxes) bounding boxes
        Return NxM tensor of pairwise IoUs, a.k.a match quality matrix.
        """

        match_quality_matrix = pairwise_iou(pred_boxes, gt_boxes)

        # print("pred_boxes: " , pred_boxes)
        # print("gt_boxes: " , gt_boxes)
        # print("match_quality_matrix: " , match_quality_matrix)
        # print("match_quality_matrix: ", match_quality_matrix.shape)

        pred_ids = []
        gt_ids = [gt_track_id.item() for gt_track_id in gt_instances.gt_track_id]

        """
        Here, we are evaluating match_quality matrix. Row by row, we will be processing each prediction.
        If all of the IoUs with GT is 0, we assign the untracked index. Otherwise, we get the maximum IoU
        then assign if the IoU is higher than the matching threshold.
        
        Returns the lists of GT IDs, which is given in the ground truth images (are shown by the palette).
        
        """

        for idx, row in enumerate(match_quality_matrix):

            if row.sum() == 0:

                pred_ids.append(self.untracked_idx)

            else:

                max_idx = torch.argmax(row)
                max_iou = row[max_idx]

                if max_iou > self.matching_threshold:

                    pred_ids.append(gt_ids[max_idx.item()])

                else:

                    pred_ids.append(self.untracked_idx)

        return gt_ids, pred_ids

    def process(self, inputs, outputs):
        """
        Args:
            inputs: Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """

        self.frame_cnt += 1
        outputs_0, outputs_1 = outputs

        for input, output_0, output_1 in zip(inputs, outputs_0, outputs_1):

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Read these two images.
            img_0, img_1 = cv2.imread(input['file_name_0']), cv2.imread(input['file_name_1'])

            gt_instances_0 = input['instances_0']
            pred_instances_0 = output_0['instances']

            im_size = pred_instances_0._image_size
            gt_instances_0 = resize_gt(gt_instances_0, im_size[0], im_size[1]) # Maybe there is a neat way of doing this.
            gt_ids_0, pred_ids_0 = self.match_instances_with_gt(gt_instances_0, pred_instances_0)
            pred_deltas = pred_instances_0.pred_deltas.cpu()

            if not gt_instances_0.has('gt_deltas'):
                return

            gt_deltas = gt_instances_0.gt_deltas.cpu()
            pred_bboxes_0 = pred_instances_0.pred_boxes

            # Draw arrows here!
            for i, gt_id in enumerate(gt_ids_0):
                for j, pred_id in enumerate(pred_ids_0):
                    if gt_id == pred_id:

                        if gt_deltas[i].isnan().sum() > 0:
                            continue

                        self.track_cnt += 1
                        # We got a match.

                        pred_bbox_0 = pred_bboxes_0[j].tensor

                        width = pred_bbox_0[:, 2] - pred_bbox_0[:, 0]
                        height = pred_bbox_0[:, 3] - pred_bbox_0[:, 1]

                        center_x = pred_bbox_0[:, 0] + width / 2.
                        center_y = pred_bbox_0[:, 1] + height / 2.

                        pred_delta_x = width * pred_deltas[j][0]
                        pred_delta_y = height * pred_deltas[j][1]

                        gt_delta_x = width * gt_deltas[i][0]
                        gt_delta_y = height * gt_deltas[i][1]


                        if self.debug_mode:
                            print("pred_delta_x: ", pred_delta_x, " pred_delta_y: ", pred_delta_y)
                            print("gt_delta_x: ", gt_delta_x, " gt_delta_y: ", gt_delta_y)
                            print("pred_bboxes_0:", pred_bboxes_0[j])
                            print("-------------------")

                        # Apply arrow based visualization.
                        if self.visualization_on:
                            amplification_const = 5.
                            pred_delta_x = amplification_const * pred_delta_x
                            pred_delta_y = amplification_const * pred_delta_y
                            gt_delta_x = amplification_const * gt_delta_x
                            gt_delta_y = amplification_const * gt_delta_y

                            # Pred motion
                            cv2.arrowedLine(img_0, (int(center_x), int(center_y)),
                                            (int(center_x + pred_delta_x),
                                             int(center_y + pred_delta_y)), (0, 0, 255), 5)

                            # GT motion
                            cv2.arrowedLine(img_0, (int(center_x), int(center_y)),
                                            (int(center_x + gt_delta_x),
                                             int(center_y + gt_delta_y)), (0, 255, 0), 5)


                        pred_delta = pred_deltas[j].cpu()
                        sq_err = (pred_delta - gt_deltas[i]).pow(2).mean()
                        self.track_error += sq_err.cpu().item()

                        # Calc angle thing here.
                        vector_1 = pred_delta.numpy()
                        vector_2 = gt_deltas[i].numpy()
                        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                        dot_product = np.dot(unit_vector_1, unit_vector_2)
                        radian_angle = np.arccos(dot_product)
                        degree_angle = np.degrees(radian_angle)
                        print("Degrees: ", degree_angle)

                        if not np.isnan(degree_angle):
                            self.degree_error += degree_angle

            if self.visualization_on:

                # TODO get experiment directory
                # TODO Draw arrows after instance visualization

                self._logger.warning(f"[TrackingDeltaEvaluator] Saving visualizations - {self.frame_cnt}.jpg")

                frame_0_vis = InstanceVisualizer(img_0[:, :, ::-1], metadata=self._metadata, scale=0.5)
                frame_1_vis = InstanceVisualizer(img_1[:, :, ::-1], metadata=self._metadata, scale=0.5)
                im0_vis = frame_0_vis.draw_preds_with_tracking_ids(pred_instances_0.to("cpu"), pred_ids_0)
                im1_vis = frame_1_vis.draw_preds_with_tracking_ids(pred_instances_1.to("cpu"), pred_ids_1)

                ax1.set_title(f"Pred 0 - {self.frame_cnt}")
                ax1.imshow(cv2.cvtColor(im0_vis.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

                ax2.set_title(f"Pred 1 - {self.frame_cnt}")
                ax2.imshow(cv2.cvtColor(im1_vis.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

                fig.savefig(f'./visualizations/tracking_delta_preds/{self.frame_cnt}.png')
                plt.close('all')

    def evaluate(self, img_ids=None):

        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset

        """

        if self.frame_cnt == 0:
            self._logger.warning("[TrackingDeltaEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()

        degree_avg = self.degree_error / self.track_cnt
        tracking_mse = self.track_error / self.track_cnt

        tracker_out = f"Tracking MSE: {tracking_mse}"
        degree_avg = f"Degree Err: {degree_avg}"

        self._logger.warning(tracker_out)
        self._logger.warning(degree_avg)

        # storage = get_event_storage()
        # storage.put_scalar("tracking/mse", tracking_mse)
        # Copy so the caller can do whatever with results

        return copy.deepcopy(self._results)
