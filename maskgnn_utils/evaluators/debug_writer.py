import copy
import logging
import os
from collections import OrderedDict
import torch
import cv2

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
import matplotlib.pyplot as plt

# For the visualization purposes
import numpy as np
from detectron2.modeling import detector_postprocess
from detectron2.structures import Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from maskgnn_utils import InstanceVisualizer


# Importing the PIL library
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

np.set_printoptions(precision=4)

def render_matrix(torch_tensor, font_size=50):
    # Open an Image
    img = Image.new(mode="RGB", size=(1280, 360), color=(255, 255, 255))

    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)

    # Custom font style and font size
    myFont = ImageFont.truetype('FreeMono.ttf', font_size)

    # Add Text to an image

    I1.text((10, 10), np.array(torch_tensor).__str__(), font=myFont, fill=(255, 0, 0))

    return img

def render_info(n_curr_dets, n_prev_dets, frame, video):
    # Open an Image
    img = Image.new(mode="RGB", size=(1280, 360), color=(255, 255, 255))

    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)

    # Custom font style and font size
    myFont = ImageFont.truetype('FreeMono.ttf', 50)

    # Add Text to an image

    I1.text((10, 10), f"Number of current detections: {n_curr_dets}", font=myFont, fill=(255, 0, 0))
    I1.text((10, 100), f"Number of previous detections: {n_prev_dets}", font=myFont, fill=(255, 0, 0))
    I1.text((10, 200), f"Video: {video} Frame: {frame}", font=myFont, fill=(255, 0, 0))

    return img


def render_info_2(matches, assignments):
    # Open an Image
    img = Image.new(mode="RGB", size=(1280, 360), color=(255, 255, 255))

    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)

    # Custom font style and font size
    myFont = ImageFont.truetype('FreeMono.ttf', 50)

    # Add Text to an image

    I1.text((10, 10), f"Matches: {matches}", font=myFont, fill=(255, 0, 0))
    I1.text((10, 100), f"Assignments: {assignments}", font=myFont, fill=(255, 0, 0))

    return img

def to_img(img, pixel_mean=np.array([103.53, 116.28, 123.675]), pixel_std=np.array([1.0, 1.0, 1.0])):

    img = img.cpu().numpy().transpose(1, 2, 0)

    print(pixel_mean.shape)

    out = img*pixel_std #+ pixel_mean.reshape(1 ,1, 3)

    return out/255.


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

        segm = results.gt_masks.tensor  # is a polygonmask object.
        # rescale = ScaleTransform(prev_size_y, prev_size_x, output_height, output_width)
        #
        print(segm[:,None,:,:].shape)
        # segm = rescale.apply_segmentation(segm)
        segm = torch.nn.functional.interpolate(segm[:,None,:,:].to(torch.float32),size=(output_height, output_width))
        results.set("gt_masks", BitMasks(segm[:,0,:,:].to(torch.bool)))

    return results


class DebugWriter(DatasetEvaluator):
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


        # Container für previous detections.
        self.prev_preds = None
        self.prev_pred_ids = None

        print("=>>>>>>>>>>>>>>>>>>>>>>>>>>> DEBUG WRITER")

    def save_visuals(self, input, gt=None, preds=None, gt_ids=None, pred_ids=None, debug_dict=None):
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

        filename = input["file_name"] if "file_name" in input else input["file_name_0"]


        self._logger.warning(f"[DebugWriter] Saving visualizations - {filename}.")

        tokens = filename.split('/')
        frame, video = tokens[-1].split('.')[0], tokens[-2]

        video_save_dir = os.path.join(self._output_dir, video)
        if not os.path.exists(video_save_dir):
            os.mkdir(video_save_dir)

        image_0 = cv2.imread(filename)[:,:,::-1]
        height, width, _ = image_0.shape

        # Draw.
        metadata = MetadataCatalog.get("coco")
        visualizer = InstanceVisualizer(image_0, metadata=metadata)

        if preds.has("pred_masks"):
            print("preds: ", len(preds))
            print("pred masks: " , preds.pred_masks.shape)
            print("pred masks sum: " , preds.pred_masks.sum())
        else:
            print("NO PREDICTED MASKS!")

        pred_ids = [id.item() for id in pred_ids]

        im0_vis = visualizer.draw_preds_with_tracking_ids(preds.to("cpu"), pred_ids)
        vis0 = im0_vis.get_image()


        if self.prev_preds is None:

            fig, axs = plt.subplots(1,1, figsize=(10,10))
            axs.imshow(vis0)
            axs.set_title("Frame 1")
            fig.savefig(os.path.join(video_save_dir, frame))
            plt.close(fig)

        else:

            prev_pred_ids = [id.item() for id in self.prev_pred_ids]



            visualizer1 = InstanceVisualizer(self.prev_image, metadata=metadata)
            im1_vis = visualizer1.draw_preds_with_tracking_ids(self.prev_preds.to("cpu"), prev_pred_ids)
            vis_prev = im1_vis.get_image()

            fig, axs = plt.subplots(3, 4, figsize=(40, 20))

            # ================= Drawing Preds =================

            gs = axs[0, 0].get_gridspec()
            # remove the underlying axes
            for ax in axs[0, 0:2]:
                ax.remove()
            axbig = fig.add_subplot(gs[0, 0:2])
            axbig.imshow(vis_prev)
            axbig.set_title("Prev Preds", fontsize=30)
            axbig.autoscale_view('tight')

            # ============= End of Drawing Preds ==============


            # ================= Drawing Prev. =================


            gs = axs[0, 2].get_gridspec()
            # remove the underlying axes
            for ax in axs[0, 2:]:
                ax.remove()
            axbig = fig.add_subplot(gs[0, 2:])
            axbig.imshow(vis0)
            axbig.set_title("Current Preds", fontsize=30)
            axbig.autoscale_view('tight')

            # ============= End of Drawing GT ==============

            # axs[0][2].imshow(np.asarray(render_matrix(debug_dict["cumulative_score"])))
            # axs[0][2].set_title("Cumulative Score")


            axs[1][0].set_title("Match LL * 1.0", fontsize=20)
            axs[1][0].get_xaxis().set_visible(False)
            axs[1][0].get_yaxis().set_visible(False)
            axs[1][0].set_frame_on(False)
            axs[1][0].autoscale_view('tight')
            if debug_dict is not None:
                axs[1][0].imshow(np.asarray(render_matrix(debug_dict["match_ll"], font_size=30)))

            axs[1][1].set_title("Bbox IoUs * 2.0", fontsize=20)
            axs[1][1].get_xaxis().set_visible(False)
            axs[1][1].get_yaxis().set_visible(False)
            axs[1][1].set_frame_on(False)
            axs[1][1].autoscale_view('tight')
            if debug_dict is not None:
                axs[1][1].imshow(np.asarray(render_matrix(debug_dict["bbox_ious"])))

            axs[1][2].set_title("Label Delta * 10.0", fontsize=20)
            axs[1][2].get_xaxis().set_visible(False)
            axs[1][2].get_yaxis().set_visible(False)
            axs[1][2].set_frame_on(False)
            axs[1][2].autoscale_view('tight')
            if debug_dict is not None:
                axs[1][2].imshow(np.asarray(render_matrix(debug_dict["label_delta"])))

            axs[1][3].set_title("Bbox Scores * 1.0", fontsize=20)
            axs[1][3].get_xaxis().set_visible(False)
            axs[1][3].get_yaxis().set_visible(False)
            axs[1][3].set_frame_on(False)
            axs[1][3].autoscale_view('tight')

            if debug_dict is not None:
                axs[1][3].imshow(np.asarray(render_matrix(debug_dict["bbox_scores"])))
            # axs[2][0].set_title("Cumulative Scores", fontsize=20)
            # axs[2][0].imshow(np.asarray(render_matrix(debug_dict["cumulative_score"])))
            # axs[2][0].get_xaxis().set_visible(False)
            # axs[2][0].get_yaxis().set_visible(False)
            # axs[2][0].set_frame_on(False)
            # axs[2][0].autoscale_view('tight')


            gs = axs[2, 0].get_gridspec()
            # remove the underlying axes
            for ax in axs[2, 0:2]:
                ax.remove()
            axbig = fig.add_subplot(gs[2, 0:2])

            if debug_dict is not None:
                axbig.imshow(np.asarray(render_matrix(debug_dict["cumulative_score"], font_size=30)))


            axbig.set_title("Cumulative Score", fontsize=30)
            axbig.get_xaxis().set_visible(False)
            axbig.get_yaxis().set_visible(False)
            axbig.set_frame_on(False)
            axbig.autoscale_view('tight')



            axs[2][2].set_title("Assignments ", fontsize=20)

            if debug_dict is not None:
                axs[2][2].imshow(np.asarray(render_info_2(matches=debug_dict["match_ids"], assignments=debug_dict["det_obj_ids"])))

            axs[2][2].get_xaxis().set_visible(False)
            axs[2][2].get_yaxis().set_visible(False)
            axs[2][2].set_frame_on(False)
            axs[2][2].autoscale_view('tight')

            axs[2][3].set_frame_on(False)
            axs[2][3].set_title("Metadata ", fontsize=20)

            if debug_dict is not None:

                axs[2][3].imshow(np.asarray(render_info(debug_dict["n_curr_dets"],
                                                        debug_dict["n_prev_dets"],
                                                        frame, video )))
            axs[2][3].autoscale_view('tight')
            axs[2][3].get_xaxis().set_visible(False)
            axs[2][3].get_yaxis().set_visible(False)

            # Export matplotlib figure to a NumPy array.
            fig.tight_layout(pad=0)
            canvas = plt.gca().figure.canvas
            canvas.draw()
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            image = data.reshape(canvas.get_width_height()[::-1] + (3,))
            height, width, _ = image.shape

            # Convert numpy-array version of the Canvas to a PIL image for further drawing
            im = Image.fromarray(image)

            # Call draw Method to add 2D graphics in an image
            I1 = ImageDraw.Draw(im)

            # Custom font style and font size
            myFont = ImageFont.truetype('FreeMono.ttf', 50)

            # Add Text to an image
            cur_preds_str = np.array(pred_ids)[None].T.__str__()[1:-1].replace("\n", "\n\n").replace(" ", "").replace("[", "").replace("]", "")
            I1.text((width/2 - 250, 100), cur_preds_str, font=myFont, fill=(255, 0, 0))
            I1.text((width/2 - 360, 170), "Curr\ntrack\nids", font=ImageFont.truetype('FreeMono.ttf', 30), fill=(0, 0, 255))


            I1.text((width/2 - 140, 10), "Prev track ids", font=ImageFont.truetype('FreeMono.ttf', 30), fill=(0, 0, 255))
            prev_preds_str = np.array([-1] + prev_pred_ids).__str__()[1:-1]
            I1.text((width/2 - 200, 50), prev_preds_str, font=myFont, fill=(255, 0, 0))

            x0 = width/2 - 200
            x1 = x0 + 400
            y0 = 100
            y1 = 400

            I1.rectangle([x0, y0, x1, y1], outline=(0,0,0), fill=(255, 255, 255), width=2)

            w = int(((x1 - x0) / 2)*0.6)
            h = int(((y1 - y0) / 2)*0.6)

            I1.text((x0+w, y0+h), "COST\nMATRIX", font=myFont, fill=(0, 255, 0))

            im.save(os.path.join(video_save_dir, frame) + ".png" )

            plt.close(fig)

        self.prev_image = image_0





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

            """
            Input has the following keys:
            reading input:  dict_keys(['file_name', 'height', 'width', 'image_id', 'annotations',
             'instances_0', 'instances_1', 'matches_0', 'matches_1', 'image_0', 'image_1'])
             
            Output has the only key: dict_keys(['instances'])            
             
            """

            self._logger.warning(f"[DebugWriter] Processing frame: {self.frame_cnt}")


            # Read these two images.
            pred_instances = output['instances']
            tracking_ids = pred_instances.tracking_id
            tracking_ids = [id + 1 for id in tracking_ids]

            if "instances_0" in input.keys():
                gt_instances = input["instances_0"]

                gt_tracking_ids = gt_instances.gt_track_id
                gt_tracking_ids = [id + 1 for id in gt_tracking_ids]

            else:
                gt_instances = None
                gt_tracking_ids = None

            debug_dict = output["debug_dict"]

            self.save_visuals(input=input, preds=pred_instances, pred_ids=tracking_ids, gt=gt_instances, gt_ids=gt_tracking_ids, debug_dict=debug_dict)

            # Save prevs
            self.prev_preds = pred_instances
            self.prev_pred_ids = tracking_ids

    def evaluate(self, img_ids=None):

        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset

        """
        if self.frame_cnt == 0:
            self._logger.warning("[DebugWriter] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        self._logger.warning("[DebugWriter] Completed the writing process.")

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
