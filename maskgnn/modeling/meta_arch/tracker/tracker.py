from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from detectron2.structures import Instances
from filterpy.kalman import KalmanFilter
from .utils import iou, linear_assignment, pairwise_distance_matrix, iou_batch, convert_bbox_to_z, convert_x_to_bbox
import torch

class InstanceTracker:

    def __init__(self, instance, id, kalman_on=True):

        self.instance = instance # A detectron instance object
        self.id = id # Tracking id
        self.kalman_on = kalman_on

        # The parameters for tracker status.
        self.hits = 0
        self.hit_streak = 0 # If an object hit
        self.age = 0
        self.time_since_update = 0 # Ideally 0, incremented by one after each step (if no match)

        if self.kalman_on: # define constant velocity model

            bbox = instance.pred_boxes.tensor.cpu().numpy()[0]

            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array(
                [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
            self.kf.H = np.array(
                [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

            self.kf.R[2:, 2:] *= 10.
            self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
            self.kf.P *= 10.
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            self.kf.x[:4] = convert_bbox_to_z(bbox)
            self.history = []

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        assert self.kalman_on, "Kalman filtering must be on!"

        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def update(self, instance):

        """
        Updates the tracker state.
        """

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.instance = instance

        bbox = instance.pred_boxes.tensor.cpu().numpy()[0]
        self.kf.update(bbox)

    def get_class_id(self):
        return self.instance.pred_classes

    def get_instance(self):
        """
        Returns the mask.
        """
        self.time_since_update += 1

        i = self.instance

        i.set("tracking_id", torch.tensor([self.id]))

        return i

    def get_obj_feat(self):

        return self.instance.obj_feats

    def get_id(self):
        return self.id

    def get_mask(self):
        return self.instance.pred_masks

    def get_bbox(self):
        return self.instance.pred_boxes.tensor


class GenericTracker:

    """
    A SORT (Simple Online Realtime Tracking) based mask tracker.
    """

    def __init__(self, mode="boxiou", max_age=5, min_hits=3, iou_threshold=0, use_class_info=False):

        # This is required to compile the results.
        self.width = -1
        self.height = -1

        # Starts from object cnt.
        self.id_counter = 1

        # Minimum IOU for match.
        self.iou_threshold = iou_threshold
        self.similarity_threshold = -3

        # Maximum number of frames to keep alive a track without associated detections.
        self.max_age = max_age

        # Minimum number of associated detections before track is initialised.
        self.min_hits = min_hits

        # Sets the mode of the tracker. It can be: maskiou, boxiou, gnn
        self.mode = mode

        # Set the trackers.
        self.trackers = []

        # Shows how many frames that we have processed.
        self.frame_count = 0

        # Threshold to reject a track.
        self.threshold = self.iou_threshold if mode == "maskiou" else self.similarity_threshold

        self.remove = False

        # Flag for making use of class information while matching.
        # When this thing is on, the cost is maximized.
        self.use_class_info = use_class_info

    def reset(self):

        self.id_counter = 0
        self.trackers = []

    def get_obj_features(self):

        # Collate everything.

        if len(self.trackers) == 0:
            return None
        else:
            vectors = [tr.get_obj_feat() for tr in self.trackers]
            vectors = torch.cat(vectors, dim=0)
            return vectors

    def update(self, dets):

        self.frame_count += 1
        self.height, self.width = dets.image_size

        # If there are no predicted instances, directly return the result.
        if len(dets) == 0:
            return self.get_result()

        # Associate detections to trackers.
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets)

        # Create and initialize new trackers for unmatched detections

        # update matched trackers with assigned detections
        print("matched: ", matched)
        print("unmatched dets: ", unmatched_dets)
        print("unmatched_trks: ", unmatched_trks)

        # Update the tracker based on the matches.
        for m in matched:
            self.trackers[int(m[1])].update(dets[int(m[0])])

        # Invoke new tracks
        for _ in unmatched_dets:
            det = dets[0]
            trk = InstanceTracker(det, self.id_counter)
            self.id_counter += 1
            self.trackers.append(trk)


        # Filter previous tracks, remove dead tracklets.
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                # indexing will change? no ,because of reverse
                self.trackers.pop(i)

        return self.get_result()


    def calc_boxiou(self, new_instances):

        # Fills going to be matched or unmatched.
        prev_boxes = []
        for tr in self.trackers:
            prev_boxes.append(tr.get_bbox().cpu().numpy())
        prev_boxes = np.concatenate(prev_boxes)

        # Calc IoU and match based on that. => expects (N, H, W) as the inputs
        new_boxes = new_instances.pred_boxes.tensor.cpu().numpy()
        print("prev_boxes: ", prev_boxes.shape, " new_boxes: ", new_boxes.shape)
        cost_matrix = iou_batch(new_boxes, prev_boxes)

        return cost_matrix

    def calc_kalman(self, new_instances):

        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        prev_boxes = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        new_boxes = new_instances.pred_boxes.tensor.cpu().numpy()
        cost_matrix = iou_batch(new_boxes, prev_boxes)

        return cost_matrix


    def associate_detections_to_trackers(self, new_instances):


        # If there are no initial detections, return a list of unmatched_detections.

        if (len(self.trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(new_instances)), np.empty((0, 5), dtype=int)

        num_prev_instances = len(self.trackers)
        num_new_instances = len(new_instances)


        if self.mode == "boxiou":
            cost_matrix = self.calc_boxiou(new_instances)


        elif self.mode == "kalman":

            cost_matrix = self.calc_kalman(new_instances)


        else:
            raise NotImplementedError


        # Set cost matrix pairs w.r.t. class info. Increase the cost when there is no match.
        if self.use_class_info:

            class_ids_0 = [tr.get_class_id().item() for tr in self.trackers]
            class_ids_1 = new_instances.pred_classes

            for i, class_id_0 in enumerate(class_ids_0):
                for j, class_id_1 in enumerate(class_ids_1):
                    if class_id_0 != class_id_1:
                        # Setting to infty (-np.inf) gives cost matrix infeasible error.
                        cost_matrix[j, i] = -1000000

        if min(cost_matrix.shape) > 0:
            a = (cost_matrix > self.threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-cost_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))

        print("COST MATRIX SHAPE: ", cost_matrix.shape, " matched_indices: ", matched_indices)
        print(matched_indices)

        # Process unmatched detections and indices.


        unmatched_detections = []
        for d in range(num_new_instances):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(num_prev_instances):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if (cost_matrix[m[0], m[1]] < self.threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def get_result(self):

        """

        This function is used to return the detection results, together with the assigned IDs.
        It aggregates instances from the instance trackers depending on the activeness.

        :return: List(Instances)
        """

        instances = []

        for trk in self.trackers:

            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                inst = trk.get_instance()
                instances.append(inst)

        if len(instances) != 0:
            return Instances.cat(instances)

        else:

            instances = Instances(image_size=(int(self.height), int(self.width)))
            instances.set("pred_boxes", torch.zeros(0,4))
            return instances
