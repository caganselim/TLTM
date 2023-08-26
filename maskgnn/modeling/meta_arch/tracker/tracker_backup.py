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

        # It keeps the tracker status.
        self.is_active = False
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0

        if self.kalman_on: # define constant velocity model

            bbox = instance.pred_bboxes

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

        self.is_active = True
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        self.instance = instance
        self.kf.update(convert_bbox_to_z(instance.pred_boxes))

    def get_class_id(self):
        return self.instance.pred_classes

    def get_instance(self):
        """
        Returns the mask.
        """
        self.time_since_update += 1
        return self.instance

    def get_id(self):
        return self.id

    def deactivate(self):
        self.is_active = False

    def activate(self):
        self.is_active = True

    def get_mask(self):
        return self.instance.pred_masks

    def get_current_state(self):
        return self.instance.current_states

    def get_next_state(self):
        return self.instance.next_states

    def get_bbox(self):
        return self.instance.pred_boxes.tensor


class GenericTracker:

    """
    A SORT (Simple Online Realtime Tracking) based mask tracker.
    """

    def __init__(self, mode="boxiou", max_age=5, min_hits=3, iou_threshold=0, use_class_info=True):

        # This is required to compile the results.
        self.width = -1
        self.height = -1

        # Starts from object cnt.
        self.id_counter = 0

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

    def get_active_trackers(self):

        ids = []
        time_since_updates = []
        for tr in self.trackers:
            if tr.is_active:
                ids.append(tr.get_id())
                time_since_updates.append(tr.time_since_update)

        return ids

    def get_inactive_trackers(self):

        ids = []
        for tr in self.trackers:
            if not tr.is_active:
                ids.append(tr.get_id())
        return ids

    def update(self, pred_instances):

        self.frame_count += 1
        self.height, self.width = pred_instances.image_size

        # If there are no predicted instances, directly return the result.
        if len(pred_instances) == 0:
            return self.get_result()

        # Associate detections to trackers.
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(pred_instances)

        # update matched trackers with assigned detections
        print("matched: ", matched)
        print("unmatched_masks: ", unmatched_dets)
        print("unmatched_trks: ", unmatched_trks)

        # Update the tracker based on the matches.
        for m in matched:
            self.trackers[int(m[1])].update(pred_instances[int(m[0])])

        # Create and initialise new trackers for unmatched detections.
        if self.frame_count < 100000000:

            for i in unmatched_dets:
                trk = InstanceTracker(pred_instances[i, :])
                self.trackers.append(trk)


            inactive_tracks = self.get_inactive_trackers()


            # Filter previous tracks.

            if self.remove:
                i = len(self.trackers)
                for trk in reversed(self.trackers):
                    i -= 1
                    # remove dead tracklet
                    if (trk.time_since_update > self.max_age):
                        self.trackers[i].deactivate()

        # If there is no room for the allowed trackers to be, drop them.
        return self.get_result()

    def associate_detections_to_trackers(self, new_instances):

        # If there are no initial detections, return a list of unmatched_detections.
        active_trackers = self.get_active_trackers()

        if (len(active_trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(new_instances)), np.empty((0, 5), dtype=int)

        class_ids_0 = []
        class_ids_1 = new_instances.pred_classes

        if self.mode == "maskiou":

            # Fills going to be matched or unmatched.
            prev_masks = []
            for i in active_trackers:
                tr = self.trackers[i]
                prev_masks.append(tr.get_mask().cpu().numpy())
            prev_masks = np.concatenate(prev_masks).astype(np.uint8)

            # Calc IoU and match based on that. => expects (N, H, W) as the inputs
            new_masks = new_instances.pred_masks.cpu().numpy().astype(np.uint8)
            print("prev_masks: ", prev_masks.shape, " new_masks: ", new_masks.shape)

            # Set these to further use
            num_prev_instances = prev_masks.shape[0]
            num_new_instances = new_masks.shape[0]

            print("prev_masks: " , prev_masks.shape[0], "new masks: " , new_masks.shape[0])

            cost_matrix = iou(new_masks, prev_masks)

            print("MaskIoU pred_states: ", num_prev_instances, " next_states: ", num_new_instances, " cost matrix: ",
                  cost_matrix.shape)

        elif self.mode == "boxiou":

            # Fills going to be matched or unmatched.
            prev_boxes = []
            for i in active_trackers:
                tr = self.trackers[i]
                prev_boxes.append(tr.get_bbox().cpu().numpy())
            prev_boxes = np.concatenate(prev_boxes)

            # Calc IoU and match based on that. => expects (N, H, W) as the inputs
            new_boxes = new_instances.pred_boxes.tensor.cpu().numpy()
            print("prev_boxes: ", prev_boxes.shape, " prev_boxes: ", new_boxes.shape)

            # Set these to further use
            num_prev_instances = prev_boxes.shape[0]
            num_new_instances = new_boxes.shape[0]
            cost_matrix = iou_batch(new_boxes, prev_boxes)

            print("BoxIoU pred_states: ", num_prev_instances, " next_states: ", num_new_instances, " cost matrix: ",
                  cost_matrix.shape)

        else:

            # GNN.
            pred_states = []
            for i in active_trackers:
                tr = self.trackers[i]
                pred_states.append(tr.get_next_state())

                # Keeps class ids for the first frame
                class_ids_0.append(tr.get_class_id())

            pred_states = torch.cat(pred_states)
            next_states = new_instances.current_states

            print("pred_states: ", pred_states, " next_states: ", next_states)

            # Multiply with the negative sign since IoU gives similarity.
            num_prev_instances = pred_states.shape[0]
            num_new_instances = next_states.shape[0]

            cost_matrix_sim = pairwise_distance_matrix(next_states, pred_states)
            cost_matrix = 1 - cost_matrix_sim.cpu().numpy()

            # print("GNN pred_states: ", num_prev_instances, " next_states: " , num_new_instances ,
            #       " cost matrix: ", cost_matrix.shape)
            #
            # print("GNN pred_states: ", num_prev_instances, " next_states: ", num_new_instances,
            #       " cost matrix: ", cost_matrix.shape)



        # Set cost matrix pairs w.r.t. class info. Increase the cost when there is no match.
        if self.use_class_info:
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

        dets = []
        hits = []

        for trk in self.trackers:

            hit_streak = trk.hit_streak
            hit_condition = hit_streak >= self.min_hits or self.frame_count <= self.min_hits

            if trk.is_active and (trk.time_since_update < 1) and hit_condition:
                # Get the instance and inject the tracking ID.
                instance = trk.get_instance()
                instance.tracking_id = [trk.id]
                dets.append(instance)
                hits.append(hit_streak)

        if len(dets) > 0:

            idxs = np.argsort(hits)

            sorted_dets = []

            for idx in idxs:
                sorted_dets.append(dets[idx])
            result = Instances.cat(sorted_dets)


        else:

            result = Instances(image_size=(int(self.height), int(self.width)))
            result.tracking_id = []

        return result