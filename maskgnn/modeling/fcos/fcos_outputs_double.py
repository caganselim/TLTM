import logging
import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from maskgnn.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from maskgnn.utils.comm import reduce_sum
from maskgnn.layers import ml_nms


logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores
    
"""


def compute_ctrness_targets(reg_targets):

    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))

    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]

    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])


    return torch.sqrt(ctrness)


def track_loss(track_pred, track_target, labels):

    num_classes = 1
    labels = labels.flatten()


    # Find fg indices
    pos_inds = torch.nonzero(labels != num_classes).squeeze(1)

    track_pred = track_pred[pos_inds]
    track_target = track_target[pos_inds]

    tracking_loss = smooth_l1_loss(track_pred, track_target, beta=0.1, reduction='mean')

    return {'tracking_loss': tracking_loss}


def fcos_losses(
        labels,
        reg_targets,
        logits_pred,
        reg_pred,
        ctrness_pred,
        focal_loss_alpha,
        focal_loss_gamma,
        iou_loss,

):

    num_classes = logits_pred.size(1)
    labels = labels.flatten()
    # Find fg indices
    pos_inds = torch.nonzero(labels != num_classes).squeeze(1)


    # Count how many fg indices are.
    num_pos_local = pos_inds.numel()

    # Merge all positives across GPUs.
    num_gpus = get_world_size()
    total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
    num_pos_avg = max(total_num_pos / num_gpus, 1.0)

    # prepare one_hot
    class_target = torch.zeros_like(logits_pred)
    class_target[pos_inds, labels[pos_inds]] = 1

    class_loss = sigmoid_focal_loss_jit(
        logits_pred,
        class_target,
        alpha=focal_loss_alpha,
        gamma=focal_loss_gamma,
        reduction="sum",
    ) / num_pos_avg

    reg_pred = reg_pred[pos_inds]
    reg_targets = reg_targets[pos_inds]
    ctrness_pred = ctrness_pred[pos_inds]

    ctrness_targets = compute_ctrness_targets(reg_targets)
    ctrness_targets_sum = ctrness_targets.sum()
    ctrness_norm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)

    reg_loss = iou_loss(
        reg_pred,
        reg_targets,
        ctrness_targets
    ) / ctrness_norm

    ctrness_loss = F.binary_cross_entropy_with_logits(
        ctrness_pred,
        ctrness_targets,
        reduction="sum"
    ) / num_pos_avg

    losses = {
        "loss_fcos_cls": class_loss,
        "loss_fcos_loc": reg_loss,
        "loss_fcos_ctr": ctrness_loss
    }


    return losses, {}


class FCOSOutputsWithTracking(object):

    def __init__(
            self,
            images_0,
            images_1,
            locations,
            output_dict,
            focal_loss_alpha,
            focal_loss_gamma,
            iou_loss,
            center_sample,
            sizes_of_interest,
            strides,
            radius,
            num_classes,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            thresh_with_ctr,
            gt_instances_0=None,
            gt_instances_1=None
    ):


        self.output_dict = output_dict

        '''
        
        logits_0, logits_1: [-inf,+inf]
        bbox_reg_0, bbox_reg_1: ltrb values.
        ctrness_0, ctrness_1: Centerness score.
        track_feats: Used for tracking. 
                
        '''

        self.locations = locations
        self.gt_instances_0 = gt_instances_0
        self.gt_instances_1 = gt_instances_1

        self.num_feature_maps = len(output_dict['logits_0'])
        self.num_images = len(images_0)
        self.image_sizes = images_0.image_sizes

        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss = iou_loss
        self.center_sample = center_sample
        self.sizes_of_interest = sizes_of_interest
        self.strides = strides
        self.radius = radius
        self.num_classes = num_classes
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.thresh_with_ctr = thresh_with_ctr

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self):

        num_loc_list = [len(loc) for loc in self.locations]

        # h*w => 15560, 9180,

        self.num_loc_list = num_loc_list

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(self.locations):

            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            #print(l , " loc_to_size_range_per_level ", loc_to_size_range_per_level[None], " loc_per-level ", loc_per_level)
            loc_to_size_range.append(loc_to_size_range_per_level[None].expand(num_loc_list[l], -1))

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(self.locations, dim=0)

        training_targets_0 = self.compute_targets_for_locations(locations, self.gt_instances_0, loc_to_size_range)
        training_targets_1 = self.compute_targets_for_locations(locations, self.gt_instances_1, loc_to_size_range)

        track_targets = self.compute_track_targets_for_locations(locations, self.gt_instances_0, self.gt_instances_1,
                                                                loc_to_size_range)


        # transpose im first training_targets to level first ones
        training_targets_0 = {k: self._transpose(v, num_loc_list) for k, v in training_targets_0.items()}
        training_targets_1 = {k: self._transpose(v, num_loc_list) for k, v in training_targets_1.items()}



        track_targets = {k: self._transpose(v, num_loc_list) for k, v in track_targets.items()}

        # we normalize reg_targets by FPN's strides here
        reg_targets_0 = training_targets_0["reg_targets"]
        reg_targets_1 = training_targets_1["reg_targets"]

        for l in range(len(reg_targets_0)):
            reg_targets_0[l] = reg_targets_0[l] / float(self.strides[l])

        for l in range(len(reg_targets_1)):
            reg_targets_1[l] = reg_targets_1[l] / float(self.strides[l])

        # for l in range(len(track_targets_0)):
        #     track_targets_0[l] = track_targets_0[l] / float(self.strides[l])

        return training_targets_0, training_targets_1, track_targets

    def get_sample_region(self, gt, strides, num_loc_list, loc_xs, loc_ys, radius=1):

        num_gts = gt.shape[0]

        """
        num_loc_list => [152000, 3800, 950, 247, 70]
        Shows how many locations inside a level. It sums up to 20267 (total locations)
        loc_xs, loc_ys => 20267
        
        gt_before expand: torch.Size([5, 4])
        gt after expand: torch.Size([20267, 5, 4])
        
        center_x : torch.Size([20267, 5])
        center_y : torch.Size([20267, 5])
        center_gt: torch.Size([20267, 5, 4])
        
        strides: [8, 16, 32, 64, 128]
                
        """

        K = len(loc_xs)
        gt = gt[None].expand(K, num_gts, 4)

        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)

        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)

        """
        
        Here beg and end are the offsets that used to predict. Example:
        lvl ------------------------------------------
        0) beg: 0 , end: 152000
        1) beg: 152000, end: 152000 + 380
        2) beg: 152000 + 380, end: 152000 + 380 + 680
        ...
        
        For each level, 
               
        """

        beg = 0

        for level, num_loc in enumerate(num_loc_list):

            end = beg + num_loc
            stride = strides[level] * radius

            """
            These values are the minimum and a maximum bounding boxes that can be regressed.
            center_x & center_y are the ground-truth object centers.
            
            center_x: torch.Size([20267, 5]) => meaning that there are 5 objects.
            center_y: torch.Size([20267, 5]) => meaning that there are 5 objects.
            
            xmin, ymin, xmax, ymax => torch.Size[152000,5] => defines the limits at the level 0
                     
            """

            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride


            """
            Limit sample region in the gt data. center_gt is initially zero.
            
            center_gt: torch.Size([20267, 5, 4]) => [num_pixels, num_instances, (xmin,ymin,xmax,ymax)]
                                
            -----------------------------------------------------------------------------------------------
            torch.where(cond, x, y) return a tensor of elements selected from either x or y, depending on condition.
            
            
            * xmin > gt[beg:end, :, 0]
            * ymin > gt[beg:end, :, 1] 
            
            (xmin,ymin)---------------
            -                        -
            -                        -
            -                        -
            -                        -
            ----------------(xmax,ymax)
            
            
            - If xmin (or ymin) is bigger than gt assign xmin (or ymin) else gt. 
            - If xmax (or ymax) is bigger than gt, assign gt else xmin (or ymin).
            
            """

            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)

            beg = end

        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]

        # Concat them in the last dimension.
        center_bbox = torch.stack((left, top, right, bottom), -1)

        """        
        If any of them (ltrb) is invalid, ignore at that scale.
        center_bbox shape:  torch.Size([20267, 5, 4])
        inside_gt_bbox_mask shape:  torch.Size([20267, 5])
        """

        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        return inside_gt_bbox_mask

    def compute_track_targets_for_locations(self, locations, targets_0, targets_1, size_ranges):

        track_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets_0)):

            targets_per_im_0 = targets_0[im_i]
            targets_per_im_1 = targets_1[im_i]

            # Export tracking ids
            track_ids_0 = targets_per_im_0.gt_track_id
            track_ids_1 = targets_per_im_1.gt_track_id

            # Match the targets based on the index.
            matches_0, matches_1 = [], []
            for i, id_0 in enumerate(track_ids_0):
                for j, id_1 in enumerate(track_ids_1):
                    if id_0 == id_1:
                        matches_0.append(i)
                        matches_1.append(j)

            if len(matches_0) == 0:
                track_targets.append(torch.zeros(len(xs), 4).cuda())
                continue

            area = targets_per_im_0.gt_boxes[matches_0].area()
            gt_bboxes_0 = targets_per_im_0.gt_boxes.tensor[matches_0,:]
            gt_bboxes_1 = targets_per_im_1.gt_boxes.tensor[matches_1,:]

            # Calculate loss target. x1 y1 x2 y2
            gt_w0 = gt_bboxes_0[:, 2] - gt_bboxes_0[:, 0]
            gt_h0 = gt_bboxes_0[:, 3] - gt_bboxes_0[:, 1]
            gt_w1 = gt_bboxes_1[:, 2] - gt_bboxes_1[:, 0]
            gt_h1 = gt_bboxes_1[:, 3] - gt_bboxes_1[:, 1]

            gt_delta_x = (gt_bboxes_1[:, 0] - gt_bboxes_0[:, 0]) / gt_w0
            gt_delta_y = (gt_bboxes_1[:, 1] - gt_bboxes_0[:, 1]) / gt_h0
            gt_delta_w = torch.log(gt_w1 / gt_w0)
            gt_delta_h = torch.log(gt_h1 / gt_h0)

            gt_deltas = torch.stack([gt_delta_x, gt_delta_y, gt_delta_w, gt_delta_h], dim=1)
            # print("gt_bboxes_0: ", gt_bboxes_0, " gt_bboxes_1: ", gt_bboxes_1,
            #       " gt_deltas: ", gt_deltas, "gt_deltas2: ", targets_per_im_0.gt_deltas)

            """
            gt_deltas:  torch.Size([5, 4])
            is_in_boxes:  torch.Size([20267, 5]) ~ 150 of them are correct.
            track_targets_per_im:  torch.Size([20267, 4])
            
            """

            is_in_boxes = self.get_sample_region(gt_bboxes_0, self.strides,
                                                 self.num_loc_list, xs, ys,
                                                 radius=self.radius)

            mask = is_in_boxes.sum(dim=1)
            K = len(xs)
            num_gts = gt_deltas.shape[0]
            gt_deltas = gt_deltas[None].expand(K, num_gts, 4)

            locations_to_gt_area = area[None].repeat(len(locations), 1)

            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            track_targets_per_im = gt_deltas[range(len(locations)), locations_to_gt_inds]

            tr_target = mask.unsqueeze(1)*track_targets_per_im
            #print("tr target: ", tr_target.shape)
            track_targets.append(tr_target)

        return {"track_targets": track_targets}

    def compute_targets_for_locations(self, locations, targets, size_ranges):

        """
        :param locations:
        :param targets: Instances object that includes gt targets
        :param size_ranges:
        :return:
        """

        labels = []
        reg_targets = []

        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):

            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                continue

            area = targets_per_im.gt_boxes.area()

            """
            xs (also ys) original shape:  torch.Size([20267])
            xs (also ys) converted (xs[:,None]) shape:  torch.Size([20267, 1])
            bboxes[:, 0] original shape:  torch.Size([5])
            bboxes[:, 0][None] converted shape: torch.Size([1, 5])
            l,t,r,b shape: torch.Size([20267, 5])
            reg_targets_per_im:  torch.Size([20267, 5, 4])
            """

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:

                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, self.num_loc_list,
                    xs, ys, radius=self.radius
                )

                """
                
                is_in_boxes (bool) => torch.Size([20267, 5])
                                               
                """

            else:

                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0


            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]

            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]


            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return {"labels": labels, "reg_targets": reg_targets}


    def losses(self):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets_0, training_targets_1, track_targets = self._get_ground_truth()

        labels_0, reg_targets_0 = training_targets_0["labels"], training_targets_0["reg_targets"]
        labels_1, reg_targets_1 = training_targets_1["labels"], training_targets_1["reg_targets"]

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.

        logits_pred_0 = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in self.output_dict['logits_0']
            ], dim=0,)

        logits_pred_1 = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in self.output_dict['logits_1']
            ], dim=0,)


        reg_pred_0 = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4)
                for x in self.output_dict['bbox_reg_0']
            ], dim=0,)

        reg_pred_1 = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4)
                for x in self.output_dict['bbox_reg_1']
            ], dim=0,)

        ctrness_pred_0 = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in self.output_dict['ctrness_0']
            ], dim=0,)

        ctrness_pred_1 = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in self.output_dict['ctrness_1']
            ], dim=0, )

        # Labels
        labels_0 = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in labels_0
            ], dim=0,)

        labels_1 = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in labels_1
            ], dim=0, )

        # Reg targets
        reg_targets_0 = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in reg_targets_0
            ], dim=0, )

        reg_targets_1 = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in reg_targets_1
            ], dim=0,)

        track_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.permute(0, 2, 3, 1).reshape(-1, 4)
                for x in self.output_dict['track_feats']
            ], dim=0,)


        track_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in track_targets['track_targets']
            ], dim=0,)

        # Calc tracking loss here.
        loss_0, _ = fcos_losses(labels_0, reg_targets_0, logits_pred_0, reg_pred_0, ctrness_pred_0,
                             self.focal_loss_alpha, self.focal_loss_gamma, self.iou_loss)

        loss_1, _ = fcos_losses(labels_1, reg_targets_1, logits_pred_1, reg_pred_1, ctrness_pred_1,
                             self.focal_loss_alpha, self.focal_loss_gamma, self.iou_loss)

        for key in loss_0.keys():
            loss_0[key] = loss_0[key] + loss_1[key]

        loss_t = track_loss(track_pred, track_targets, labels_0)
        loss_0['tracking_loss'] = loss_t['tracking_loss']

        return loss_0

    def predict_proposals(self):

        def process_single(idx):
            sampled_boxes = []

            bundle = (self.locations,
                      self.output_dict[f'logits_{idx}'],
                      self.output_dict[f'bbox_reg_{idx}'],
                      self.output_dict[f'ctrness_{idx}'],
                      self.output_dict['track_feats'],
                      self.strides)

            for i, (l, o, r, c, t, s) in enumerate(zip(*bundle)):
                # recall that during training, we normalize regression targets with FPN's stride.
                # we denormalize them here.
                r = r * s

                # Process it for the first frame only.
                t = t if idx == 0 else None

                sample = self.forward_for_single_feature_map(
                        l, o, r, c, t, self.image_sizes
                    )

                sampled_boxes.append(sample)

            boxlists = list(zip(*sampled_boxes))
            boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
            boxlists = self.select_over_all_levels(boxlists)

            return boxlists

        return process_single(0), process_single(1)

    def forward_for_single_feature_map(
            self, locations, box_cls,
            reg_pred, ctrness, tracking, image_sizes
    ):
        N, C, H, W = box_cls.shape


        """
        put in the same format as locations (flatten H*W)
        N is the number of images inside a batch.
        """

        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness = ctrness.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness = ctrness.reshape(N, -1).sigmoid()

        if tracking is not None:
            tracking = tracking.view(N, 4, H, W).permute(0, 2, 3, 1)
            tracking = tracking.reshape(N, -1, 4)

        """
        
        * if self.thresh_with_ctr is True, we multiply the classification
        scores with centerness scores before applying the threshold.
        
        * This will affect the final performance by about 0.05 AP but save some time
        
        """


        if self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, None]

        candidate_inds = box_cls > self.pre_nms_thresh
        # pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        """
        pre_nms_top_n holds how many indices are we going to keep. 
        pre_nms_top_n shape: torch.size(N) where N is the number of images in a batch.

        """

        if not self.thresh_with_ctr:

            box_cls = box_cls * ctrness[:, :, None]

        results = []


        """
        Forward for a single image (feature map)
        
        - box_cls: [N, H*W, C], per_box_cls: [H*W, C]
        - candidate_inds: [N, H*W, C], per_candidate_inds: [H*W, C]
        - box_regression: [N, H*W, 4], per_box_regression: [H*W, 4]
        """


        for i in range(N):

            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            """
            size => [K,2]
            per_box_cls, per_box_loc => K 
            per_box_loc holds the flattened index. (like 55, 23, 42 ...)
            per_locations: [K,2]
            per_pre_nms_top_n: how many indices that we include (shape: scalar)
            """

            per_candidate_nonzeros = torch.nonzero(per_candidate_inds, as_tuple=False)
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            if tracking is not None:
                per_tracking = tracking[i]
                per_tracking = per_tracking[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

                # MaskGNN
                if tracking is not None:
                    per_tracking = per_tracking[top_k_indices]


            # Convert ltrb + center coordinate to bbox.
            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations

            if tracking is not None:

                boxlist.pred_deltas = per_tracking

            results.append(boxlist)

        return results



    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            results.append(result)
        return results
