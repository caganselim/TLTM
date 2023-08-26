# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI) in 28/01/2020.
import math
import torch
from detectron2.structures import Instances

def add_ground_truth_to_proposals(targets, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        targets(list[Instances]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert targets is not None

    assert len(proposals) == len(targets)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(tagets_i, proposals_i)
        for tagets_i, proposals_i in zip(targets, proposals)
    ]


def add_ground_truth_to_proposals_single_image(targets_i, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with targets and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """

    """
    Instances(num_instances=5, image_height=704, image_width=1056, fields=[gt_boxes: Boxes(tensor([[278.4375, 534.1875, 543.8125, 703.3125],
        [508.0625, 486.0625, 774.8125, 703.3125],
        [486.0625, 608.4375, 564.4375, 703.3125],
        [458.5625, 685.4375, 876.5625, 703.3125],
        [443.4375, 435.1875, 769.3125, 703.3125]], device='cuda:1')), gt_classes: tensor([0, 0, 0, 0, 0], device='cuda:1'), gt_track_id: tensor([1, 2, 3, 4, 5], device='cuda:1'), gt_masks: PolygonMasks(num_instances=5), gt_deltas: tensor([[ 0.0259,  0.0407,  0.0205, -0.0415],
        [ 0.0103, -0.0063, -0.0261,  0.0063],
        [-0.2982, -0.1014,  0.2877,  0.0966],
        [-0.0263, -0.3077,  0.0000,  0.2683],
        [ 0.0422, -0.0564, -0.0042,  0.0549]], device='cuda:1')])
    """

    device =  targets_i.gt_boxes.device
    # proposals.proposal_boxes = proposals.pred_boxes
    # proposals.remove("pred_boxes")

    # Concatenating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.

    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(targets_i), device=device)

    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = targets_i.gt_boxes
    # to have the same fields with proposals
    gt_proposal.scores = gt_logits
    gt_proposal.pred_classes = targets_i.gt_classes
    gt_proposal.locations = torch.ones((len(targets_i), 2), device=device)

    gt_proposal.gt_boxes = targets_i.gt_boxes
    gt_proposal.gt_classes = targets_i.gt_classes
    gt_proposal.pred_classes = targets_i.gt_classes
    gt_proposal.gt_track_id = targets_i.gt_track_id

    # new_proposals = Instances.cat([proposals, gt_proposal])
    # assert len(new_proposals) > 0


    return gt_proposal
