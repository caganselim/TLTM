import torch
from .build import MATCHER_REGISTRY
from ..obj_encoder import build_obj_encoder

def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = torch.nn.functional.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / "avg_factor"
    else:
        return raw * weight / avg_factor


@MATCHER_REGISTRY.register()
class ObjectMatcher():
    def __init__(self, cfg):
        super(ObjectMatcher, self).__init__()
        self.match_coeff = cfg.MODEL.MATCHER.COEF
        self.bbox_dummy_iou = 0

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False, debug_mode=True):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        to_remove = bbox_ious < 0.3
        bbox_ious[to_remove] = -10000

        if add_bbox_dummy:
            bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1, device=torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1, device=torch.cuda.current_device())

            label_delta = torch.cat((label_dummy, label_delta), dim=1)

        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 4
            assert (len(self.match_coeff) == 4)

            #print(f"match_ll: {match_ll.shape}, bbox_scores: {bbox_scores.shape}, bbox_ious: {bbox_ious.shape}, label_delta: {label_delta.shape}")

            bbox_scores = torch.log(bbox_scores)
            cumulative_score = self.match_coeff[0]*match_ll + \
                               self.match_coeff[1] * bbox_scores +\
                               self.match_coeff[2] * bbox_ious +\
                               self.match_coeff[3] * label_delta
            if debug_mode:
                debug_dict = {}
                debug_dict["cumulative_score"] = cumulative_score.to("cpu")
                debug_dict["match_ll"] = match_ll.to("cpu")
                debug_dict["bbox_scores"] = bbox_scores.to("cpu")
                debug_dict["bbox_ious"] = bbox_ious.to("cpu")
                debug_dict["label_delta"] = label_delta.to("cpu")

                return cumulative_score, debug_dict

            else:

                return cumulative_score, None

    def matching_loss(self, match_score, ids, reduce=True):

        losses = dict()
        n = len(match_score)
        x_n = [s.size(0) for s in match_score]
        loss_match = 0.
        match_acc = 0.
        n_total = 0
        batch_size = len(ids)

        for score, cur_ids in zip(match_score, ids):

            n = len(cur_ids)
            valid_idx = torch.arange(n, device=torch.cuda.current_device())
            if len(valid_idx.size()) == 0: continue
            n_valid = valid_idx.size(0)
            n_total += n_valid
            ce_loss = torch.nn.CrossEntropyLoss()
            loss_match += torch.sum(ce_loss(score, cur_ids))[None]
            #loss_match += weighted_cross_entropy(score, cur_ids, torch.ones(n, device=torch.cuda.current_device()), reduce=reduce)
            match_acc += accuracy(torch.index_select(score, 0, valid_idx), torch.index_select(cur_ids, 0, valid_idx)) * n_valid

        losses['loss_match'] = loss_match / n

        if n_total > 0:
            match_acc = match_acc / n_total
        else:
            match_acc = 0.

        return losses, match_acc

    def trackrcnn_forward(self, x, ref_x, x_n, ref_x_n):

        """
        here we compute a correlation matrix of x and ref_x
        we also add a all 0 column denote no matching
        :param x:  the grouped bbox features of current frame
        :param ref_x:  the grouped bbox features of prev frame
        :param x_n: n are the numbers of proposals in the current images in the mini-batch,
        :param ref_x_n: are the numbers of ground truth bboxes in the reference images.
        :return:
        """

        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []

        for i in range(n):
            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            prods.append(prod)

        match_score = []
        for prod in prods:
            m = prod.size(0)
            dummy = torch.zeros(m, 1, device=torch.cuda.current_device())
            prod_ext = torch.cat([dummy, prod], dim=1)
            match_score.append(prod_ext)

        return match_score
