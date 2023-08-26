import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

"""
These are used in Kalman based filtering functions:
- convert_bbox_to_z(bbox)
- convert_x_to_bbox(x, score)
"""

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))






def box_iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)

def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))
EPSILON = 1e-7


def area(masks):
    """Computes area of masks.
    Args:
      masks: Numpy array with shape [N, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.
    Returns:
      a numpy array with shape [N*1] representing mask areas.
    Raises:
      ValueError: If masks.dtype is not np.uint8
    """
    if masks.dtype != np.uint8:
        raise ValueError('Masks type should be np.uint8')

    return np.sum(masks, axis=(1, 2), dtype=np.float32)


def intersection(masks1, masks2):
  """Compute pairwise intersection areas between masks.
  Args:
    masks1: a numpy array with shape [N, height, width] holding N masks. Masks
      values are of type np.uint8 and values are in {0,1}.
    masks2: a numpy array with shape [M, height, width] holding M masks. Masks
      values are of type np.uint8 and values are in {0,1}.
  Returns:
    a numpy array with shape [N*M] representing pairwise intersection area.
  Raises:
    ValueError: If masks1 and masks2 are not of type np.uint8.
  """
  if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
    raise ValueError('masks1 and masks2 should be of type np.uint8')
  n = masks1.shape[0]
  m = masks2.shape[0]
  answer = np.zeros([n, m], dtype=np.float32)
  for i in np.arange(n):
    for j in np.arange(m):
      answer[i, j] = np.sum(np.minimum(masks1[i], masks2[j]), dtype=np.float32)
  return answer


def iou(masks1, masks2):
  """Computes pairwise intersection-over-union between mask collections.
  Args:
    masks1: a numpy array with shape [N, height, width] holding N masks. Masks
      values are of type np.uint8 and values are in {0,1}.
    masks2: a numpy array with shape [M, height, width] holding N masks. Masks
      values are of type np.uint8 and values are in {0,1}.
  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
  Raises:
    ValueError: If masks1 and masks2 are not of type np.uint8.
  """
  if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
    raise ValueError('masks1 and masks2 should be of type np.uint8')
  intersect = intersection(masks1, masks2)
  area1 = area(masks1)
  area2 = area(masks2)
  union = np.expand_dims(area1, axis=1) + np.expand_dims(
      area2, axis=0) - intersect
  return intersect / np.maximum(union, EPSILON)


def ioa(masks1, masks2):
  """Computes pairwise intersection-over-area between box collections.
  Intersection-over-area (ioa) between two masks, mask1 and mask2 is defined as
  their intersection area over mask2's area. Note that ioa is not symmetric,
  that is, IOA(mask1, mask2) != IOA(mask2, mask1).
  Args:
    masks1: a numpy array with shape [N, height, width] holding N masks. Masks
      values are of type np.uint8 and values are in {0,1}.
    masks2: a numpy array with shape [M, height, width] holding N masks. Masks
      values are of type np.uint8 and values are in {0,1}.
  Returns:
    a numpy array with shape [N, M] representing pairwise ioa scores.
  Raises:
    ValueError: If masks1 and masks2 are not of type np.uint8.
  """
  if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
    raise ValueError('masks1 and masks2 should be of type np.uint8')
  intersect = intersection(masks1, masks2)
  areas = np.expand_dims(area(masks2), axis=0)
  return intersect / (areas + EPSILON)



def pairwise_distance_matrix(x, y):

    dim = x.size(1)
    num_samples_x = x.size(0)
    num_samples_y = y.size(0)

    x = x.unsqueeze(1).expand(num_samples_x, num_samples_y, dim)
    y = y.unsqueeze(0).expand(num_samples_x, num_samples_y, dim)

    return torch.pow(x - y, 2).sum(2)


def greedy_assignment(dist):

    """
    Got this greedy assigner from CenterTrack.
    Not tested yet.
    :param dist:
    :return:
    """

    matched_indices = []

    if dist.shape[1] == 0:

        return np.array(matched_indices, np.int32).reshape(-1, 2)

    for i in range(dist.shape[0]):

        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])

    return np.array(matched_indices, np.int32).reshape(-1, 2)


def do_hungarian(objs_0, objs_1):

    # Calculate cost matrix.
    cost_matrix = pairwise_distance_matrix(objs_0, objs_1)

    # Apply hungarian => return indices
    inds = linear_sum_assignment(cost_matrix.cpu().numpy())

    return inds
