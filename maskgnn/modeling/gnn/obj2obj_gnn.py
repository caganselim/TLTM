import torch
from detectron2.utils.events import get_event_storage
from torch import nn
from maskgnn.modeling.gnn.build import GNN_REGISTRY


def _get_edge_list(num_objects_0, num_objects_1):
    # Create fully-connected adjacency matrix for single sample.
    adj_full = torch.ones(num_objects_0, num_objects_1)

    # Remove diagonal.
    # adj_full -= torch.eye(num_objects)
    edge_list = adj_full.nonzero(as_tuple=False)

    # Transpose to COO format -> Shape: [2, num_edges].
    edge_list = edge_list.transpose(0, 1)

    return edge_list.cuda()


def _get_edge_list_from_list(obj_cnts_0, obj_cnts_1):
    offset_0 = 0
    offset_1 = 0
    edge_list = []

    for obj_cnt_0, obj_cnt_1 in zip(obj_cnts_0, obj_cnts_1):
        # Get the connections of a single graph.
        local_connections = _get_edge_list(obj_cnt_0, obj_cnt_1)

        # Add the corresponding offset.
        local_connections[0, :] += offset_0
        local_connections[1, :] += offset_1

        # Update the offset
        offset_0 += obj_cnt_0
        offset_1 += obj_cnt_1

        edge_list.append(local_connections)

    edge_list = torch.cat(edge_list, dim=1)

    return edge_list.cuda()


@GNN_REGISTRY.register()
class Obj2ObjGNN(nn.Module):

    def __init__(self, cfg):
        super(Obj2ObjGNN, self).__init__()

        self.input_dim = cfg.MODEL.OBJ_ENCODER.OUTPUT_DIM
        self.hidden_dim = cfg.MODEL.GNN.HIDDEN_DIM
        self.action_dim = 0
        self.mp_iter = cfg.MODEL.GNN.MP_ITERS

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))

        node_input_dim = self.hidden_dim + self.input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim))

        self.edge_list = None

        self.edge_predictor = nn.Sequential(nn.Linear(self.input_dim * 2, self.hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_dim, 1))

        self.bce_loss = torch.nn.BCELoss(reduction="mean")

    def _unsorted_segment_sum(self, tensor, segment_ids, num_segments):

        """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
        result_shape = (num_segments, tensor.size(1))
        result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
        result.scatter_add_(0, segment_ids, tensor)

        return result

    def _edge_model(self, source, target):
        out = torch.cat([source, target], dim=1)
        # print(out.shape)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, indexes, edge_attr):
        if edge_attr is not None:
            agg = self._unsorted_segment_sum(edge_attr, indexes, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr

        return self.node_mlp(out)

    def node_pred_loss(self, node_scores, labels, frame):

        # print("node_scores: " , node_scores.shape, "labels: " , labels.shape)

        loss = self.bce_loss(node_scores, labels)

        # acc
        preds = node_scores > 0.5
        acc = (preds == labels).sum() / len(preds)
        storage = get_event_storage()
        storage.put_scalar(f"gnn/node_pred_acc_{frame}", acc)

        """
        Important logs here
        """

        return loss

    def edge_pred_loss(self, scores_0, connectivity, fg_edges_bool=None):

        # Apply loss only for fg-to-fg nodes.
        if fg_edges_bool is not None:
            scores_0 = scores_0[fg_edges_bool]
            connectivity = connectivity[fg_edges_bool]

        #print("connectivity: ", connectivity.shape, " scores_0: ", scores_0.shape)

        l_0 = self.bce_loss(scores_0, connectivity.to(torch.float32))  # TODO BCE WEIGHT

        # acc für 0
        preds_0 = scores_0 > 0.5
        acc_0 = (preds_0 == connectivity).sum() / len(preds_0)
        storage = get_event_storage()
        storage.put_scalar("gnn/edge_acc", acc_0)

        """
        Important logs here.
        """

        num_edges = connectivity.shape[0]
        num_pos_edges = connectivity.sum()
        num_neg_edges = num_edges - num_pos_edges

        storage.put_scalar("gnn/num_edges", num_edges)
        storage.put_scalar("gnn/num_pos_edges", num_pos_edges)
        storage.put_scalar("gnn/num_neg_edges", num_neg_edges)

        # Let's log them.
        tp = preds_0[connectivity].sum()

        # Connectivites are neg, preds are pos.
        fp = preds_0[torch.logical_not(connectivity)].sum()

        # Connectivities are positive, pred is neg.
        fn = torch.logical_not(preds_0[connectivity]).sum()

        precision = tp / (tp + fp + 0.000001)
        recall = tp / (tp + fn + 0.000001)

        storage.put_scalar("gnn/edge_precision", precision)
        storage.put_scalar("gnn/edge_recall", recall)


        return l_0

    def message_passing(self, node_attr_0, node_attr_1, edge_index):

        sigm = nn.Sigmoid()
        row, col = edge_index

        # Update node features
        for _ in range(self.mp_iter):
            edge_attr_01 = self._edge_model(node_attr_0[row], node_attr_1[col])  # message
            node_attr_0 = self._node_model(node_attr_0, row, edge_attr_01) + node_attr_0

        # We are not using edges here.
        edge_pred_in = torch.cat([node_attr_0[row], node_attr_1[col]], dim=1)

        edge_preds = sigm(self.edge_predictor(edge_pred_in))

        # Predict node scores
        return edge_preds

    def forward_test(self, prev_obj_features, curr_instances):

        assert len(curr_instances) == 1, len(curr_instances)

        # print("Inside GNN forward test!")

        num_objects_prev = prev_obj_features.shape[0]
        num_objects_curr = len(curr_instances[0])

        # print("num_objects_prev: ", num_objects_prev, " num_objects_curr: ", num_objects_curr)
        edge_index = _get_edge_list(num_objects_prev, num_objects_curr)

        row, col = edge_index
        node_attr_1 = curr_instances[0].obj_feats
        edge_preds = self.message_passing(prev_obj_features, node_attr_1, edge_index)

        pairwise_scores = torch.zeros(num_objects_curr, num_objects_prev + 1).cuda()
        for score, i, j in zip(edge_preds, col, row):
            pairwise_scores[i, j + 1] = score

        return curr_instances, pairwise_scores

    def forward_test_v2(self, prev_obj_features, curr_obj_features):


        # print("Inside GNN forward test!")

        num_objects_prev = prev_obj_features.shape[0]
        num_objects_curr = curr_obj_features.shape[0]

        # print("num_objects_prev: ", num_objects_prev, " num_objects_curr: ", num_objects_curr)
        edge_index = _get_edge_list(num_objects_prev, num_objects_curr)

        row, col = edge_index
        node_attr_1 = curr_obj_features
        edge_preds = self.message_passing(prev_obj_features, node_attr_1, edge_index)

        pairwise_scores = torch.zeros(num_objects_curr, num_objects_prev).cuda()
        for score, i, j in zip(edge_preds, col, row):
            pairwise_scores[i, j] = score

        return pairwise_scores


    def forward_train(self, proposals_0, proposals_1):

        obj_feats_0, obj_cnts_0 = [], []
        obj_feats_1, obj_cnts_1 = [], []
        track_ids_0, track_ids_1 = [], []

        for i0 in proposals_0:
            obj_feats_0.append(i0.obj_feats)
            track_ids_0.append(i0.gt_track_id)
            cnt = len(i0)
            assert cnt != 0, "No object case in proposals_0"
            obj_cnts_0.append(len(i0))

        # Export for the second.
        for i1 in proposals_1:
            obj_feats_1.append(i1.obj_feats)
            track_ids_1.append(i1.gt_track_id)
            cnt = len(i1)
            assert cnt != 0, "No object case in proposals_1"
            obj_cnts_1.append(len(i1))

        edge_index = _get_edge_list_from_list(obj_cnts_0, obj_cnts_1)
        row, col = edge_index
        node_attr_0, node_attr_1 = torch.cat(obj_feats_0), torch.cat(obj_feats_1)
        edge_preds = self.message_passing(node_attr_0, node_attr_1, edge_index)

        losses_dict = {}
        track_ids_0 = torch.cat(track_ids_0)
        track_ids_1 = torch.cat(track_ids_1)

        connectivity = (track_ids_0[row] == track_ids_1[col]).reshape(-1, 1)
        losses_dict["loss_gnn_edge"] = self.edge_pred_loss(edge_preds, connectivity)

        return losses_dict


if __name__ == "__main__":
    gnn = Obj2ObjGNN().cuda()

    obj_cnts = [20, 20]
    prev_img_objs = torch.rand((40, 128)).cuda()
    curr_img_objs = torch.rand((40, 128)).cuda()