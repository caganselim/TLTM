import torch


def _get_edge_list(num_objects_0, num_objects_1):

    # Create fully-connected adjacency matrix for single sample.
    adj_full = torch.ones(num_objects_0, num_objects_1)

    # Remove diagonal.
    #adj_full -= torch.eye(num_objects)
    edge_list = adj_full.nonzero(as_tuple=False)

    # Transpose to COO format -> Shape: [2, num_edges].
    edge_list = edge_list.transpose(0, 1)

    return edge_list



if __name__ == "__main__":

    edges = _get_edge_list(3,4)

    # Edges
    row, col = edges

    print(row)

    print(col)