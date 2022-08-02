import torch

def dense_to_sparse(adj):
    adj = adj.squeeze()
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(0) == adj.size(1)

    edge_index = adj.nonzero(as_tuple=True)[:2]
    edge_attr = adj[edge_index]

    return torch.stack(list(edge_index), dim=0), edge_attr
