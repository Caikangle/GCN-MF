import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from preprocess_help import preprocess_adj, get_sparse_input


# dropout
def sparse_dropout(x, rate, noise_shape, training=False):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    if not training:
        return x
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    i = x._indices()  # [2, 49216]
    v = x._values()  # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1. / (1 - rate))

    return out


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer, n_feat_nonzero, dropout=0, device=torch.device('cuda:0'), **kwargs):
        super(SimpleGCN, self).__init__()

        self.device = device
        self.dropout = dropout

        self.n_feat_nonzero = n_feat_nonzero
        self.n_layer = n_layer
        self.lin = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.lin)

    def forward(self, features, adj, enhanced_message_matrix=None):
        adj_drop = adj
        supports = preprocess_adj(adj_drop)
        X, A = get_sparse_input(features, supports, self.device)
        X = sparse_dropout(X, self.dropout, self.n_feat_nonzero, self.training)
        X = torch.sparse.mm(X, self.lin)
        layerwise_feat_list = []
        for i in range(self.n_layer):
            X = torch.sparse.mm(A, X)
            # enhanced_message_matrix plays a role in middle layers
            if enhanced_message_matrix is not None and i != 0 and i != self.n_layer - 1:
                X = torch.sparse.mm(enhanced_message_matrix, X)
            layerwise_feat_list.append(X)
        return F.log_softmax(X, dim=1), X, layerwise_feat_list
