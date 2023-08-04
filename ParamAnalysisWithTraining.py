"""
    Parameters(alpha,beta) analysis based on SGC model
"""
import os

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import ShuffleSplit
from torch import optim
from torch.optim import Adam

from utils.DataStorage import load_sparse_matrix
from utils.TrainingUtils import train
from utils.models import SimpleGCN
from utils.preprocess_help import load_data, preprocess_features, preprocess_adj

device = torch.device('cpu')

# dataset_str = 'citeseer'
dataset_str = 'cora'

adj, features, labels, _, _, _ = load_data(dataset_str)
features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]
supports = preprocess_adj(adj)
labels_idx = labels.argmax(axis=1)
labels_cuda = torch.from_numpy(labels.argmax(axis=1)).long().to(device)
labels_cpu = torch.from_numpy(labels.argmax(axis=1)).long()

row = features[0][:, 0]  # rows
col = features[0][:, 1]  # cols
data = features[1]
shape = features[2]
X = sp.csr_matrix((data, (row, col)), shape)

supports = preprocess_adj(adj)  # coordinates, data, shape
row = supports[0][:, 0]  # rows
col = supports[0][:, 1]  # cols
data = supports[1]
shape = supports[2]
A = sp.csr_matrix((data, (row, col)), shape)


def getTrainValTestSet(dataset_name,
                       splits_file_path,
                       train_percentage=None,
                       val_percentage=None):
    if splits_file_path:
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
    else:
        assert (train_percentage is not None and val_percentage is not None)
        assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)

        if dataset_name in {'cora', 'citeseer'}:
            disconnected_node_file_path = os.path.join('unconnected_nodes', f'{dataset_name}_unconnected_nodes.txt')
            with open(disconnected_node_file_path) as disconnected_node_file:
                disconnected_node_file.readline()
                disconnected_nodes = []
                for line in disconnected_node_file:
                    line = line.rstrip()
                    disconnected_nodes.append(int(line))

            disconnected_nodes = np.array(disconnected_nodes)
            connected_nodes = np.setdiff1d(np.arange(features.shape[0]), disconnected_nodes)

            connected_labels = labels[connected_nodes]

            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(connected_labels), connected_labels))
            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
                np.empty_like(connected_labels[train_and_val_index]), connected_labels[train_and_val_index]))
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            train_mask = np.zeros_like(labels)
            train_mask[connected_nodes[train_index]] = 1
            val_mask = np.zeros_like(labels)
            val_mask[connected_nodes[val_index]] = 1
            test_mask = np.zeros_like(labels)
            test_mask[connected_nodes[test_index]] = 1
        else:
            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(labels), labels))
            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
                np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            train_mask = np.zeros_like(labels)
            train_mask[train_index] = 1
            val_mask = np.zeros_like(labels)
            val_mask[val_index] = 1
            test_mask = np.zeros_like(labels)
            test_mask[test_index] = 1
    return train_mask, val_mask, test_mask


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    convert scipy.sparseMatrix into torch.sparseTensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


batch = 1
splits_file_path = 'splits/' + str(dataset_str) + '_split_0.6_0.2_' + str(batch) + '.npz'
train_mask, val_mask, test_mask = getTrainValTestSet(dataset_str, splits_file_path)
train_mask = train_mask.astype(bool)
val_mask = val_mask.astype(bool)
test_mask = test_mask.astype(bool)

in_dim = features[2][1]
out_dim = labels.shape[1]
n_layers = 4
n_feat_nonzero = features[0].shape[0]
epochs = 200

# set random seed
random_seeds = [123, 345, 234, 683, 372, 385, 348, 823, 644, 765]
s = random_seeds[batch]
np.random.seed(s)
torch.manual_seed(s)
torch.cuda.manual_seed(s)
torch.cuda.manual_seed_all(s)

# load message reinforcement matrix
dir = 'ParWalksEnhancedMatrix/'
suffix = '.npz'

# beta_list
beta_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20]
# alpha_list
upper_percent_list = [99.9, 99.8, 99.7, 99.6, 99.5, 99.4, 99.3, 99.2, 99.1, 99.0]

# parwalks enhanced matrix
res_list = []
for i in range(len(upper_percent_list)):
    upper_percent = upper_percent_list[i]
    for j in range(len(beta_list)):
        print('------------------------------------')
        beta = beta_list[j]
        filename = 'parwalks_enhanced_upper_percent_' + str(upper_percent) + '%_beta_' + str(beta)
        parwalks_enhanced_matrix = load_sparse_matrix(dir + filename + suffix)
        parwalks_enhanced_matrix = scipy_sparse_mat_to_torch_sparse_tensor(parwalks_enhanced_matrix)
        parwalks_enhanced_matrix = parwalks_enhanced_matrix.to(device)
        data = {
            'X_original': (features, adj, None),
            'X_parwalks': (features, adj, parwalks_enhanced_matrix),
            'y': labels_cuda
        }
        dropout = 0.0
        gcn_model = SimpleGCN(in_dim, out_dim, n_layers, n_feat_nonzero, dropout, device).to(device)

        lr = 0.05
        weight_decay = 1e-4
        optimizer = Adam(gcn_model.parameters(), lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 60, 0.97)
        best_acc, test_acc, _, _, _, _ = train(gcn_model=gcn_model, optimizer=optimizer,
                                               scheduler=scheduler, data=data, train_mask=train_mask,
                                               val_mask=val_mask, test_mask=test_mask, labels_cpu=labels_cpu,
                                               epochs=epochs)
        s = str(upper_percent) + '-' + str(beta) + '-' + str(test_acc)
        res_list.append(s)
        print(s)
        print('------------------------------------')
print('Training Over')
# save alpha-beta-accuracy list
with open('alpha_beta_accuracy.txt', 'w') as f:
    f.write(' '.join(res_list))

print(res_list)
print("Done")
