"""
    Training GCN,GCN-MF model
"""
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from torch import optim
from torch.optim import Adam

from utils.DataStorage import load_sparse_matrix
from utils.preprocess_help import load_data, preprocess_features
from utils.models import SimpleGCN


def train(gcn_model, optimizer, scheduler, data, train_mask,
          val_mask, test_mask, labels_cpu, epochs=100):
    """
    Training function
    :param gcn_model: gcn_model object
    :param optimizer:
    :param scheduler:
    :param data: training data
    :param train_mask:
    :param val_mask:
    :param test_mask:
    :param labels_cpu: label of each Training node
    :param epochs:
    :return: Training results(loss,accuracy,...,etc)
    """
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    step = 0
    best_acc = 0
    for epoch in range(epochs):
        step += 1
        gcn_model.train()
        optimizer.zero_grad()
        # X_original X_parwalks
        logits, lastlayerdata, _ = gcn_model(*data['X_parwalks'])
        loss = F.nll_loss(logits[train_mask], data['y'][train_mask])
        loss.backward()
        optimizer.step()
        if True:
            gcn_model.eval()
            # evaluate
            logits = gcn_model(*data['X_original'])[0].cpu().detach()
            train_loss = F.nll_loss(logits[train_mask], labels_cpu[train_mask])
            train_acc = accuracy_score(labels_cpu[train_mask], logits.argmax(axis=1)[train_mask]).item()
            val_loss = F.nll_loss(logits[val_mask], labels_cpu[val_mask])
            val_acc = accuracy_score(labels_cpu[val_mask], logits.argmax(axis=1)[val_mask]).item()
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.4f}'.format(train_loss.item()),
                  'acc_train: {:.4f}'.format(train_acc),
                  'loss_val: {:.4f}'.format(val_loss.item()),
                  'acc_val: {:.4f}'.format(val_acc))
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss.item())
            val_loss_list.append(val_loss.item())

            if val_acc > best_acc:
                best_acc = val_acc
                test_acc = accuracy_score(labels_cpu[test_mask], logits.argmax(axis=1)[test_mask]).item()
        scheduler.step()
    return best_acc, test_acc, train_loss_list, val_loss_list, train_acc_list, val_acc_list


def getBatchTrainValTestSet(dataset_str,
                            batch=0):
    """
    get different TrainValTestSet
    :param dataset_str:
    :param batch: batch number of the spliting
    :return: train_mask,val_mask,test_mask
    """
    splits_file_path = 'splits/' + str(dataset_str) + '_split_0.6_0.2_' + str(batch) + '.npz'
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    return train_mask.astype(bool), val_mask.astype(bool), test_mask.astype(bool)


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    convert scipy.sparse matrix into torch.sparse Tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# prepare data
dataset_str = 'cora'
# dataset_str = 'citeseer'
device = torch.device('cpu')

adj, features, labels, _, _, _ = load_data(dataset_str)
features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]
labels_cuda = torch.from_numpy(labels.argmax(axis=1)).long().to(device)
labels_cpu = torch.from_numpy(labels.argmax(axis=1)).long()

# parwalks_enhanced_matrix

dir = 'ParWalksEnhancedMatrix/' + dataset_str + '/'
upper_percent = 99
beta = 0.1
suffix = '.npz'

# parwalks enhanced matrix
filename = 'parwalks_enhanced_upper_percent_' + str(upper_percent) + '%_beta_' + str(beta)
parwalks_enhanced_matrix = load_sparse_matrix(dir + filename + suffix)
parwalks_enhanced_matrix = scipy_sparse_mat_to_torch_sparse_tensor(parwalks_enhanced_matrix)
parwalks_enhanced_matrix = parwalks_enhanced_matrix.to(device)

data = {
    'X_original': (features, adj, None),
    'X_parwalks': (features, adj, parwalks_enhanced_matrix),
    'y': labels_cuda
}

in_dim = features[2][1]
out_dim = labels.shape[1]
n_layers = 4
n_feat_nonzero = features[0].shape[0]
epochs = 200
random_seeds = [123, 345, 234, 683, 372, 385, 348, 823, 644, 765]

test_acc_list = []
for batch in range(10):
    # Set random seeds
    s = random_seeds[batch]
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # get Train_Val_Test Set
    train_mask, val_mask, test_mask = getBatchTrainValTestSet(dataset_str=dataset_str, batch=batch)
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
    test_acc_list.append(test_acc)
print('test_acc_list=', test_acc_list)
print('Average test accuracy:', np.mean(test_acc_list))
