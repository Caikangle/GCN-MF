import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def train(gcn_model, optimizer, scheduler, data, train_mask,
          val_mask, test_mask, labels_cpu, epochs=100):
    """
    training function
    :param gcn_model:
    :param optimizer:
    :param scheduler:
    :param data:
    :param train_mask:
    :param val_mask:
    :param test_mask:
    :param labels_cpu:
    :param model_name:
    :param dataset_str:
    :param enhance_methods:
    :param epochs:
    :return:
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
        # X_original X_parwalks X_ppmi
        logits, lastlayerdata, _ = gcn_model(*data['X_parwalks'])
        # loss = F.nll_loss(logits[train_idx], data['y'][train_idx])
        loss = F.nll_loss(logits[train_mask], data['y'][train_mask])
        loss.backward()
        optimizer.step()
        if True:
            gcn_model.eval()
            # evaluate
            logits = gcn_model(*data['X_original'])[0].cpu().detach()
            # train_loss = F.nll_loss(logits[train_idx], labels_cpu[train_idx])
            # train_acc = accuracy_score(labels_cpu[train_idx], logits.argmax(axis=1)[train_idx]).item()
            # val_loss = F.nll_loss(logits[val_idx], labels_cpu[val_idx])
            # val_acc = accuracy_score(labels_cpu[val_idx], logits.argmax(axis=1)[val_idx]).item()
            train_loss = F.nll_loss(logits[train_mask], labels_cpu[train_mask])
            train_acc = accuracy_score(labels_cpu[train_mask], logits.argmax(axis=1)[train_mask]).item()
            val_loss = F.nll_loss(logits[val_mask], labels_cpu[val_mask])
            val_acc = accuracy_score(labels_cpu[val_mask], logits.argmax(axis=1)[val_mask]).item()
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.4f}'.format(train_loss.item()),
                  'acc_train: {:.4f}'.format(train_acc),
                  'loss_val: {:.4f}'.format(val_loss.item()),
                  'acc_val: {:.4f}'.format(val_acc))
            # print('Epoch: {%f},val_loss:%f,val_acc:%f' % (i, val_loss, val_acc))
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss.item())
            val_loss_list.append(val_loss.item())

            if val_acc > best_acc:
                best_acc = val_acc
                test_acc = accuracy_score(labels_cpu[test_mask], logits.argmax(axis=1)[test_mask]).item()
                # best_model_path = 'best_models/' + dataset_str + '_' + model_name + '_' + enhance_methods + '.pkl'
                # torch.save(gcn_model.state_dict(), best_model_path)
        scheduler.step()
    return best_acc, test_acc, train_loss_list, val_loss_list, train_acc_list, val_acc_list
