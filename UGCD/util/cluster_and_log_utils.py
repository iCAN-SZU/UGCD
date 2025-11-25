from sklearn.dummy import check_random_state
import torch
import torch.distributed as dist
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


def all_sum_item(item):
    item = torch.tensor(item).cuda()
    dist.all_reduce(item)
    return item.item()

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = y_pred.size
    try: 
        if dist.get_world_size() > 0:
            total_acc = all_sum_item(total_acc)
            total_instances = all_sum_item(total_instances)
    except:
        pass
    total_acc /= total_instances

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    
    try:
        if dist.get_world_size() > 0:
            old_acc = all_sum_item(old_acc)
            total_old_instances = all_sum_item(total_old_instances)
    except:
        pass
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    
    try:
        if dist.get_world_size() > 0:
            new_acc = all_sum_item(new_acc)
            total_new_instances = all_sum_item(total_new_instances)
    except:
        pass
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


def split_cluster_acc_v2_balanced(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}

    old_acc = np.zeros(len(old_classes_gt))
    total_old_instances = np.zeros(len(old_classes_gt))
    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(len(new_classes_gt))
    total_new_instances = np.zeros(len(new_classes_gt))
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    try:
        if dist.get_world_size() > 0:
            old_acc, new_acc = torch.from_numpy(old_acc).cuda(), torch.from_numpy(new_acc).cuda()
            dist.all_reduce(old_acc), dist.all_reduce(new_acc)
            dist.all_reduce(total_old_instances), dist.all_reduce(total_new_instances)
            old_acc, new_acc = old_acc.cpu().numpy(), new_acc.cpu().numpy()
            total_old_instances, total_new_instances = total_old_instances.cpu().numpy(), total_new_instances.cpu().numpy()
    except:
        pass

    total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_acc /= total_old_instances
    new_acc /= total_new_instances
    total_acc, old_acc, new_acc = total_acc.mean(), old_acc.mean(), new_acc.mean()
    return total_acc, old_acc, new_acc


EVAL_FUNCS = {
    'v2': split_cluster_acc_v2,
    'v2b': split_cluster_acc_v2_balanced
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs, save_name, T=None,
                        print_output=True, args=None):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask)
        log_name = f'{save_name}_{f_name}'

        if i == 0:
            to_return = (all_acc, old_acc, new_acc)

        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            try:
                if dist.get_rank() == 0:
                    try:
                        args.logger.info(print_str)
                    except:
                        print(print_str)
            except:
                pass

    return to_return


def kmeans_plusplus_initializer(u_feats, l_feats, l_targets, k=3, random_state=0):
    r'''
    Args:
        u_feats: unlabel features
        
    '''
    
    def supp_idxs(c):
        return l_targets.eq(c).nonzero().squeeze(1)

    l_classes = torch.unique(l_targets)
    support_idxs = list(map(supp_idxs, l_classes))
    l_centers = torch.stack([l_feats[idx_list].mean(0) for idx_list in support_idxs])
    cat_feats = torch.cat((l_feats, u_feats))

    centers = torch.zeros([k, cat_feats.shape[1]]).type_as(cat_feats)
    centers[:len(l_classes)] = l_centers

    labels = -torch.ones(len(cat_feats)).type_as(cat_feats).long()

    l_classes = l_classes.cpu().long().numpy()
    l_targets = l_targets.cpu().long().numpy()
    l_num = len(l_targets)
    cid2ncid = {cid:ncid for ncid, cid in enumerate(l_classes)}  # Create the mapping table for New cid (ncid)
    for i in range(l_num):
        labels[i] = cid2ncid[l_targets[i]]

    #initialize the centers, the first 'k' elements in the dataset will be our initial centers
    centers = kpp(u_feats, l_centers, k=k, random_state=random_state)
    return centers

    
def kpp(X, pre_centers=None, k=10, random_state=None):
    random_state = check_random_state(random_state)

    if pre_centers is not None:

        C = pre_centers

    else:

        C = X[random_state.randint(0, len(X))]

    C = C.view(-1, X.shape[1])

    while C.shape[0] < k:

        dist = pairwise_distance(X, C)
        dist = dist.view(-1, C.shape[0])
        d2, _ = torch.min(dist, dim=1)
        prob = d2/d2.sum()
        cum_prob = torch.cumsum(prob, dim=0)
        r = random_state.rand()

        if len((cum_prob >= r).nonzero()) == 0:
            debug = 0
        else:
            ind = (cum_prob >= r).nonzero()[0][0]
        C = torch.cat((C, X[ind].view(1, -1)), dim=0)

    return C

    
def pairwise_distance(data1, data2, batch_size=1024):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)

    if batch_size == None:
        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
                #  torch.cuda.empty_cache()
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis
