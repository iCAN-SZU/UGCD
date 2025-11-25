import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from util.general_utils import concat_all_gather


class SinkhornKnopp(nn.Module):
    r"""
    Sinkhorn-Knopp algorithm for computing the optimal transport between two
    probability distributions.
    
    Args:
        num_iters (int): number of iterations to perform
        epsilon (float): regularization parameter
    """
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits, labels_hist=None):
        _got_dist = dist.is_initialized() and dist.get_world_size() > 1

        if _got_dist:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        Q = logits / self.epsilon
        Q -= torch.max(Q)
        Q = torch.exp(Q).t()
        Q[torch.isinf(Q)] = 0  # replace inf with 0

        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if _got_dist:
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        if labels_hist is not None:
            # compute the target transport distribution
            W = 1 / labels_hist.unsqueeze(1)
            W /= torch.sum(W)
        else:
            W = K

        for _ in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/w
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if _got_dist:
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= W

            # normalize each column: total weight per sample must be 1/B
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.T
    

# class SinkhornKnopp(nn.Module):
#     r"""
#     Sinkhorn-Knopp algorithm for computing the optimal transport between two
#     probability distributions.
    
#     Args:
#         num_iters (int): number of iterations to perform
#         epsilon (float): regularization parameter
#     """
#     def __init__(self, num_iters=3, epsilon=0.05):
#         super().__init__()
#         self.num_iters = num_iters
#         self.epsilon = epsilon

#     @torch.no_grad()
#     def forward(self, logits):
#         B, K = logits.shape
        
#         Q = logits / self.epsilon
#         M = torch.max(Q)
#         Q -= M
#         Q = torch.exp(Q).t()

#         Q[torch.isinf(Q)] = 0  # replace inf with 0
#         sum_Q = torch.sum(Q)
#         Q /= sum_Q

#         c = torch.ones(B).to(Q.device) / B  # Samples
#         r = torch.ones(K).to(Q.device) / K  # Classes 

#         for _ in range(self.num_iters):
#             u = torch.sum(Q, dim=1)
#             Q *= (r / u).unsqueeze(1)
#             Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

#         Q /= torch.sum(Q, dim=0, keepdim=True)
#         return Q.t()


class ClassAwareContrastiveWithoutMemory(nn.Module):

    def __init__(self, args):
        super(ClassAwareContrastiveWithoutMemory, self).__init__()
        self.args = args
        self.id_temperature = args.id_temp
        self.ood_temperature = args.ood_temp

    def forward(self, features, max_idx, max_probs, reduction="mean"):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] // 2
        max_idx = max_idx.contiguous().view(-1, 1)
        mask = torch.eq(max_idx, max_idx.T).float().to(device)
        max_probs = max_probs.contiguous().view(-1, 1)
        score_mask = torch.matmul(max_probs, max_probs.T)
        score_mask = score_mask.fill_diagonal_(1)
        mask = mask.mul(score_mask)

        if self.args.select_threshold > 0:
            select = (max_probs >= self.args.select_threshold).float()
            select_matrix = torch.matmul(select, select.T)
            select_matrix.fill_diagonal_(1)
            mask = mask * select_matrix

        # compute logits
        cos_dist = torch.matmul(features, features.T)
        if self.ood_temperature is not None:
            mask_ood = max_idx.view(-1) >= self.args.num_labeled_classes
            mask_ood = mask_ood.repeat(2)
            cos_dist[~mask_ood] /= self.id_temperature
            cos_dist[mask_ood] /= self.ood_temperature
            anchor_dot_contrast = cos_dist
        else:
            anchor_dot_contrast = cos_dist / self.id_temperature

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size)

        if reduction == "mean":
            loss = loss.mean()

        return loss
    
class ClassAwareContrastive(nn.Module):
    r"""
    Class-aware contrastive loss.
    
    Args:
        args (argparse.Namespace): parsed command-line arguments.
    """
    def __init__(self, args):
        super(ClassAwareContrastive, self).__init__()
        self.args = args
        self.id_temperature = args.id_temp
        self.ood_temp_schedule = iter(torch.linspace(args.ood_temp, args.id_temp, args.iters_per_epoch * 20))
        # self.ood_temp_schedule = None
        self.ood_temperature = args.ood_temp

    def forward(self, features, max_idx, max_probs, select=None, reduction="mean"):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = self.args.batch_size
        
        max_idx = max_idx.contiguous().view(-1, 1)
        mask = torch.eq(max_idx[:2*batch_size], max_idx.T).float().to(device)
        max_probs = max_probs.contiguous().view(-1, 1)
        score_mask = torch.matmul(max_probs[:2*batch_size], max_probs.T)
        score_mask = score_mask.fill_diagonal_(1)
        mask = mask.mul(score_mask)

        if select is not None:
            select = select.contiguous().view(-1, 1).float()
            # select = (max_probs >= self.args.select_threshold).float()
            select_matrix = torch.matmul(select[:2*batch_size], select.T)
            select_matrix.fill_diagonal_(1)
            mask = mask * select_matrix

        # compute logits
        cos_dist = torch.matmul(features[:2*batch_size], features.T)
        # if self.ood_temperature is not None:
        #     mask_ood = max_idx.view(-1)[:batch_size] >= self.args.num_labeled_classes
        #     mask_ood = mask_ood.repeat(2)
        #     cos_dist[~mask_ood] /= self.id_temperature
        #     cos_dist[mask_ood] /= self.ood_temperature
        #     anchor_dot_contrast = cos_dist
        if self.ood_temp_schedule is not None:
            try:
                self.ood_temperature = next(self.ood_temp_schedule)
            except StopIteration:
                self.ood_temperature = self.id_temperature
                self.ood_temp_schedule = None
        mask_ood = max_idx.view(-1)[:batch_size] >= self.args.num_labeled_classes
        mask_ood = mask_ood.repeat(2)
        cos_dist[~mask_ood] /= self.id_temperature
        cos_dist[mask_ood] /= self.ood_temperature
        anchor_dot_contrast = cos_dist

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # loss
        loss = - mean_log_prob_pos
        
        if reduction == "mean":
            loss = loss.mean()

        return loss
    

class GenClassAwareContrastive(nn.Module):
    r"""
    Class-aware contrastive loss.
    
    Args:
        args (argparse.Namespace): parsed command-line arguments.
    """
    def __init__(self, args):
        super(GenClassAwareContrastive, self).__init__()
        self.args = args
        self.id_temperature = args.id_temp
        self.ood_temp_schedule = iter(torch.linspace(args.ood_temp, args.id_temp, args.iters_per_epoch * 20))
        # self.ood_temp_schedule = None
        self.ood_temperature = args.ood_temp

    def forward(self, features, probs, select=None, reduction="mean"):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = self.args.batch_size
        
        probs = torch.cat([probs[:batch_size], probs], dim=0)

        anchor_labels = probs[:2*batch_size]
        contrast_labels = probs

        anchor_norm = torch.norm(anchor_labels, p=2, dim=-1, keepdim=True) # [anchor_N, 1]
        contrast_norm = torch.norm(contrast_labels, p=2, dim=-1, keepdim=True) # [contrast_N, 1]
        
        deno = torch.mm(anchor_norm, contrast_norm.T)
        mask = torch.mm(anchor_labels, contrast_labels.T) / deno # cosine similarity: [anchor_N, contrast_N]

        if select is not None:
            select = select.contiguous().view(-1, 1).float()
            # select = (max_probs >= self.args.select_threshold).float()
            select_matrix = torch.matmul(select[:2*batch_size], select.T)
            select_matrix.fill_diagonal_(1)
            mask = mask * select_matrix

        logits_mask = torch.ones_like(mask)
        # logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:2*batch_size], features.T),
            self.id_temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # loss
        loss = - mean_log_prob_pos
        
        if reduction == "mean":
            loss = loss.mean()

        return loss
    
class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        # self.cos_dist = nn.CosineSimilarity(dim=1, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])
            # distances = 1 - self.cos_dist(student_output, student_output[I])
            loss = -torch.log(distances + eps).mean()
        return loss


class KNearestNeighborEntropyLoss(nn.Module):
    """K-nearest neighbor entropy loss"""

    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_distance_matrix(self, x):
        """
        Compute the pairwise distance matrix for a batch of L2-normalized vectors.
        """
        dists = torch.cdist(x, x)
        return dists

    def k_nearest_neighbors(self, dists):
        """
        Compute the indices of the k-nearest neighbors for each element in a distance matrix.
        """
        _, indices = torch.sort(dists, dim=1)
        return indices[:, 1 : self.k + 1]

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            dists = self.pairwise_distance_matrix(student_output)
            knn_indices = self.k_nearest_neighbors(dists)
            knn_dists = torch.gather(dists, dim=1, index=knn_indices)
            knn_dists_mean = torch.mean(knn_dists, dim=1)
            loss = -torch.log(knn_dists_mean + eps).mean()
        return loss
    
class STKNearestNeighborEntropyLoss(nn.Module):
    """K-nearest neighbor entropy loss"""

    def __init__(self, k=5):
        super().__init__()
        self.k = k
        # self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_distance_matrix(self, x, y):
        """
        Compute the pairwise distance matrix for a batch of L2-normalized vectors.
        """
        dists = torch.mm(x, y.t())
        return dists

    def k_nearest_neighbors(self, dists):
        """
        Compute the indices of the k-nearest neighbors for each element in a distance matrix.
        """
        _, indices = torch.topk(dists, k=self.k, dim=1, sorted=False)
        return indices

    def forward(self, student_output, teacher_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            dists = self.pairwise_distance_matrix(student_output, teacher_output)
            knn_indices = self.k_nearest_neighbors(dists)
            knn_dists = torch.gather(dists, dim=1, index=knn_indices)
            knn_dists_mean = torch.mean(knn_dists, dim=1)
            loss = -torch.log(knn_dists_mean + eps).mean()
        return loss
