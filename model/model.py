import copy
import math
import torch
import torch.nn as nn
from model.loss import ClassAwareContrastive, ClassAwareContrastiveWithoutMemory, KNearestNeighborEntropyLoss, KoLeoLoss, STKNearestNeighborEntropyLoss, SinkhornKnopp
import torch.distributed as dist

from util.general_utils import concat_all_gather
from vision_transformer import vit_base

class ProjectHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, args=None):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            self.mlp = nn.Sequential(*layers)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        # self.prototype = Prototype(out_dim, bottleneck_dim, momentum=0.9, steps=args.steps)
        self.prototype = Prototype(out_dim, bottleneck_dim, momentum=0.9)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x_proj = nn.functional.normalize(x_proj, dim=-1, p=2)
        if not self.training:
            return self.prototype(x_proj)
        return x_proj


class Prototype(nn.Module):
    def __init__(self, K, D, momentum=0.9, steps=None):
        super(Prototype, self).__init__()

        self.momentum = momentum
        self.K, self.D = K, D
        self.momentum_scheduler = None
        # if steps is not None:
        #     self.momentum_scheduler = iter(torch.linspace(0.9, 0.999, steps=steps))
        # self.proto = nn.utils.weight_norm(nn.Linear(D, K, bias=False))
        # self.proto.weight_g.data.fill_(1)
        # self.proto.weight_g.requires_grad = False
        # self.proto.weight_v.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        proto = torch.empty(K, D, requires_grad=False)#.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # nn.init.kaiming_uniform_(self.proto.weight_v, a=math.sqrt(5))
        # nn.init.normal_(proto)
        # nn.init.orthogonal_(proto)
        nn.init.trunc_normal_(proto, std=.02)
        self.register_buffer('proto', nn.functional.normalize(proto, dim=-1))

        # torch.nn.init.orthogonal_(self.proto.weight_v)
        # torch.nn.init.trunc_normal_(self.proto.weight_v, std=.02)
        # torch.nn.init.normal_(self.proto.weight_v)

    @torch.no_grad()
    def update(self, features, labels, weights=None):
        if torch.distributed.is_initialized():
            features = concat_all_gather(features)
            labels = concat_all_gather(labels)
            if weights is not None:
                weights = concat_all_gather(weights)
        # for i, l in enumerate(labels):
        #     alpha = 1 - (1. - 0.9) * weights[i]
        #     self.proto[l] = nn.functional.normalize(alpha * self.proto[l].data + (1. - alpha) * features[i], dim=-1)
        if self.momentum_scheduler is None:
            self.momentum = 0.9
        else:
            self.momentum = next(self.momentum_scheduler)

        # Compute the unique labels and their counts.
        unique_labels, inverse_indices = torch.unique(labels.squeeze(), return_inverse=True)
        label_counts = torch.bincount(inverse_indices)

        # Compute the new features weighted.
        if weights is not None:
            # weights = (weights / 0.5)
            weighted_features = features * weights.view(-1, 1)
        else:
            weighted_features = features

        # Compute the sum of weighted features for each unique label.
        summed_weighted_features = torch.zeros((len(unique_labels), features.size(1)), device=features.device)
        summed_weighted_features.scatter_add_(0, inverse_indices.view(-1, 1).expand_as(weighted_features), weighted_features)

        # Compute the average of summed_weighted_features for each unique label.
        averaged_weighted_features = summed_weighted_features / label_counts.view(-1, 1)
        averaged_weighted_features = nn.functional.normalize(averaged_weighted_features, dim=-1)
        
        # Update the proto weights for the unique labels with the averaged weighted features.
        self.proto[unique_labels] = nn.functional.normalize(self.proto[unique_labels] * self.momentum + (1 - self.momentum) * averaged_weighted_features, dim=-1)
        # self.proto.weight_v[unique_labels] = self.proto.weight_v[unique_labels] * self.momentum + (1 - self.momentum) * averaged_weighted_features

    def forward(self, x):
        with torch.no_grad():
            # x = self.proto(x)
            x = x @ self.proto.T
        return x


class CACGCD(nn.Module):
    r"""
    CACGCD: Class-Aware Contrastive for Generalized Category Discovery
    
    Args:
        args (argparse.Namespace): Arguments.
    """
    def __init__(self, args):
        super(CACGCD, self).__init__()

        self.args = args
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        # backbone = vit_base(img_size=518, patch_size=14, init_values=1.0)
        # backbone = vit_base()
        # state_dict = torch.load('/home/sybapp/PycharmProjects/SimGCDDINOV2BAK/dev_outputs/simgcd/log/cub_ijepa_(29.06.2023_|_07.270)/checkpoints/model.pt', map_location='cpu')['model']
        # state_dict = {k.replace('student.', ''): v for k, v in state_dict.items()}
        # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # state_dict = {k.replace('module.blocks.', 'blocks.0.'): v for k, v in state_dict.items()}
        # state_dict = {k.replace('teacher.', ''): v for k, v in state_dict.items()}
        # msg = backbone.load_state_dict(state_dict, strict=False)
        # args.logger.info(msg)

        # freeze backbone except for the last block
        for name, m in backbone.named_parameters():
            m.requires_grad = False
            if 'block' in name:
                block_num = int(name.split('.')[2])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
        
        projector = ProjectHead(in_dim=args.feat_dim, out_dim=args.num_classes, nlayers=args.num_mlp_layers, use_bn=True, args=args)
        self.model = nn.Sequential(backbone, projector)
        self.prototype = projector.prototype
        
        if args.ema and args.ema_decay is not None:
            self.ema_decay = args.ema_decay
            self.model_ema = copy.deepcopy(self.model)
            
        self.sinkhorn = SinkhornKnopp()
        self.koleoloss = KoLeoLoss()
        # self.koleoloss = STKNearestNeighborEntropyLoss()
        self.criterion = ClassAwareContrastive(args=args)
        
        self.K = 4096
        self._initialize_buffers()

    def _initialize_buffers(self):
        r"""
        Initialize the queue and the queue pointer.
        """
        self.register_buffer('queue', torch.zeros(self.K, 256))
        self.register_buffer('queue_lab', torch.zeros(self.K, dtype=torch.int64) - 1)
        self.register_buffer('queue_mask', torch.zeros(self.K).bool())
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        # adaptive threshold
        self.register_buffer('p_model', torch.ones(self.args.num_classes) / self.args.num_classes)
        self.register_buffer('label_hist', torch.ones(self.args.num_classes) / self.args.num_classes)
        # unlabeled_dis = torch.ones(self.args.num_unlabeled_classes) / self.args.num_unlabeled_classes
        # labeled_dis = self.args.label_count
        # hist = torch.cat([labeled_dis, unlabeled_dis], dim=0)
        # self.register_buffer('label_hist', hist / hist.sum())

        self.register_buffer('time_p', self.p_model.mean())
        # self.register_buffer('time_p', torch.tensor(0.5))
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, features, labels, mask_lab):
        r"""
        Update the queue and the queue pointer.

        Args:
            features (torch.Tensor): Features to be added to the queue.
            labels (torch.Tensor): Labels to be added to the queue.
            mask_lab (torch.Tensor): Mask for labels to be added to the queue.
        """
        if dist.is_initialized() and dist.get_world_size() > 1:
            # gather keys before updating queue
            features = concat_all_gather(features)
            labels = concat_all_gather(labels)
            mask_lab = concat_all_gather(mask_lab)
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = features
        self.queue_lab[ptr:ptr + batch_size] = labels
        self.queue_mask[ptr:ptr + batch_size] = mask_lab
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
        
    @torch.no_grad()
    def update_ema(self):
        r"""
        Update the EMA model.
        """
        for param_train, param_eval in zip(self.model.parameters(), self.model_ema.parameters()):
            param_eval.copy_(param_eval * self.ema_decay + param_train.detach() * (1 - self.ema_decay))
        for buffer_train, buffer_eval in zip(self.model.buffers(), self.model_ema.buffers()):
            buffer_eval.copy_(buffer_train)

    @torch.no_grad()
    def update(self, probs_x_ulb):
        self.m = 0.99
        if dist.is_initialized() and dist.get_world_size() > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1, keepdim=True)

        self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        
        self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0])
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())
    
    @torch.no_grad()
    def masking(self, logits_x_ulb):
        probs_x_ulb = logits_x_ulb.detach()

        self.update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx])
        return mask

    def forward(self, images, labels=None, mask_lab=None):
        if not self.training:
            if self.args.ema:
                return self.model_ema(images)
            return self.model(images)
        
        batch_size = images.shape[0] // 2
        
        images_w, images_s = images.chunk(2)
        contrast_features = self.model(images_s)
        weak_features = self.model(images_w)
        anchor_features = weak_features.clone().detach()
        # anchor_features = self.model_ema(images_w).detach()
        #anchor_features = anchor_features_.detach()
        
        queue_lab, queue_features, queue_mask = self.get_memory_bank()

        all_anchor_features = torch.cat([anchor_features, queue_features], dim=0)
        all_features = torch.cat([contrast_features, all_anchor_features], dim=0)
        all_class_labels = torch.cat([labels, queue_lab], dim=0)
        all_mask_lab = torch.cat([mask_lab, queue_mask], dim=0)
        
        with torch.no_grad():
            all_anchor_logits = self.prototype(all_anchor_features)
            # all_sk_logits = self.sinkhorn(all_anchor_logits, self.label_hist)
            all_sk_logits = self.sinkhorn(all_anchor_logits)
            # all_sk_logits = nn.functional.softmax(all_sk_logits / 0.1, dim=-1)
            all_select = self.masking(all_sk_logits)

            # max_idx, max_probs, select for queue
            all_max_probs, all_max_idx = all_sk_logits.max(1)
            all_max_idx[all_mask_lab] = all_class_labels[all_mask_lab]
            all_max_probs[all_mask_lab] = 1
            all_select[all_mask_lab] = True

            # max_idx, max_probs, select for anchor
            max_idx = all_max_idx[:batch_size]
            max_probs = all_max_probs[:batch_size]
            select = all_select[:batch_size]

            # max_idx, max_probs, select for all
            all_max_idx = torch.cat([max_idx, all_max_idx], dim=0)
            all_max_probs = torch.cat([max_probs, all_max_probs], dim=0)
            all_select = torch.cat([select, all_select], dim=0)

        with torch.cuda.amp.autocast(enabled=False):
            loss = self.criterion(features=all_features, max_idx=all_max_idx, max_probs=all_max_probs, select=all_select)
            # loss += 0.5 * (self.koleoloss(weak_features) + self.koleoloss(contrast_features))
            loss += self.koleoloss(contrast_features)
        self._dequeue_and_enqueue(features=anchor_features, labels=labels, mask_lab=mask_lab)
        # self.prototype.update(features=anchor_features[select], labels=max_idx[select], weights=max_probs[select])
        # self.prototype.update(features=anchor_features[select], labels=max_idx[select])
        self.prototype.update(features=anchor_features, labels=max_idx, weights=max_probs)
        return loss, select

    def get_memory_bank(self):
        queue_lab = self.queue_lab.clone().detach()
        queue_features = self.queue.clone().detach()
        queue_mask = self.queue_mask.clone().detach()

        idx = queue_lab != -1

        queue_lab = queue_lab[idx]
        queue_features = queue_features[idx]
        queue_mask = queue_mask[idx]

        return queue_lab,queue_features,queue_mask
