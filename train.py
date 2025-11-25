import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_contrast_transform
from data.get_datasets import get_datasets, get_class_splits
from model.model import CACGCD

from util.general_utils import AverageMeter, ContrastiveLearningViewGenerator, WarmUpCosineAnnealingLR, get_params_groups, init_experiment, seed_everything
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root

def train(model, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(model)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = AdamW(params_groups, lr=args.lr, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )
    # exp_lr_scheduler = WarmUpCosineAnnealingLR(
    #         optimizer,
    #         T_max=args.epochs,
    #         warmup=2,
    #         eta_min=args.lr * 1e-3,
    #     )

    # # inductive
    # best_test_acc_lab = 0
    # best_test_acc_ubl = 0
    # best_test_acc_all = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0
    # best_train_acc_all = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        select_record = AverageMeter()

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                loss, select = model(images, class_labels, mask_lab)

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            select = select.sum() / select.size(0)
            select_record.update(select.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if args.ema:
                model.update_ema()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f} [{:.5f}] select {:.5f} [{:.5f}]'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), loss_record.avg, select.item(), select_record.avg))

        mod = model.p_model / torch.max(model.p_model, dim=-1)[0]
        threshold = model.time_p * mod
        threshold = threshold.detach()
        args.logger.info('Threshold: {}'.format(threshold))
        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        if args.ema:
            all_acc, old_acc, new_acc = test(model.model_ema, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        else:
            all_acc, old_acc, new_acc = test(model, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        # args.logger.info('Testing on disjoint test set...')
        # if args.ema:
        #     all_acc_test, old_acc_test, new_acc_test = test(model.model_ema, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        # else:
        #     all_acc_test, old_acc_test, new_acc_test = test(model, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        

        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()
        
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        # if all_acc_test > best_test_acc_all:

        #     args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        #     torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

        #     # inductive
        #     best_test_acc_lab = old_acc_test
        #     best_test_acc_ubl = new_acc_test
        #     best_test_acc_all = all_acc_test
        #     # transductive            
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc

        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2b'])

    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--proto_delay', type=float, default=0.9)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--id_temp', default=0.1, type=float)
    parser.add_argument('--ood_temp', default=0.2, type=float)
    parser.add_argument('--select_threshold', default=0., type=float)

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    # seed_everything(0)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['cacgcd'])
    args.logger.info(args)
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 2
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_weak_transform, train_strong_transform, test_transform = get_contrast_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=[train_weak_transform, train_strong_transform], n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=512, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=512, shuffle=False, pin_memory=False)

    args.iters_per_epoch = len(train_loader)
    args.steps = args.epochs * args.iters_per_epoch

    # 统计有标记数据的分布
    # for batch_idx, (images, class_labels, uq_idxs, mask_lab) in enumerate(tqdm(train_loader)):
    #     mask_lab = mask_lab[:, 0]
    #     if batch_idx == 0:
    #         label_count = torch.zeros(args.num_labeled_classes)
    #     for i in range(args.num_labeled_classes):
    #         label_count[i] += torch.sum(class_labels == i)
    # label_count = label_count / torch.sum(label_count)
    # args.label_count = label_count
    # print(label_count)
    # 统计有标记数据的分布
    # label_count = torch.zeros(args.num_labeled_classes)
    # for images, class_labels, _, mask_lab in tqdm(train_loader):
    #     mask_lab = mask_lab[:, 0]
    #     label_count += torch.bincount(class_labels, minlength=args.num_labeled_classes)

    # label_count /= torch.sum(label_count)
    # args.label_count = label_count
    # print(label_count)

    # ----------------------
    # MODEL BUILD
    # ----------------------
    model = CACGCD(args).to(device)
    # torch.set_float32_matmul_precision('high')
    args.logger.info('model build')

    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
