"""File containing model related functionality."""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from syn_cerebral_octa_seg.cldice_loss import soft_dice_cldice


def get_model(config, lr_schedule):
    model = OCTASegNet3D(config, lr_schedule)
    return model

def get_loss_dict(iter, alpha, smooth):
    loss = {
        'seg_dice': DiceLoss(),
        'seg_cldice': soft_dice_cldice(iter_=iter, alpha=alpha, smooth=smooth),
    }
    return  loss

def cl_dice(v_p, v_l):
    def cl_score(v, s):
        return np.sum(v*s)/np.sum(s)

    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def estimate_metrics(pred, gt, threshold=0.5, fast=False, dilate=0, small_regions=[], large_regions=[], comp_region=None):
    pred_labels = (pred >= threshold).float().cpu()

    # dilate predicted labels
    for _ in range(dilate):
        pred_labels = torch.clamp(F.conv3d(pred_labels.int(), torch.ones([1, 1, 3, 3, 3]).int(), padding=(1, 1, 1)), 0, 1)

    # estimate metrics over all regions including the complete image ('a')
    all_metrics = defaultdict(list)
    for region, region_id in ([(reg, 's') for reg in small_regions] + [(reg, 'l') for reg in large_regions] + [(comp_region, 'a')]):
        metrics = {}

        if fast:    # skip metrics computation for a drastic speed-up
            metrics['recall_tpr_sensitivity'] = 0
            metrics['fpr'] = 0
            metrics['precision'] = 0
            metrics['specificity'] = 0
            metrics['jaccard_iou'] = 0
            metrics['dice'] = 0
            metrics['cldice'] = 0
            metrics['accuracy'] = 0
            metrics['roc_auc'] = 0
            metrics['pr_auc_ap'] = 0
            all_metrics[region_id].append(metrics)
            continue

        # select region
        pred_labels_cropped = pred_labels[:, :, region[0]:region[1], region[2]:region[3], region[4]:region[5]]
        pred_cropped = pred[:, :, region[0]:region[1], region[2]:region[3], region[4]:region[5]]
        gt_cropped = gt[:, :, region[0]:region[1], region[2]:region[3], region[4]:region[5]]

        # estimate metrics
        tn, fp, fn, tp = confusion_matrix(
            gt_cropped.flatten().cpu().clone().numpy(), 
            pred_labels_cropped.flatten().cpu().clone().numpy(), 
            labels=[0, 1]
        ).ravel()

        roc_auc = roc_auc_score(
            gt_cropped.flatten().cpu().clone().detach().numpy(),
            pred_cropped.flatten().cpu().clone().detach().numpy()
        )

        pr_auc = average_precision_score(
            gt_cropped.flatten().cpu().clone().detach().numpy(),
            pred_cropped.flatten().cpu().clone().detach().numpy()
        )

        cldice = cl_dice(
            pred_labels_cropped.squeeze().cpu().clone().detach().byte().numpy(), 
            gt_cropped.squeeze().cpu().clone().detach().byte().numpy()
        )

        metrics['recall_tpr_sensitivity'] = tp / (tp + fn)
        metrics['fpr'] = fp / (fp + tn)
        metrics['precision'] = tp / (tp + fp)
        metrics['specificity'] = tn / (tn + fp)
        metrics['jaccard_iou'] = tp / (tp + fp + fn)
        metrics['dice'] = (2 * tp) / (2 * tp + fp + fn)
        metrics['cldice'] = cldice
        metrics['accuracy'] = (tp + tn) / (tn + fp + tp + fn)
        metrics['roc_auc'] = roc_auc
        metrics['pr_auc_ap'] = pr_auc
        metrics['volume'] = gt_cropped.numel()
        all_metrics[region_id].append(metrics)

    return all_metrics


class OCTASegNet3D(nn.Module):
    def __init__(self, config, lr_schedule):
        super().__init__()
        self._bs = config['batch_size']
        self._max_epoch = config['epochs']
        self._config = config

        # load individual models
        in_channels = 4 if self._config['use_doppler'] else 1
        self._segmenter = UNet(
            spatial_dims=3, in_channels=in_channels, out_channels=1,
            channels=self._config['seg']['channels'], strides=self._config['seg']['strides'],
            act=self._config['seg']['activation'], norm=self._config['seg']['normalization'], 
            dropout=self._config['seg']['dropout']
        )
        
        # get relevant components for optimization
        self._scaler = GradScaler()
        self._optim_s = torch.optim.Adam(self._segmenter.parameters(), lr=float(self._config['seg']['lr']), betas=(0.5 , 0.999))
        self._sched_s = torch.optim.lr_scheduler.LambdaLR(self._optim_s, lr_schedule)

        # get loss functions
        loss_functions = get_loss_dict(self._config['seg']['iter'], self._config['seg']['alpha'], self._config['seg']['smooth'])
        self._loss_function = loss_functions['seg_dice'] if not self._config['seg']['cldice'] else loss_functions['seg_cldice']

    def forward(self, real_data, synthetic_data=None, synthetic_label=None, inference=False):
        if inference:   # just feed through segmenter
            real_seg = self._segmenter(real_data).sigmoid()
            return real_seg

        self._optim_s.zero_grad()
        syn_seg = self._segmenter(synthetic_data).sigmoid()

        if self._config['seg']['cldice']:
            loss_seg = self._loss_function(synthetic_label.float(), syn_seg)
        else:
            loss_seg = self._loss_function(syn_seg, synthetic_label)

        self._scaler.scale(loss_seg).backward()
        self._scaler.step(self._optim_s)
        self._scaler.update()

        loss_dict = {
            'seg': loss_seg
        }
        return syn_seg, loss_dict