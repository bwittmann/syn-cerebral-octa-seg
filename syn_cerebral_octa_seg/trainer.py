"""Trainer class."""

import torch
from torch.cuda.amp import autocast
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from syn_cerebral_octa_seg.models import estimate_metrics


class Trainer:
    def __init__(
        self, config, loaders, model, path_to_run, writer, device, 
        metric_start_val, epoch
    ):
        self._config = config
        self._train_loader = loaders['train']
        self._val_loader = loaders['val']

        self._model = model
        self._device = device
        self._path_to_run = path_to_run
        self._writer = writer
        self._metric_max_val = metric_start_val
        self._epoch = epoch

    def run(self):
        if self._epoch == 0:   # for initial performance estimation
            self._validate(0, 'val')

        for epoch in range(self._epoch + 1, self._config['epochs'] + 1):
            self._train_one_epoch(epoch)

            # log learning rates
            self._write_to_logger(
                epoch, 'lr',
                seg=self._model._optim_s.param_groups[0]['lr']
            )

            if epoch % self._config['val_interval'] == 0:
                self._validate(epoch, 'val')

            self._save_checkpoint(epoch, 'model_last.pt')


    def _train_one_epoch(self, num_epoch):
        print(f'train epoch: {num_epoch}')
        self._model.train()

        loss_seg_comb = 0
        train_dice_comb = 0
        train_cldice_comb = 0
        for synthetic_data, synthetic_label in tqdm(self._train_loader):
            synthetic_data, synthetic_label = synthetic_data.to(device=self._device), synthetic_label.to(device=self._device)

            # make prediction
            with autocast():
                out, loss_dict = self._model(None, synthetic_data=synthetic_data, synthetic_label=synthetic_label)

            metrics = estimate_metrics(out, synthetic_label, fast=True)

            loss_seg_comb += loss_dict['seg'].item()
            train_dice_comb += metrics['a'][0]['dice']
            train_cldice_comb += metrics['a'][0]['cldice']

        self._model._sched_s.step()

        loss_seg = loss_seg_comb / len(self._train_loader)
        train_dice = train_dice_comb / len(self._train_loader)
        train_cldice = train_cldice_comb / len(self._train_loader)

        self._write_to_logger(
            num_epoch, 'train', 
            loss_seg_seg=loss_seg,
            train_dice=train_dice,
            train_cldice=train_cldice,
        )

    @torch.no_grad()
    def _validate(self, num_epoch, split):
        print(f'validate epoch: {num_epoch}, {split}')
        self._model.eval()
        loader = self._val_loader

        val_s_comb = torch.zeros(10)
        val_l_comb = torch.zeros(10)
        val_a_comb = torch.zeros(10)
        volume_comb = torch.zeros(3)
        for data, label, small_regions, large_regions, comp_region in tqdm(loader):
            data, label = data.to(device=self._device), label.to(device=self._device)

            # make prediction
            with autocast():
                out = sliding_window_inference(
                    inputs=data,
                    roi_size=self._config['augmentation']['patch_size'],
                    sw_batch_size=self._config['batch_size'],
                    predictor=self._model,
                    overlap=self._config['sliding_window_overlap'],
                    mode=self._config['sliding_window_mode'],
                    padding_mode='constant',
                    inference=True
                )

                # estimate metrics
                metrics = estimate_metrics(
                    out, label, threshold=self._config['threshold'], 
                    small_regions=small_regions, large_regions=large_regions, comp_region=comp_region
                )

                # aggregate results on different regions & weigh them based on their volume to estimate accurate mean
                for reg in metrics['s']:
                    reg_metrics = torch.tensor(list(reg.values()))[:-1]
                    reg_volume = torch.tensor(list(reg.values()))[-1]
                    val_s_comb += (reg_metrics * reg_volume)
                    volume_comb[0] += reg_volume.item()

                for reg in metrics['l']:
                    reg_metrics = torch.tensor(list(reg.values()))[:-1]
                    reg_volume = torch.tensor(list(reg.values()))[-1]
                    val_l_comb += (reg_metrics * reg_volume)
                    volume_comb[1] += reg_volume.item()

                for reg in metrics['a']:
                    reg_metrics = torch.tensor(list(reg.values()))[:-1]
                    reg_volume = torch.tensor(list(reg.values()))[-1]
                    val_a_comb += (reg_metrics * reg_volume)
                    volume_comb[2] += reg_volume.item()
       
        val_s = val_s_comb / volume_comb[0]
        val_l = val_l_comb / volume_comb[1]
        val_a = val_a_comb / volume_comb[2]

        for reg_metrics, reg_id in ([val_s, '_small'], [val_l, '_large'], [val_a, '_all']):
            self._write_to_logger(
                num_epoch, split + reg_id, 
                true_positive_rate=reg_metrics[0],
                false_positive_rate=reg_metrics[1],
                precision=reg_metrics[2],
                specificity=reg_metrics[3],
                iou=reg_metrics[4],
                dice=reg_metrics[5],
                cldice=reg_metrics[6],
                accuracy=reg_metrics[7],
                roc_auc=reg_metrics[8],
                pr_auc=reg_metrics[9],
            )

        print('\n', split, '\t',  'cldice:', val_a[6].item(), '\t', 'dice:', val_a[5].item(), '\n')

        # check if new best checkpoint
        if val_a[5] >= self._metric_max_val and split == 'val':
            self._metric_max_val = val_a[5].item()
            self._save_checkpoint(num_epoch, f'model_best_{self._metric_max_val:.3f}.pt')


    def _write_to_logger(self, num_epoch, category, **kwargs):
        for key, value in kwargs.items():
            name = category + '/' + key
            self._writer.add_scalar(name, value, num_epoch)

    def _save_checkpoint(self, num_epoch, name, pretrain=False):
        if not self._config['store_ckpt']:
            return
        
        if pretrain:
            name = 'pre_' + name

        # Delete prior best checkpoint
        if 'best' in name:
            if pretrain:
                [path.unlink() for path in self._path_to_run.iterdir() if 'best' in str(path.name)]
            else:
                [path.unlink() for path in self._path_to_run.iterdir() if 'best' in str(path.name) and 'pre' not in str(path.name)]

        torch.save({
            'epoch': num_epoch,
            'metric_max_val': self._metric_max_val,
            'model_state_dict': self._model.state_dict(),
            'optim_s_state_dict': self._model._optim_s.state_dict(),
            'sched_s_state_dict': self._model._sched_s.state_dict()
        }, self._path_to_run / name)