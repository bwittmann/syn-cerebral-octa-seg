"""Script for training."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import monai
from torch.utils.tensorboard import SummaryWriter

from syn_cerebral_octa_seg.utils import get_config, get_meta_data, write_json
from syn_cerebral_octa_seg.dataset import get_loaders
from syn_cerebral_octa_seg.models import get_model
from syn_cerebral_octa_seg.trainer import Trainer


def train(config, args):
    device = torch.device(config['device']) if 'cuda' in config['device'] else torch.device('cpu')

    # get data loaders
    loaders = get_loaders(config)

    def lr_schedule(step):
        if step < (config['epochs'] - config['epochs_decay']):
            return 1
        else:
            return (config['epochs'] - step) * (1 / max(1, config['epochs_decay']))

    # get model and metric
    model = get_model(config, lr_schedule).to(device=device)

    # analysis of model trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num params: {num_params}')

    # load checkpoint if applicable
    if args.resume is not None and args.pre_train == False:
        checkpoint = torch.load(Path(args.resume))

        # unpack and load content
        model.load_state_dict(checkpoint['model_state_dict'])
        model._optim_s.load_state_dict(checkpoint['optim_s_state_dict'])
        model._sched_s.load_state_dict(checkpoint['sched_s_state_dict'])

        epoch = checkpoint['epoch']
        metric_start_val = checkpoint['metric_max_val']
    elif args.resume is not None and args.pre_train == True:
        checkpoint = torch.load(Path(args.resume))
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = 0
        metric_start_val = 0
    else:
        epoch = 0
        metric_start_val = 0

    # init logging
    path_to_run = Path(args.path_to_runs) / config['experiment_name'] if args.path_to_runs else Path(os.getcwd()) / 'runs' / config['experiment_name']
    path_to_run.mkdir(exist_ok=True)

    # get meta data and write config to run
    config.update(get_meta_data())
    write_json(config, path_to_run / 'config.json')

    # init tensorboard logging
    writer = SummaryWriter(log_dir=path_to_run)

    # build trainer and start training
    trainer = Trainer(
        config, loaders, model, path_to_run, writer, device, metric_start_val, epoch
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # change working dir
    working_dir = Path(__file__).parent.resolve().parent
    os.chdir(working_dir)

    # get args and config
    parser.add_argument("--resume", type=str, help="Path to checkpoint to use.", default=None)
    parser.add_argument('--pre_train', action='store_true', help='Use solely weights.')
    parser.add_argument("--path_to_runs", type=str, help="Path to runs dir.", default='')
    parser.add_argument("--config", type=str, help="Name of the config to use.", default='config')

    args = parser.parse_args()
    config = get_config(args.config)

    # to get reproducable results
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    monai.utils.set_determinism(seed=config['seed'])
    random.seed(config['seed'])

    torch.backends.cudnn.benchmark = False  # performance vs. reproducibility
    torch.backends.cudnn.deterministic = True

    train(config, args)
