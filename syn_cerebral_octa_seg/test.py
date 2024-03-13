"""Script to evalute performance on val/test sets."""

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from skimage.filters import frangi, threshold_otsu

from syn_cerebral_octa_seg.utils import read_json, write_json, write_nifti, read_nifti
from syn_cerebral_octa_seg.dataset import get_loaders
from syn_cerebral_octa_seg.models import get_model, estimate_metrics


def test(args):
    path_to_run = Path('./runs/' + args.run)
    device = torch.device('cuda:' + str(args.num_gpu))
    config = read_json(path_to_run / 'config.json')

    img_names = ' '.join(config['test_imgs'])
    print('\n', f'Testing {args.run} on {img_names}')

    # get path to checkpoints
    avail_checkpoints = [path for path in path_to_run.iterdir() if 'model_' in str(path)]
    avail_checkpoints.sort(key=lambda x: len(str(x)))
    if args.last:
        path_to_ckpt = avail_checkpoints[0]
    else:
        path_to_ckpt = avail_checkpoints[-1]

    def lr_schedule(step):
        if step < (config['epochs'] - config['epochs_decay']):
            return 1
        else:
            return (config['epochs'] - step) * (1 / max(1, config['epochs_decay']))

    # load model and checkpoint
    model = get_model(config, lr_schedule).to(device=device) 
    checkpoint = torch.load(path_to_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()   

    # create dir to store results
    path_to_results = path_to_run / 'results' / path_to_ckpt.parts[-1][:-3]
    path_to_results.mkdir(parents=True, exist_ok=True)

    loaders = get_loaders(config)
    for split in ['test', 'val']:
        loader = loaders[split]

        test_s_comb = torch.zeros(10)
        test_l_comb = torch.zeros(10)
        test_a_comb = torch.zeros(10)
        volume_comb = torch.zeros(3)
        for idx, (data, label, small_regions, large_regions, comp_region) in enumerate(tqdm((loader))):
            data, label = data.to(device=device), label.to(device=device)

            if args.inference:
                data = read_nifti(args.inference)

            # make prediction
            with autocast():
                out = sliding_window_inference(
                    inputs=data,
                    roi_size=config['augmentation']['patch_size'],
                    sw_batch_size=config['batch_size'],
                    predictor=model,
                    overlap=config['sliding_window_overlap'],
                    mode=config['sliding_window_mode'],
                    padding_mode='constant',
                    inference=True
                )

                # FRANGI
                # out = data.clone()
                # out = torch.tensor(
                #     frangi(
                #         out.squeeze().cpu().numpy(),
                #         sigmas=range(1, 20, 1),
                #         black_ridges=False,
                #         alpha=0.5,
                #         beta=0.5,
                #         gamma=None
                #     )
                # )
                # threshold = 0.005
                # out[out > threshold] = 1
                # out[out <= threshold] = 0
                # out = out[None][None]

                # THRESHOLDING
                # threshold = threshold_otsu(data.clone().squeeze().cpu().numpy())
                # out = data.clone()
                # out[out >= threshold] = 1
                # out[out < threshold] = 0

                # save segmentation mask for visualization purposes
                if args.inference:
                    write_nifti((out >= args.threshold).int().squeeze().cpu().numpy(), path_to_results / ('seg_' + Path(args.inference).parts[-1] + '.nii'))
                    return
                else:
                    write_nifti((out >= args.threshold).int().squeeze().cpu().numpy(), path_to_results / ('seg_' + split + '_' + str(idx) + '.nii'))

                # estimate metrics
                metrics = estimate_metrics(
                    out, label, threshold=args.threshold, dilate=0,
                    small_regions=small_regions, large_regions=large_regions, comp_region=comp_region
                )

                # aggregate results on different regions & weigh them based on their volume to estimate accurate mean
                for reg in metrics['s']:
                    reg_metrics = torch.tensor(list(reg.values()))[:-1]
                    reg_volume = torch.tensor(list(reg.values()))[-1]
                    test_s_comb += (reg_metrics * reg_volume)
                    volume_comb[0] += reg_volume.item()

                for reg in metrics['l']:
                    reg_metrics = torch.tensor(list(reg.values()))[:-1]
                    reg_volume = torch.tensor(list(reg.values()))[-1]
                    test_l_comb += (reg_metrics * reg_volume)
                    volume_comb[1] += reg_volume.item()

                for reg in metrics['a']:
                    reg_metrics = torch.tensor(list(reg.values()))[:-1]
                    reg_volume = torch.tensor(list(reg.values()))[-1]
                    test_a_comb += (reg_metrics * reg_volume)
                    volume_comb[2] += reg_volume.item()

        test_s = test_s_comb / volume_comb[0]
        test_l = test_l_comb / volume_comb[1]
        test_a = test_a_comb / volume_comb[2]

        for reg_metrics, reg_id in ([test_s, '_small'], [test_l, '_large'], [test_a, '_all']):
            results = {
                'test_true_positive_rate': reg_metrics[0].item(),
                'test_false_positive_rate': reg_metrics[1].item(),
                'test_precision': reg_metrics[2].item(),
                'test_specificity': reg_metrics[3].item(),
                'test_iou': reg_metrics[4].item(),
                'test_dice': reg_metrics[5].item(),
                'test_cldice': reg_metrics[6].item(),
                'test_accuracy': reg_metrics[7].item(),
                'test_roc_auc': reg_metrics[8].item(), 
                'test_pr_auc': reg_metrics[9].item()
            }

            # write and log results
            print('\n', split + reg_id, '\t',  'cldice:', results['test_cldice'], '\t', 'dice:',  results['test_dice'], '\n')
            write_json(results, path_to_results / ('results_' + split + reg_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # get args and config
    parser.add_argument('--run', required=True, type=str, help='Name of experiment in syn_cerebral_octa_seg/runs.')
    parser.add_argument('--num_gpu', type=int, default=0, help='Id of GPU to run inference on.')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--last', action='store_true', help='Use model_last instead of model_best.')
    parser.add_argument("--inference", type=str, help="Path to img.", default='')
    args = parser.parse_args()

    # run test/inference
    test(args)