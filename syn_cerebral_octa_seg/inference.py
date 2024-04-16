"""Script to performce inference tailored to our in-house data."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm
from torch.cuda.amp import autocast
from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensityRange

from syn_cerebral_octa_seg.utils import read_json, write_nifti, read_nifti, read_tiff
from syn_cerebral_octa_seg.models import get_model


def inference(args):
    path_to_run = Path('./runs/' + args.run)

    args_modifier = args.run    # for logging purposes
    if args.last:
        args_modifier += '_last'
    if args.ind_per:
        args_modifier += '_indper'
    if args.ensemble:
        args_modifier += '_ensemble'

    device = torch.device('cpu') if args.cpu else torch.device('cuda:' + str(args.num_gpu))
    config = read_json(path_to_run / 'config.json')

    # get path to checkpoints
    avail_checkpoints = [path for path in path_to_run.iterdir() if 'model_' in str(path)]
    avail_checkpoints.sort(key=lambda x: len(str(x)))
    if args.last:
        path_to_ckpt = avail_checkpoints[0]
    else:
        path_to_ckpt = avail_checkpoints[-1]

    # load model and checkpoint
    def lr_schedule(step):
        if step < (config['epochs'] - config['epochs_decay']):
            return 1
        else:
            return (config['epochs'] - step) * (1 / max(1, config['epochs_decay']))
        
    model = get_model(config, lr_schedule).to(device=device) 
    checkpoint = torch.load(path_to_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # load raw data
    files = [file for file in Path(args.data_folder).glob('**/*') if file.is_file() and file.suffix in ['.tif']]

    # perform inference
    for file in files:
        print(f'Processing {file}')
        data = read_tiff(file).astype(np.float32)

        # transform data
        data = preprocess_image(data, ind_percentile=args.ind_per)[None][None].to(device=device)

        # make prediction
        with torch.no_grad():
            with autocast():
                out = sliding_window_inference(
                    inputs=data,
                    roi_size=config['augmentation']['patch_size'],
                    sw_batch_size=config['batch_size'],
                    predictor=model,
                    overlap=args.overlap,
                    mode=config['sliding_window_mode'],
                    padding_mode='constant',
                    progress=True,
                    inference=True
                )

        # save segmentation masks
        write_nifti((out >= args.threshold).int().squeeze().cpu().numpy(), args.data_folder / Path(file.parts[-1].split('.')[0] + '_' + args_modifier + '_seg.nii'))
        write_nifti(data.squeeze().cpu().numpy(), args.data_folder / Path(file.parts[-1].split('.')[0] + '_' + args_modifier + '_data.nii'))

        print(f'Finished processing {file}.', '\n')

def preprocess_image(
        img, scale_factor=1.53, percentile_lower_pre=2.0915021896362305, percentile_upper_pre=878.1568603515625, 
        percentile_lower_post=0.00019024491484742612, percentile_upper_post=1.0, ind_percentile=False
    ):
    # isotropic voxel size (2um)
    interpolated_img = F.interpolate(torch.tensor(img)[None][None], scale_factor=(scale_factor, 1, 1), mode='trilinear').squeeze()

    # pre scale and clip
    if ind_percentile:
        percentile_lower_pre = np.percentile(interpolated_img, 1)
        percentile_upper_pre = np.percentile(interpolated_img, 97)

    transform_pre = ScaleIntensityRange(percentile_lower_pre, percentile_upper_pre, 0, 1, clip=True)    # percent [1, 97] of m4 raw (angioTestSet_fullReference_raw_unscaled_m4.tif)
    interpolated_img_transformed = transform_pre(interpolated_img)

    # apply median filter
    interpolated_img_transformed = median_filter(interpolated_img_transformed, size=3)
        
    # post scale and clip
    if ind_percentile:
        percentile_lower_post = np.percentile(interpolated_img, 1)
        percentile_upper_post = np.percentile(interpolated_img, 99)

    transform_post = ScaleIntensityRange(percentile_lower_post, percentile_upper_post, 0, 1, clip=True) # percent [1, 99] of m4 raw (angioTestSet_fullReference_raw_unscaled_m4.tif)
    interpolated_img_transformed = transform_post(interpolated_img_transformed)

    return interpolated_img_transformed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # get args and config
    parser.add_argument('--data_folder', type=str, required=True, help='Path to raw data. A folder full of files (.nii or .tif).')
    parser.add_argument('--run', required=True, type=str, help='Name of experiment in syn_cerebral_octa_seg/runs.')
    parser.add_argument('--num_gpu', type=int, default=0, help='Id of GPU to run inference on.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for foreground voxels.')
    parser.add_argument('--overlap', type=float, default=0.8, help='Overlap in sliding window inference scheme.')
    parser.add_argument('--last', action='store_true', help='Use model_last instead of model_best.')
    parser.add_argument('--cpu', action='store_true', help='Utilize cpu.')
    parser.add_argument('--ind_per', action='store_true', help='Estimate for each image an individual percentile value.')
    parser.add_argument('--ensemble', action='store_true', help='Use 6-fold ensemble for inference.')
    args = parser.parse_args()

    # run inference
    inference(args)