"""Script to performce inference tailored to our in-house data."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter
from tqdm import tqdm
from torch.cuda.amp import autocast
from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensityRange

from syn_cerebral_octa_seg.utils import read_json, write_nifti, read_nifti, read_tiff
from syn_cerebral_octa_seg.models import get_model


def inference(args):
    path_to_run = Path('./runs/' + args.run)
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
    files = [file for file in Path(args.data_folder).glob('**/*') if file.is_file() and file.suffix in ['.tif', '.nii']]

    # perform inference
    for file in tqdm(files):
        print(f'Processing {file}.')

        try:
            data = read_tiff(file)
        except:
            data = read_nifti(file)

        # transform data
        data = preprocess_image(data)[None][None]

        # make prediction
        with autocast():
            out = sliding_window_inference(
                inputs=data,
                roi_size=config['augmentation']['patch_size'],
                sw_batch_size=config['batch_size'],
                predictor=model,
                overlap=args.overlap,
                mode=config['sliding_window_mode'],
                padding_mode='constant',
                inference=True
            )

        # save segmentation masks
        write_nifti((out >= args.threshold).int().squeeze().cpu().numpy(), args.data_folder / ('seg_' + Path(args.inference).parts[-1] + '.nii'))
        print(f'Finished processing {file}.')

def preprocess_image(
        img, scale_factor=1.53, percentile_lower_pre=2.0915021896362305, percentile_upper_pre=878.1568603515625, 
        percentile_lower_post=0.00019024491484742612, percentile_upper_post=1.0
    ):
    # isotropic voxel size
    interpolated_img = F.interpolate(img[None][None].float(), scale_factor=(scale_factor, 1, 1), mode='trilinear').squeeze()

    # pre scale and clip 
    transform_pre = ScaleIntensityRange(percentile_lower_pre, percentile_upper_pre, 0, 1, clip=True)    # percent [1, 97] of m4 raw (angioTestSet_fullReference_raw_unscaled_m4.tif)
    interpolated_img_transformed = transform_pre(interpolated_img)

    # apply median filter
    interpolated_img_transformed = median_filter(interpolated_img_transformed, size=3)
        
    # post scale and clip
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
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap in sliding window inference scheme.')
    parser.add_argument('--last', action='store_true', help='Use model_last instead of model_best.')
    parser.add_argument('--cpu', action='store_true', help='Utilize cpu.')
    args = parser.parse_args()

    # run inference
    inference(args)