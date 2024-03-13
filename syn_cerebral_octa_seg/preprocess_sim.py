"""Script to process vessel graphs and simulated artifacts."""

import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from monai.transforms import Compose, RandGaussianNoise, GaussianSmooth, EnsureType, ScaleIntensityRange

from syn_cerebral_octa_seg.utils import write_nifti


def add_tails_and_signal_loss(label, volume_angle, volume_radius, volume_occupancy, args):
    mask = np.zeros_like(label, dtype=float)    # generate output volume
    label = label.astype(float)
    _, R, _ = label.shape

    # transform radius to intensities
    volume_radius = np.clip(volume_radius / args.upper_radius_thresh, 0, 1)

    # exponential signal decay for modeling angle-dependent signal loss
    angle2intensity = np.geomspace(0.1, 1, num=101)
    
    # iteratively add tails by looping over voxels
    for slice_id, label_slice in enumerate(tqdm(label, 'Modeling artifacts', disable=args.dt)):   # loop over voxels
        for row_id, row_elem in enumerate(label_slice):
            for cul_id, elem in enumerate(row_elem):
                if elem == 1:
                    # retrieve relevant metadata
                    angle = volume_angle[slice_id, row_id, cul_id] * 100
                    radius = volume_radius[slice_id, row_id, cul_id]
                    occ_below = volume_occupancy[slice_id, row_id, cul_id]

                    # model angle-dependent intensity
                    if args.angle_correction:
                        angle_correction = torch.sigmoid(args.angle_delta * torch.tensor(radius - args.micro_radius_thresh)).item()
                        angle_intensity = np.maximum(angle2intensity[int(angle)], angle_correction)
                    else:
                        angle_intensity = angle2intensity[int(angle)]

                    # determine signal strength
                    signal_strength = (args.lambda_intensity * angle_intensity + radius)/(args.lambda_intensity + 1)    # in range [0, 1]
                    label[slice_id, row_id, cul_id] = signal_strength * args.center_factor  # signal in vessel higher

                    if signal_strength == 0:    # just model tail artifacts if voxel in lower vessel wall
                        continue

                    len_tail = int(radius * R * args.len_tail) # determine length of tail artifacts primary based on radius information

                    if occ_below > 1:
                        continue
                    
                    # generate tail artifacts & add noise to tails
                    tail = np.geomspace(signal_strength * radius * args.int_tail, 0.001, num=len_tail)
                    tail = np.clip(tail + np.random.normal(args.tail_noise_mean, args.tail_noise_std, len(tail)), 0, signal_strength * args.center_factor * args.tail_clip_factor)

                    tail_end = row_id + len(tail)
                    if tail_end > R:
                        tail = tail[:-(tail_end - R)]
                        tail_end = R

                    mask[slice_id, row_id: tail_end, cul_id] = np.max([mask[slice_id, row_id: tail_end, cul_id], tail], axis=0)   # add tail to output volume

    return np.maximum(mask, label)

def preprocess_simulated_data(args):
    print(f'Preprocess volumes; experiment name: {args.name}.')
    paths_to_volumes = Path(args.path_to_syn_data)

    # init granular noise transform
    apply_granular_noise = get_transform_granular_noise(args)

    # loop over volumes
    for idx, path_to_volume in enumerate(paths_to_volumes.iterdir()):
        print('\n', f'Preprocess volume {idx}')

        volume_angle = np.load(path_to_volume / 'ang.npy')
        volume_radius = np.load(path_to_volume / 'rad.npy')
        volume_segmentation = np.load(path_to_volume / 'seg.npy')
        volume_occupancy = np.load(path_to_volume / 'occ.npy')

        # add artifacts
        volume_tails = add_tails_and_signal_loss(volume_segmentation, volume_angle, volume_radius, volume_occupancy, args).transpose(1, 2, 0)
        volume_tails = apply_granular_noise(volume_tails[None])[0]

        volume_segmentation = torch.tensor(volume_segmentation.transpose(1, 2, 0)).int().numpy()

        # save processed images
        path_to_volume_sim = path_to_volume / 'sim'
        path_to_volume_sim.mkdir(parents=True, exist_ok=True)

        np.save(path_to_volume_sim / ('sim_data_' + args.name + '.npy'), volume_tails)
        np.save(path_to_volume_sim / ('sim_seg_' + args.name + '.npy'), volume_segmentation)

        if args.write_nifti:
            write_nifti(volume_tails, path_to_volume_sim / ('sim_data_' + args.name + '.nii.gz'))
            write_nifti(volume_segmentation, path_to_volume_sim / ('sim_seg_' + args.name + '.nii.gz'))

def get_transform_granular_noise(args):
    transform = [
        RandGaussianNoise(prob=1, mean=args.gran_noise_mean, std=args.gran_noise_std),
        GaussianSmooth(sigma=args.gaus_smooth_sigma),
        ScaleIntensityRange(
                a_min=args.int_lower, a_max=args.int_upper,
                b_min=0.0, b_max=1.0, clip=True
        ),
        EnsureType(track_meta=False)
    ]
    return Compose(transform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--name", type=str, help="Name appendix for simulated data.", required=True)
    parser.add_argument("--path_to_syn_data", type=str, help="Path to synthetic_cerebral_octa.", default='./dataset/synthetic_cerebral_octa')
    parser.add_argument("--write_nifti", action='store_true')
    parser.add_argument("--dt", action='store_true', help='Disable tqdm logging.')

    # genaral signal strength parameters
    parser.add_argument("--upper_radius_thresh", type=float, default=10.0)
    parser.add_argument("--micro_radius_thresh", type=float, default=0.5)   # translates to 5um
    parser.add_argument("--angle_correction", action='store_true')
    parser.add_argument("--angle_delta", type=float, default=5.0)
    parser.add_argument("--lambda_intensity", type=float, default=9.0)
    parser.add_argument("--center_factor", type=float, default=4)

    # tail artifacts
    parser.add_argument("--len_tail", type=float, default=1.0)
    parser.add_argument("--int_tail", type=float, default=5.0)
    parser.add_argument("--tail_noise_std", type=float, default=0.5)
    parser.add_argument("--tail_noise_mean", type=float, default=0)
    parser.add_argument("--tail_clip_factor", type=float, default=0.7)

    # granular noise
    parser.add_argument("--gran_noise_std", type=float, default=1.4)
    parser.add_argument("--gran_noise_mean", type=float, default=0.0)
    parser.add_argument("--gaus_smooth_sigma", type=float, default=1.6)
    parser.add_argument("--int_lower", type=float, default=0.0)
    parser.add_argument("--int_upper", type=float, default=2.43)

    # get args 
    args = parser.parse_args()
    preprocess_simulated_data(args)