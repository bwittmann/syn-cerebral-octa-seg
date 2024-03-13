"""Transformations for data augmentation."""

import numpy as np
from monai.transforms import (
    Compose,
    EnsureTyped,
    ThresholdIntensityd,
    RandSpatialCropd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    RandRotated,
    RandZoomd,
    RandAffined,
    RandFlipd
)


def get_patch_size_before_rot(patch_size):
    return [int(np.ceil(ps * np.sqrt(2))) for ps in patch_size]


def get_transforms_domain_adaptation(data_type, config):
    rotate_range = [i / 180 * np.pi for i in config['augmentation']['rotation']]
    
    if data_type in ['sim', 'real']:
        if config['augmentation']['use_augmentation']:
            transform = [
                RandSpatialCropd(
                    keys=['image', 'label'],
                    roi_size=get_patch_size_before_rot(config['augmentation']['patch_size']),
                    random_size=False, random_center=True
                ),
                RandRotated(    # Rotation    
                    keys=['image', 'label'], prob=config['augmentation']['p_rotate'],
                    range_x=rotate_range, range_y=0, range_z=0,
                    mode=['bilinear', 'nearest'], padding_mode='zeros'
                ),
                RandZoomd(      # Zoom
                    keys=['image', 'label'], prob=config['augmentation']['p_zoom'],
                    min_zoom=config['augmentation']['min_zoom'],
                    max_zoom=config['augmentation']['max_zoom'],
                    mode=['area', 'nearest'], padding_mode='constant', constant_values=0
                ),
                RandAffined(    # Shear
                    keys=['image', 'label'], prob=config['augmentation']['p_shear'],
                    shear_range=config['augmentation']['shear_range'],
                    mode=['bilinear', 'nearest'], padding_mode='zeros'
                ),
                RandFlipd(      # Flip axis 0
                    keys=['image', 'label'], prob=config['augmentation']['p_flip'][0],
                    spatial_axis=0
                ),
                RandFlipd(      # Flip axis 1
                    keys=['image', 'label'], prob=config['augmentation']['p_flip'][1],
                    spatial_axis=1
                ),
                RandFlipd(      # Flip axis 2
                    keys=['image', 'label'], prob=config['augmentation']['p_flip'][2],
                    spatial_axis=2
                ),
                RandSpatialCropd(
                    keys=['image', 'label'],
                    roi_size=config['augmentation']['patch_size'],
                    random_size=False, #random_center=True
                ),

                # Intensity transformations
                RandGaussianSmoothd(
                    keys=['image'], prob=config['augmentation']['p_gaussian_smooth'],
                    sigma_x=config['augmentation']['gaussian_smooth_sigma'], 
                    sigma_y=config['augmentation']['gaussian_smooth_sigma'],
                    sigma_z=config['augmentation']['gaussian_smooth_sigma'],
                ),
                RandGaussianNoised(
                    keys=['image'], prob=config['augmentation']['p_gaussian_noise'], 
                    mean=config['augmentation']['gaussian_noise_mean'], std=config['augmentation']['gaussian_noise_std']
                ),
                RandScaleIntensityd(
                    keys=['image'], prob=config['augmentation']['p_intensity_scale'],
                    factors=config['augmentation']['intensity_scale_factors']
                ),
                RandShiftIntensityd(
                    keys=['image'], prob=config['augmentation']['p_intensity_shift'],
                    offsets=config['augmentation']['intensity_shift_offsets']
                ),
                RandAdjustContrastd(
                    keys=['image'], prob=config['augmentation']['p_adjust_contrast'],
                    gamma=config['augmentation']['adjust_contrast_gamma']
                ),
                RandHistogramShiftd(
                    keys=['image'], prob=config['augmentation']['p_histogram_shift'],
                    num_control_points=config['augmentation']['control_points']
                ),

                # Bring to [0, 1] range
                ThresholdIntensityd(
                    keys=['image'], threshold=0, above=True, cval=0
                ),
                ThresholdIntensityd(
                    keys=['image'], threshold=1, above=False, cval=1
                ),
                
                # Disable meta tracking for faster training
                EnsureTyped(
                    keys=['image', 'label'], track_meta=False,
                    # device=config['device']
                )
            ]
        else:
            transform = [
                RandSpatialCropd(
                    keys=['image', 'label'],
                    roi_size=config['augmentation']['patch_size'],
                    random_size=False
                ),
                EnsureTyped(
                    keys=['image', 'label'], track_meta=False,
                    # device=config['device']
                )
            ]
  
    elif data_type in ['test_val']:
        transform = [
            EnsureTyped(
                keys=['image', 'label'], track_meta=False,
                # device=config['device']
            )
        ]
    else:
        raise ValueError("Please use 'test_val', 'sim', or 'real' as arg.")
    return Compose(transform)