"""Module containing the dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from syn_cerebral_octa_seg.utils import read_nifti
from syn_cerebral_octa_seg.transforms import get_transforms_domain_adaptation


class RealDatasetAnnotated(Dataset):
    def __init__(self, config, split):
        assert split in ['train']
        self._config = config
        self._split = split

        train_img = config['train_img']
        test_set_folder = Path("./dataset/").resolve() / Path(self._config['annotations'])
        self._data = [torch.tensor(read_nifti(test_set_folder / f'test_set_{img}.nii')) for img in train_img]
        self._doppler = [torch.tensor(read_nifti(test_set_folder / f'test_set_{img}_inclDopp.nii')) for img in train_img]
        self._label = [torch.tensor(read_nifti(test_set_folder / f'test_set_{img}_label.nii').astype(int)).int() for img in train_img]
        self._modulo = len(self._data)

        # inits augmentations
        self._augmentation = get_transforms_domain_adaptation('real', config)

    def __len__(self):
        return self._config['itera_pro_epoch']

    def __getitem__(self, idx):
        data_dict = {
            'image': self._data[idx % self._modulo].clone().detach()[None],
            'doppler': self._doppler[idx % self._modulo].clone().detach().squeeze() if self._config['use_doppler'] else torch.zeros((1, 160, 160, 160)),
            'label': self._label[idx % self._modulo].clone().detach()[None]
        }

        # ensure different augmentation each itera
        self._augmentation.set_random_state(torch.initial_seed() + idx)
        data_transformed = self._augmentation(data_dict)

        if self._config['use_doppler']:
            img = torch.cat([data_transformed['image'].float(), data_transformed['doppler'].float()])
        else:
            img = data_transformed['image'].float()

        return img, data_transformed['label'].int()


class TestDataset(Dataset):
    def __init__(self, config, split):
        assert split in ['test', 'val']

        self._config = config
        self._split = split

        test_set_folder = Path("./dataset/").resolve() / Path(self._config['annotations'])
        
        # load annotated data + define regions occupied by large and small vessels
        self.m4_0 = {
            'img': torch.tensor(read_nifti(test_set_folder / 'test_set_m4_0.nii')),
            'doppler': torch.tensor(read_nifti(test_set_folder / 'test_set_m4_0_inclDopp.nii')),
            'label': torch.tensor(read_nifti(test_set_folder / 'test_set_m4_0_label.nii').astype(int)).int(),
            'large_region': [[0, 160, 70, 160, 110, 160], [0, 160, 0, 40, 0, 105]],    # new split w/o kissing vessels (breaks cldice)
            'small_region': [[0, 160, 64, 160, 0, 100]],
            'comp_region': [0, 160, 70, 160, 0, 160]    # exclude kissing vessels
        }

        self.m4_1 = {
            'img': torch.tensor(read_nifti(test_set_folder / 'test_set_m4_1.nii')),
            'doppler': torch.tensor(read_nifti(test_set_folder / 'test_set_m4_1_inclDopp.nii')),
            'label': torch.tensor(read_nifti(test_set_folder / 'test_set_m4_1_label.nii').astype(int)).int(),
            'large_region': [[0, 160, 0, 100, 110, 160]],
            'small_region': [[0, 160, 40, 140, 0, 110]],
            'comp_region': [0, 160, 0, 160, 0, 160]
        }

        self.m78_0 = {
            'img': torch.tensor(read_nifti(test_set_folder / 'test_set_m78_0.nii')),
            'doppler': torch.tensor(read_nifti(test_set_folder / 'test_set_m78_0_inclDopp.nii')),
            'label': torch.tensor(read_nifti(test_set_folder / 'test_set_m78_0_label.nii').astype(int)).int(),
            'large_region': [[0, 160, 0, 120, 115, 160]],
            'small_region': [[0, 160, 0, 150, 0, 110]],
            'comp_region': [0, 160, 0, 160, 0, 160]
        }

        self.m78_1 = {
            'img': torch.tensor(read_nifti(test_set_folder / 'test_set_m78_1.nii')),
            'doppler': torch.tensor(read_nifti(test_set_folder / 'test_set_m78_1_inclDopp.nii')),
            'label': torch.tensor(read_nifti(test_set_folder / 'test_set_m78_1_label.nii').astype(int)).int(),
            'large_region': [[0, 160, 0, 160, 0, 54]],
            'small_region': [[0, 160, 35, 160, 54, 160]],
            'comp_region': [0, 160, 0, 160, 0, 160]
        }

        self.m44_0 = {
            'img': torch.tensor(read_nifti(test_set_folder / 'test_set_m44_0.nii')),
            'doppler': torch.tensor(read_nifti(test_set_folder / 'test_set_m44_0_inclDopp.nii')),
            'label': torch.tensor(read_nifti(test_set_folder / 'test_set_m44_0_label.nii').astype(int)).int(),
            'large_region': [[0, 160, 30, 160, 0, 50]],
            'small_region': [[0, 160, 0, 100, 50, 160]],
            'comp_region': [0, 160, 0, 160, 0, 160]
        }

        self.m44_1 = {
            'img': torch.tensor(read_nifti(test_set_folder / 'test_set_m44_1.nii')),
            'doppler': torch.tensor(read_nifti(test_set_folder / 'test_set_m44_1_inclDopp.nii')),
            'label': torch.tensor(read_nifti(test_set_folder / 'test_set_m44_1_label.nii').astype(int)).int(),
            'large_region': [[0, 160, 0, 130, 120, 160], [0, 160, 105, 150, 0, 125]],
            'small_region': [[0, 160, 0, 105, 0, 95]],
            'comp_region': [0, 160, 0, 160, 0, 160]
        }

        # collect relevant data
        img_ids = config['val_imgs'] if split == 'val' else config['test_imgs']
        self._data = [getattr(self, img_id) for img_id in img_ids]

        # inits augmentations
        self._augmentation = get_transforms_domain_adaptation('test_val', config)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        data = self._data[idx]
        small_regions = data['small_region']
        large_regions = data['large_region']
        comp_region = data['comp_region']

        data_dict = {
            'image': data['img'].clone().detach()[None],
            'doppler': data['doppler'].clone().detach().squeeze(),
            'label': data['label'].clone().detach()[None]
        }

        # ensure different augmentation each itera
        self._augmentation.set_random_state(torch.initial_seed() + idx)
        data_transformed = self._augmentation(data_dict)

        if self._config['use_doppler']:
            img = torch.cat([data_transformed['image'].float(), data_transformed['doppler'].float()])
        else:
            img = data_transformed['image'].float()

        return img, data_transformed['label'].int(), small_regions, large_regions, comp_region


class SimDataset(Dataset):
    def __init__(self, config, split):
        assert split in ['train']   # train only 
        self._config = config
        self._split = split
        self._dataset_version = config['sim_data_version']

        dataset_path = Path("./dataset/").resolve() / Path(self._config['sim_data'])
        data_dirs = sorted([data_path for data_path in dataset_path.resolve().iterdir()])
        self._data = data_dirs

        # inits augmentations
        self._augmentation = get_transforms_domain_adaptation('sim', config)

    def __len__(self):
        return self._config['itera_pro_epoch']

    def __getitem__(self, idx):
        img_idx = np.random.randint(low=0, high=len(self._data))    # randomly select image to extract crop from

        # load data
        case = self._data[img_idx]
        data_path, label_path = case / 'sim' / f'sim_data_{self._dataset_version}.npy', case / 'sim' / f'sim_seg_{self._dataset_version}.npy'
        data, label = np.load(data_path), np.load(label_path)

        data_dict = {
            'image': data[None],
            'label': label[None]
        }

        self._augmentation.set_random_state(torch.initial_seed() + idx + img_idx ) # ensure different augmentation each itera
        data_transformed = self._augmentation(data_dict)

        return data_transformed['image'].float(), data_transformed['label'].int()


def get_loaders(config):
    loaders = {}
    
    # add loaders
    dataset = TestDataset(config, 'test')
    loaders['test'] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])
                                 
    dataset = TestDataset(config, 'val')
    loaders['val'] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])

    dataset = RealDatasetAnnotated(config, 'train') if config['train_on_real'] else SimDataset(config, 'train')
    loaders['train'] = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)

    return loaders