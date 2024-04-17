"""Utils for syn_cerebral_octa_seg."""

import sys
import json
import socket
import subprocess
from pathlib import Path

import yaml
import h5py
import torch
import numpy as np
from scipy.io import loadmat
from skimage import io
import SimpleITK as sitk

PATH_TO_CONFIG = Path('syn_cerebral_octa_seg/')

def write_nifti(data, file_path):
    meta_data = {}
    meta_data['itk_spacing'] = [1, 1, 1]

    data_itk = sitk.GetImageFromArray(data)
    # data_itk.SetOrigin(meta_data['itk_origin'])
    data_itk.SetSpacing(meta_data['itk_spacing'])
    # data_itk.SetDirection(meta_data['itk_direction'])

    sitk.WriteImage(data_itk, str(file_path))

def read_nifti(path):
    sitk_img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(sitk_img)

def read_mat(file_path):
    try:
        data = loadmat(file_path)
    except:
        f = h5py.File(file_path,'r')
        
        data = {}
        for k in f.keys():

            data_ret = f.get(k)
            data[k] = np.array(data_ret)
    return data

def read_tiff(file_path):
    img = io.imread(file_path)
    return np.array(img)

def write_tiff(file_path, data):
    io.imsave(file_path, data.astype(np.float32))

def normalize_np(np_array):
    np_array = np_array.astype(np.float)
    np_array *= (1.0/np_array.max())
    return np_array

def get_config(config_name):
    with open(PATH_TO_CONFIG / f'{config_name}.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def get_meta_data():
    meta_data = {}
    meta_data['git_commit_hash'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    meta_data['python_version'] = sys.version.splitlines()[0]
    meta_data['gcc_version'] = sys.version.splitlines()[1]
    meta_data['pytorch_version'] = torch.__version__
    meta_data['host_name'] = socket.gethostname()
    return meta_data

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=3)

def read_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data