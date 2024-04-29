## SuperUNet

This branch contains code for the best-performing model & an inference script tailored to our in-house data. Further, this branch contains code to train with doppler images. Instructions from the `main` branch hold.

### Setup

Make sure to have Git LFS installed to download the checkpoints:

    git lfs install

Download checkpoints:

    git clone https://huggingface.co/bwittmann/OCTA-superunet

and put them into the `./runs` folder. The structure should follow:

    └── runs/
        └── superunet_fold0
        └── superunet_fold0_crop96
        ...

### Inference

    python syn_cerebral_octa_seg/inference.py --run superunet_fold0_pre --data_folder <folder_to_raw_tif_files> --ensemble

The `<folder_to_raw_tif_files>` should contain raw .tif files. Preprocessing is done in the inference script.

Optional arguments: 

1. `--overlap`: Overlap of tiles in sliding window inference scheme
2. `--last`: Use checkpoint `last` instead of `best`
3. `--cpu`: Run on `cpu` instead of `gpu`
4. `--ind_per`: Use an individual percentile-based threshold per image rather than a global one
5. `--ensemble`: Aktivate 6-fold ensemble inference strategy; just for `superunet_foldx_pre`
5. `--doppler`: TODO

### Provided models:
For a selection of models, please see below.

- `superunet_foldx`: SuperUNet trained on fold x
- `superunet_foldx_crop96`: SuperUNet trained on fold x, pre-trained on synthetic data, with an increased FoV (64 -> 96)
- `superunet_foldx_intaug`: SuperUNet trained on fold x, pre-trained on synthetic data, more intensity scale/shift augmentations
- `superunet_foldx_pre`:  SuperUNet trained on fold x, pre-trained on synthetic data
- `superunet_foldx_doppler`: TODO <!-- SuperUNet trained on fold x including Doppler data as additional input channels -->