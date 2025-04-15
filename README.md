## OCTA-UNet++

This branch contains code for the best-performing model & an inference script tailored to our in-house data. Instructions from the `main` branch hold.
OCTA-UNet++ was utilized in the manuscript "In Vivo Network-Level Cerebrovascular Mapping Reveals the Effects of Flow Topology on Capillary Stalls After Stroke."
<!-- Further, this branch contains code to train with doppler images.  -->

### Setup

Make sure to have Git LFS installed to download the checkpoints:

    git lfs install

Download checkpoints:

    git clone https://huggingface.co/bwittmann/octa-unetplusplus

and put them into the `./runs` folder. The structure should follow:

    └── runs/
        └── model_fold0
        └── model_fold0_crop96
        ...

### Inference

    python syn_cerebral_octa_seg/inference.py --run model_fold0_pre --data_folder <folder_to_raw_tif_files> --ensemble

The `<folder_to_raw_tif_files>` should contain raw .tif files. Preprocessing is done in the inference script.

Optional arguments: 

1. `--overlap`: Overlap of tiles in sliding window inference scheme
2. `--last`: Use checkpoint `last` instead of `best`
3. `--cpu`: Run on `cpu` instead of `gpu`
4. `--ind_per`: Use an individual percentile-based threshold per image rather than a global one
5. `--ensemble`: Aktivate 6-fold ensemble inference strategy; just for `model_foldx_pre`
<!-- 5. `--doppler`: TODO -->

### Provided Models
For a selection of models, please see below.

- `model_foldx`: Model trained on fold x
- `model_foldx_crop96`: Model trained on fold x with an increased FoV (64 -> 96)
- `model_foldx_intaug`: Model trained on fold x with more intensity scale/shift augmentations
- `model_foldx_pre`:  Model trained on fold x, pre-trained on synthetic data
<!-- - `model_foldx_doppler`: Model trained on fold x including Doppler data as additional input channels -->