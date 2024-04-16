## SuperUNet

This branch contains code for the best-performing model & an inference script tailored to our in-house data. Instructions from the `main` branch hold.

### Inference

    python syn_cerebral_octa_seg/inference.py --run superunet_fold0 --data_folder <folder_to_raw_tif_files>

The `<folder_to_raw_tif_files>` should contain raw .tif files. Preprocessing is done in the inference script.
For a selection of models, please see below.

Optional arguments: 

1. `--overlap`: Overlap of tiles in sliding window inference scheme
2. `--last`: Use checkpoint `last` instead of `best`
3. `--cpu`: Run on `cpu` instead of `gpu`
4. `--ind_per`: Use an individual percentile-based threshold per image rather than a global one
5. `--ensemble`: Aktivate 6-fold ensemble inference strategy
5. `--doppler`: TODO

### Provided models:

- `superunet_foldx`: ...
- `superunet_pre_foldx`: ...
- `superunet_pre_doppler_foldx`: ...
- `superunet_pre_96_foldx`: ...