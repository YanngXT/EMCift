# Supplemental Datasets

Only EMCift is included. Supplemental configurations are in `experiments/supplemental/`; the three main-table configurations remain in `configs/`.

## iWildCamTinyPlus

`protocols/iwildcam/manifest.json` preserves the reported image IDs, species labels, camera IDs, timestamps, exact train/test membership, phase order and indices. Its only portability changes are the dataset display name and the relative image root. The stream has seven phases, six camera-location domains and 3--6 species per phase. Camera 105 recurs over time, species overlap between cameras, and phase class counts vary. There are 274 training and 134 held-out test images. All images originate from the official labeled training portion of iWildCam 2022, not its unlabeled competition test set.

This is a small metadata-derived case study, not the full benchmark or an untouched acquisition log. The split is image-disjoint within camera/month/species fragments; it is not guaranteed sequence-disjoint. Do not describe it as testing generalization to unseen camera sites or future-only time blocks.

Prepare from a local official extraction:

```bash
python tools/prepare_iwildcam.py --data-root /path/to/datasets --source-dir /path/to/iwildcam-2022
```

Or download only the fixed selected members from the public archive:

```bash
python tools/prepare_iwildcam.py --data-root /path/to/datasets --download
```

Images are placed under `DATA_ROOT/iWildCamTinyPlus/images/{train,test}/locCAMERA/SPECIES/IMAGE_ID.jpg`. The command verifies readable images and never changes manifest membership. Existing files are reused. The byte-range utility checks server support before downloading members. A cached ZIP central directory is retained under `DATA_ROOT/iWildCamTinyPlus/cache/`; the full archive is not fetched. Image/data rights remain governed by the original release; no images are redistributed here.

EMCift uses learning rate0.001, batch8, 10 initial and10 incremental epochs, metric weight0.2, DANN weight0.1 and one exemplar per observed pair. The reported three-run reference is average accuracy66.46 +/-0.09, final cumulative accuracy53.73 +/-2.24 and forgetting13.81 +/-0.38. This package has not been freshly retrained to reproduce it.

## PACS

Place the original data as `DATA_ROOT/PACS/kfold/{art_painting,cartoon,photo,sketch}/CLASS/IMAGE`. The original loader uses a dedicated torch generator seeded by `data_seed` to make an80/20 permutation split within each entire domain; it is not an official or stratified split. Train augmentation and prototype extraction reuse identical training indices. The exact fragmentation YAML is in `protocols/PACS.yaml`.

```bash
python tools/check_data.py --dataset PACS --data-root /path/to/datasets
bash scripts/run.sh PACS /path/to/datasets 0 42 1004 1108
```

The original PACS configuration uses learning rate0.0028125, batch64, 2 initial and3 incremental epochs, metric weight0.5 (default), DANN weight0.1, `use_mlp=1` and one exemplar per observed pair. PACS was already evaluated in the original manuscript's supplementary experiments; its inclusion here does not imply a newly run experiment.
