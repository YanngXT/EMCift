# Dataset Layouts

Dataset images, archives and pretrained weights are not included. Obtain the original datasets under their respective terms. `DATA_ROOT` is a directory containing the following prepared trees; the three protocol YAMLs are loaded from this repository, not from a server-specific directory.

```text
DATA_ROOT/
  MNIST_RGB/raw/                  # standard MNIST IDX files
  MNIST-M/processed/
    mnist_m_train.pt
    mnist_m_test.pt
  SVHN/
    train_32x32.mat
    test_32x32.mat
  SynDigit/processed/
    synth_train.pt
    synth_test.pt
  DomainNet/
    train/{clipart,infograph,painting,quickdraw,real,sketch}/CLASS/IMAGE
    test/{clipart,infograph,painting,quickdraw,real,sketch}/CLASS/IMAGE
  core50_128x128/
    train/{s1,s2,s4,s5,s6,s8,s9,s11}/OBJECT/IMAGE.png
    test/OBJECT/IMAGE.png
```

MNIST uses the four standard IDX image/label files. Their gzip archives can also be downloaded by the torchvision-compatible loader. Dataset class-label folder names and their ordering must be preserved.

For DomainNet, preserve all345 semantic classes in each domain and use the original training/test lists. The legacy preparation path downloads the archive/list URLs embedded in `my_dataset_utils/continual_datasets.py` and moves files into the above structure. Already prepared train/test trees do not require keeping a second copy of the six archives.

For CORe50, training domains are sessions1,2,4,5,6,8,9,11; test sessions3,7,10 are pooled by object into the test directory. There are50 object classes. The legacy preparation path rearranges files from the original `core50_128x128` archive. Work on a separate copy if enabling it.

```bash
python tools/check_data.py --dataset DomainNet --data-root /path/to/datasets
```

This command only checks directory presence. It is not a checksum or full-image-integrity audit. During actual loading, the dataset classes check their required files. If using the legacy downloader, explicitly add `--download` to the training command; network availability is not guaranteed. Original split assignments and the CDIL fragmentation algorithm have not been redesigned for this release.

| Dataset | Domains | Semantic classes | Phases with data_seed0 | Classes per phase before final leftovers |
|---|---:|---:|---:|---:|
| iDigits | 4 | 10 | 19 | 1--3 |
| DomainNet | 6 | 345 | 167 | 10--15 |
| CORe50 | 8 training sessions | 50 | 56 | 5--10 |

Training domain IDs are provided by the acquisition domain. Inference does not receive them. The semantic label remapping and phase order remain those of the original implementation.
