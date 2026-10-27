# EMCift

Implementation of **Class-Domain Incremental Learning with Extensible Multi-Center Modeling** (NeurIPS 2026).

This release contains the full EMCift implementation, the three main-table configurations (iDigits, DomainNet and CORe50), and the supplemental iWildCamTinyPlus and PACS dataset integrations. No competing-method implementations, ablation configurations, alternative-backbone experiments, images, trained weights, private experiment logs or reviewer discussions are included.

## Installation

The recorded development environment uses Python 3.9, PyTorch 2.2.0, torchvision 0.17.0 and timm 1.0.17. Training requires a CUDA-capable GPU. CPU-only configuration checks and unit tests are available.

```bash
conda create -n emcift python=3.9 pip
conda activate emcift
pip install -r requirements.txt
```

Alternatively, run `conda env create -f environment.yml`. Install a PyTorch build appropriate for your GPU/driver if the default pip build is unsuitable. Pretrained ViT weights are obtained through timm's normal cache; no model files are bundled and no download mirror is forced.

## Data

Prepare the original benchmark data outside this repository. See [dataset layouts](docs/DATASETS.md). The exact stream YAMLs are included under `protocols/`; `data_seed=0` produces 19, 167 and 56 phases for iDigits, DomainNet and CORe50 respectively.

```bash
python tools/check_data.py --dataset iDigits --data-root /path/to/datasets
python main.py --config configs/iDigits.json --data-root /path/to/datasets --dry-run
```

Training uses existing prepared data by default. Downloads and the original file-reorganizing preprocessing require the explicit `--download` flag. Never enable preprocessing on the only copy of a raw dataset.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/iDigits.json --data-root /path/to/datasets --device 0 --seed 42
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/DomainNet.json --data-root /path/to/datasets --device 0 --seed 42
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/CORe50.json --data-root /path/to/datasets --device 0 --seed 42
```

For repeated runs with separate processes and output folders:

```bash
bash scripts/run.sh iDigits /path/to/datasets 0 42 1004 1108
```

`--data-root` can be replaced by the `EMCIFT_DATA_ROOT` environment variable. GPU IDs passed to `--device` are logical IDs after `CUDA_VISIBLE_DEVICES` is applied. Logs and accuracy matrices are written below `outputs/` by default. Use `--output-dir` to change this location. Optional final-phase visualization is enabled with `--plot`; final model serialization is controlled by `save_checkpoint` in the JSON.

## Configurations

`configs/` contains exactly one full-model configuration for each main-table dataset. The settings are retained from the original repeated-run configurations and cross-checked against available native training logs. There are no historical or tuning alternatives in this release. Source checksums and packaging-only changes are recorded in `CONFIG_ORIGINS.json`.

| Dataset | LR | Batch | Initial/incremental epochs | Metric loss weight |
|---|---:|---:|---:|---:|
| iDigits | 0.001 | 32 | 2/2 | 0.2 |
| DomainNet | 0.001 | 32 | 2/2 | 0.2 |
| CORe50 | 0.0028125 | 64 | 2/3 | 0.5 (implementation default) |

All retain `use_mlp=1`, `lambda_dann=0.1`, adapter bottleneck16 and one exemplar per observed class-domain pair, with the ImageNet-21K supervised ViT-B/16 and its frozen reference branch. The configuration string `optimizer="adam"` selects `torch.optim.AdamW` in the original implementation. The manuscript's shorthand "Adam" should be clarified to AdamW; no optimizer has been changed for packaging. Optimization settings are dataset-specific, not identical across all datasets.

## Supplemental Datasets

The two supplemental configurations are separate from the main-table directory:

```bash
python tools/prepare_iwildcam.py --data-root /path/to/datasets --source-dir /path/to/iwildcam-2022
bash scripts/run.sh iWildCamTinyPlus /path/to/datasets 0 42 1004 1108
bash scripts/run.sh PACS /path/to/datasets 0 42 1004 1108
```

Without an existing iWildCam download, use `--download` instead of `--source-dir` to fetch only the 408 selected images via ZIP byte ranges, not the full image archive. Network transfer requires explicit opt-in and the upstream server's range support. The bundled manifest fixes all seven phases, six recurring camera domains and the exact 274/134 train/test examples; preparation never resamples the stream. PACS retains the original seeded 80/20 split within each domain and its original CDIL YAML. See [supplemental details](docs/SUPPLEMENTAL.md) for layouts, limitations and settings.

## Reported Main Results

These values are transcribed from the paper, not measurements from a new training run of this packaged release. Packaging is verified by CPU tests and original-source comparisons, not by a fresh three-dataset retraining. Available historical logs do not establish a complete one-to-one mapping of every paper seed to a retained log; these reference values are not a new reproduction claim. Exact repeated-run reproduction depends on data files, stream and execution environment.

| Dataset | Average incremental accuracy | Final cumulative accuracy | Forgetting |
|---|---:|---:|---:|
| iDigits | 87.65 +/- 0.76 | 85.85 +/- 0.56 | 10.45 +/- 0.52 |
| DomainNet | 66.54 +/- 0.21 | 58.70 +/- 0.29 | 17.54 +/- 0.15 |
| CORe50 | 90.69 +/- 0.27 | 92.19 +/- 0.37 | 2.27 +/- 0.28 |

## Evaluation

The trainer evaluates the prototype tree without test-time domain IDs. Average incremental accuracy is the mean of the `CNN top1 curve`; final accuracy is its **last value over all seen test samples**, not the accuracy of the last arriving fragment. Forgetting is the final-stage `Forgetting` value from the per-task accuracy matrix. See [implementation notes](docs/IMPLEMENTATION.md).

```bash
python tools/read_results.py /path/to/seed42.log /path/to/seed1004.log /path/to/seed1108.log
python -m unittest discover -s tests -v
```

The result reader excludes incomplete runs and reports sample standard deviation. Only explicitly supplied runs are aggregated; no best-seed selection is performed. Configuration checks and tests do not download datasets or pretrained weights.

## Attribution and Licensing

Necessary shared utilities and pretrained-backbone components retain their upstream references; this is not a distribution of the baseline methods. See [third-party notices](THIRD_PARTY.md) and the accompanying license texts. No new blanket license has been assigned to the authors' project-specific code in this release candidate; the authors should confirm that choice before public distribution. Third-party terms are not replaced by that choice.

Please cite *Class-Domain Incremental Learning with Extensible Multi-Center Modeling*, NeurIPS 2026, when using this work.
