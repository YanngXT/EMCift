"""Portable entry point for the main-table EMCift implementation."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ('iDigits', 'DomainNet', 'CORe50', 'PACS', 'iWildCamTinyPlus')


def load_config(path, data_root=None, seed=None, device=None, download=False, plot=False):
    with Path(path).open(encoding='utf-8') as handle:
        config = json.load(handle)
    if config.get('dataset') not in DATASETS:
        raise ValueError('Supported datasets: ' + ', '.join(DATASETS))
    if config.get('model_name', '').lower() != 'emcift':
        raise ValueError('This release contains EMCift only.')
    if config.get('convnet_type') != 'pretrained_vit_b16_224_in21k_adapter':
        raise ValueError('The main-table release uses the ImageNet-21K ViT-B/16 adapter backbone.')
    config['model_name'] = 'emcift'
    config['mode'] = 'cdil'
    config['data_path'] = str(Path(data_root or os.environ.get('EMCIFT_DATA_ROOT') or
                                   config.get('data_path', './data')).expanduser().resolve())
    protocol = Path(config.get('protocol_path', 'protocols/' + config['dataset'] + '.yaml'))
    config['protocol_path'] = str(protocol if protocol.is_absolute() else ROOT / protocol)
    if not Path(config['protocol_path']).is_file():
        raise ValueError('Missing stream protocol: ' + config['protocol_path'])
    if seed is not None:
        config['seed'] = [seed]
    if device is not None:
        config['device'] = [str(device)]
    config['download_data'] = download
    config['plot'] = int(plot)
    for key in ('memory_per_class', 'batch_size', 'tuned_epoch', 'inc_epoch', 'ffn_num'):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(key + ' must be a positive integer.')
    if not isinstance(config.get('seed'), list) or not config['seed']:
        raise ValueError('seed must be a nonempty list.')
    if len(config.get('device', [])) != 1 or not str(config['device'][0]).isdigit():
        raise ValueError('Specify one logical CUDA device, e.g. --device 0.')
    return config


def main():
    parser = argparse.ArgumentParser(description='EMCift: Class-Domain Incremental Learning')
    parser.add_argument('--config', default=str(ROOT / 'configs/iDigits.json'))
    parser.add_argument('--data-root', help='Dataset directory; alternatively set EMCIFT_DATA_ROOT.')
    parser.add_argument('--seed', type=int, help='Override the training seed without changing data_seed.')
    parser.add_argument('--device', help='Logical CUDA device after CUDA_VISIBLE_DEVICES, e.g. 0.')
    parser.add_argument('--output-dir', default='./outputs', help='Parent directory for logs, matrices and optional checkpoints.')
    parser.add_argument('--download', action='store_true', help='Allow legacy dataset download/preprocessing; may reorganize raw files.')
    parser.add_argument('--plot', action='store_true', help='Generate the original final-phase visualization.')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print configuration without loading datasets, weights or CUDA.')
    args = parser.parse_args()
    try:
        config = load_config(args.config, args.data_root, args.seed, args.device, args.download, args.plot)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    from tools.check_data import check_layout
    if not config['download_data']:
        missing = check_layout(config['dataset'], Path(config['data_path']))
        if missing:
            parser.error('Missing prepared data: ' + ', '.join(missing) +
                         '. See docs/DATASETS.md; --download explicitly enables legacy preprocessing.')
    import torch
    if not torch.cuda.is_available():
        parser.error('Training requires CUDA. --dry-run and unit tests are CPU-only.')
    import trainer
    if config['dataset'] == 'iWildCamTinyPlus':
        from my_dataset_utils.iwildcam import build_iwildcam_tiny_dataloader
        trainer.build_continual_dataloader = build_iwildcam_tiny_dataloader
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    trainer.train(config)


if __name__ == '__main__':
    main()
