"""Check prepared layouts without downloading, reading images or modifying data."""
import argparse
import json
from pathlib import Path


def check_layout(dataset, root):
    if dataset == 'iDigits':
        required = [root / name for name in ('MNIST_RGB', 'MNIST-M/processed', 'SVHN', 'SynDigit/processed')]
    elif dataset == 'DomainNet':
        domains = ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch')
        required = [root / 'DomainNet' / split / domain for split in ('train', 'test') for domain in domains]
    elif dataset == 'CORe50':
        required = [root / 'core50_128x128/train' / session for session in ('s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11')]
        required.append(root / 'core50_128x128/test')
    elif dataset == 'PACS':
        required = [root / 'PACS/kfold' / domain for domain in ('art_painting', 'cartoon', 'photo', 'sketch')]
    elif dataset == 'iWildCamTinyPlus':
        manifest = json.loads((Path(__file__).resolve().parents[1] / 'protocols/iwildcam/manifest.json').read_text())
        images = root / 'iWildCamTinyPlus' / manifest['data_root']
        return [str(images / row['rel_path']) for row in manifest['train_examples'] + manifest['test_examples']
                if not (images / row['rel_path']).is_file()]
    else:
        raise ValueError('Unsupported dataset: ' + dataset)
    return [str(path) for path in required if not path.is_dir()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('iDigits', 'DomainNet', 'CORe50', 'PACS', 'iWildCamTinyPlus'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    missing = check_layout(args.dataset, args.data_root.expanduser().resolve())
    print(json.dumps(dict(dataset=args.dataset, prepared=not missing, missing=missing), indent=2))
    raise SystemExit(bool(missing))


if __name__ == '__main__':
    main()
