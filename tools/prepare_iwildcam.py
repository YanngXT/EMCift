"""Materialize only the fixed reported iWildCam images, without selecting a new stream."""
import argparse
import json
from pathlib import Path
import shutil

from PIL import Image

try:
    from tools.iwildcam_download import download_index, extract_member
except ModuleNotFoundError:
    from iwildcam_download import download_index, extract_member

ROOT = Path(__file__).resolve().parents[1]


def materialize(data_root, source_dir=None, download=False):
    manifest = json.loads((ROOT / 'protocols/iwildcam/manifest.json').read_text())
    base = Path(data_root).expanduser().resolve() / 'iWildCamTinyPlus'
    images = base / manifest['data_root']
    rows = manifest['train_examples'] + manifest['test_examples']
    entries = None
    for row in rows:
        target = images / row['rel_path']
        if target.is_file():
            try:
                with Image.open(target) as image:
                    image.verify()
                continue
            except (OSError, SyntaxError):
                raise ValueError('Invalid existing image; inspect before replacing: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_dir is not None:
            origin = Path(source_dir).expanduser().resolve()
            candidates = [origin / row['zip_name'], origin / row['file_name'],
                          origin / 'train' / Path(row['file_name']).name,
                          origin / Path(row['file_name']).name]
            source = next((path for path in candidates if path.is_file()), None)
            if source is None:
                raise FileNotFoundError('Selected official image not found: ' + row['image_id'])
            shutil.copy2(source, target)
        elif download:
            if entries is None:
                entries = download_index(manifest['source_url'], base / 'cache/central_directory.bin')
            raw = extract_member(manifest['source_url'], entries, row['zip_name'])
            temporary = target.with_suffix('.jpg.part')
            temporary.write_bytes(raw)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(target)
        else:
            raise ValueError('Missing image: use --source-dir or explicitly opt into --download.')
        with Image.open(target) as image:
            image.verify()
    return dict(prepared=True, images=len(rows), train=len(manifest['train_examples']),
                test=len(manifest['test_examples']), image_root=str(images))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--source-dir', type=Path)
    source.add_argument('--download', action='store_true')
    args = parser.parse_args()
    print(json.dumps(materialize(args.data_root, args.source_dir, args.download), indent=2))


if __name__ == '__main__':
    main()
