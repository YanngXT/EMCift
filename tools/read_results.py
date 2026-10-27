"""Read native EMCift metrics; never substitute final-fragment accuracy for A_T."""
import argparse
import ast
import json
import hashlib
from pathlib import Path
import re
import statistics

EXPECTED = {'iDigits': 19, 'DomainNet': 167, 'CORe50': 56, 'iWildCamTinyPlus': 7}


def read_log(path):
    curve, af, seed, dataset, data_seed, prefix = None, None, None, None, None, None
    training_config = {}
    training_keys = {'convnet_type', 'optimizer', 'weight_decay', 'init_lr', 'batch_size', 'tuned_epoch',
                     'inc_epoch', 'ffn_num', 'lambda_dann', 'lambda_metric', 'use_mlp', 'rescale_ce_loss',
                     'memory_per_class'}
    with Path(path).open(errors='replace') as handle:
        for line in handle:
            if 'CNN top1 curve:' in line:
                curve = ast.literal_eval(line.split('CNN top1 curve:', 1)[1].strip())
            match = re.search(r'\bForgetting:\s*([-+\d.eE]+)', line)
            if match:
                af = float(match.group(1))
            match = re.search(r'\] => (seed|dataset|data_seed|model_prefix):\s*(.*)', line)
            if match:
                name, value = match.groups()
                if name == 'seed': seed = int(value)
                elif name == 'dataset': dataset = value.strip()
                elif name == 'data_seed': data_seed = int(value)
                elif name == 'model_prefix': prefix = value.strip()
            match = re.search(r'\] => (\w+):\s*(.*)', line)
            if match and match.group(1) in training_keys:
                try:
                    training_config[match.group(1)] = ast.literal_eval(match.group(2))
                except (ValueError, SyntaxError):
                    training_config[match.group(1)] = match.group(2).strip()
    protocol = Path(__file__).resolve().parents[1] / 'protocols/PACS.yaml'
    if dataset == 'PACS':
        import random
        from types import SimpleNamespace
        from my_dataset_utils.my_datasets import build_dicil_scenario
        state = random.getstate()
        try:
            random.seed(data_seed)
            fake = SimpleNamespace(classes=list(range(7)), targets=list(range(7)))
            fragments = {str(domain): (fake, fake, fake) for domain in range(4)}
            expected = len(build_dicil_scenario(fragments, dict(dataset='PACS', protocol_path=str(protocol)))[0])
        finally:
            random.setstate(state)
    else:
        expected = EXPECTED.get(dataset)
    complete = expected is not None and curve is not None and len(curve) == expected and af is not None
    return dict(source=str(path), dataset=dataset, seed=seed, data_seed=data_seed, model_prefix=prefix,
                training_config=training_config,
                config_fingerprint=hashlib.sha256(json.dumps(training_config, sort_keys=True).encode()).hexdigest(),
                stages=len(curve) if curve else 0, complete=complete,
                barA=statistics.mean(curve) if complete else None,
                A_T=curve[-1] if complete else None, A_F=af if complete else None)


def summarize(records):
    if not records or not all(row['complete'] for row in records):
        raise ValueError('Only complete runs can be aggregated.')
    if len({(row['dataset'], row['data_seed'], row['model_prefix']) for row in records}) != 1:
        raise ValueError('Do not aggregate different datasets, streams or configuration prefixes.')
    if len({row.get('config_fingerprint') for row in records}) != 1:
        raise ValueError('Do not aggregate different optimization/model configurations.')
    if len({row['seed'] for row in records}) != len(records):
        raise ValueError('Duplicate training seeds; choose one completed run per seed.')
    return {key: dict(mean=statistics.mean(row[key] for row in records),
                      sample_std=statistics.stdev(row[key] for row in records) if len(records) > 1 else None)
            for key in ('barA', 'A_T', 'A_F')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path, help='One native log file per completed seed.')
    args = parser.parse_args()
    records = [read_log(path) for path in args.logs]
    result = dict(runs=records)
    if all(row['complete'] for row in records):
        try:
            result['summary'] = summarize(records)
        except ValueError as error:
            parser.error(str(error))
    else:
        result['summary'] = None
        result['note'] = 'Incomplete logs are not included in a final-result summary.'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
