import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import load_config
from tools.read_results import read_log, summarize


class EntryTests(unittest.TestCase):
    def test_all_main_configs(self):
        for name in ('iDigits', 'DomainNet', 'CORe50'):
            config = load_config(ROOT / 'configs' / (name + '.json'))
            self.assertEqual(config['memory_per_class'], 1)
            self.assertEqual(config['use_mlp'], 1)
            self.assertEqual(config['model_name'], 'emcift')
            self.assertEqual(config['data_seed'], 0)
            self.assertTrue(Path(config['protocol_path']).is_file())
        self.assertEqual(len(list((ROOT / 'configs').rglob('*.json'))), 3)
        for name, batch, lr, inc, metric in [('iDigits', 32, .001, 2, .2), ('DomainNet', 32, .001, 2, .2),
                                              ('CORe50', 64, .0028125, 3, .5)]:
            config = load_config(ROOT / 'configs' / (name + '.json'))
            self.assertEqual((config['batch_size'], config['init_lr'], config['inc_epoch'], config.get('lambda_metric', .5)),
                             (batch, lr, inc, metric))

    def test_supplemental_configs(self):
        for name in ('PACS', 'iWildCamTinyPlus'):
            config = load_config(ROOT / 'experiments/supplemental' / (name + '.json'))
            self.assertEqual(config['dataset'], name)
            self.assertEqual(config['memory_per_class'], 1)
            self.assertTrue(Path(config['protocol_path']).is_file())

    def test_dry_run_is_portable(self):
        result = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--config',
                                 str(ROOT / 'configs/iDigits.json'), '--seed', '1004', '--dry-run'],
                                cwd=tempfile.gettempdir(), capture_output=True, text=True, check=True)
        config = json.loads(result.stdout)
        self.assertEqual(config['seed'], [1004])
        self.assertFalse(config['download_data'])


class MetricsTests(unittest.TestCase):
    def test_global_last_not_final_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.log'
            path.write_text('[trainer.py] => dataset: iDigits\n[trainer.py] => seed: 42\n'
                            '[trainer.py] => data_seed: 0\n[trainer.py] => model_prefix: main\n'
                            '[base.py] => Current Task Acc: 99.0 Forgetting: 7.5\n'
                            '[trainer.py] => CNN top1 curve: ' + str(list(range(1, 20))) + '\n')
            result = read_log(path)
            self.assertTrue(result['complete'])
            self.assertEqual(result['barA'], 10)
            self.assertEqual(result['A_T'], 19)
            self.assertEqual(result['A_F'], 7.5)
            self.assertIsNone(summarize([result])['A_T']['sample_std'])

    def test_incomplete_and_duplicate_seeds_rejected(self):
        row = dict(complete=False, dataset='iDigits', seed=42, data_seed=0, model_prefix='main')
        with self.assertRaises(ValueError):
            summarize([row])
        row.update(complete=True, barA=80.0, A_T=75.0, A_F=9.0)
        with self.assertRaises(ValueError):
            summarize([row, row])

    def test_mixed_configurations_rejected(self):
        row = dict(complete=True, dataset='iDigits', seed=42, data_seed=0, model_prefix='main',
                   barA=80.0, A_T=75.0, A_F=9.0, config_fingerprint='first')
        other = dict(row, seed=1004, config_fingerprint='second')
        with self.assertRaises(ValueError):
            summarize([row, other])


class ModelTests(unittest.TestCase):
    def test_other_methods_are_not_shipped(self):
        from my_utils.factory import get_model
        with self.assertRaises(NotImplementedError):
            get_model('simplecil', {})

    def test_domain_free_multi_center_scoring(self):
        import torch
        from torch import nn
        from convs.linears import CosineLinear
        from models.emcift import MultiPrototypeManager
        from my_utils.inc_net import MultiBranchCosineIncrementalNet_v5
        model = MultiBranchCosineIncrementalNet_v5.__new__(MultiBranchCosineIncrementalNet_v5)
        nn.Module.__init__(model)
        model.fc = CosineLinear(4, 2)
        manager = MultiPrototypeManager()
        manager.add_prototype(0, torch.tensor([1., 0., 0., 0.]), domain=0)
        manager.add_prototype(0, torch.tensor([0., 1., 0., 0.]), domain=1)
        manager.add_prototype(1, torch.tensor([0., 0., 1., 0.]), domain=0)
        model.prototype_manager = manager
        features = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.]])
        scores = model._forward_with_multi_prototypes(features)['logits']
        self.assertEqual(scores.argmax(dim=1).tolist(), [0, 1])

    def test_gradient_reversal(self):
        import torch
        from models.distribution_module import GradientReversalLayer
        x = torch.tensor([2.0], requires_grad=True)
        layer = GradientReversalLayer()
        layer(x).sum().backward()
        self.assertLess(x.grad.item(), 0)

    def test_herding_retains_one_index_per_class(self):
        import torch
        from my_utils.data_manager import DataManager
        indices = torch.tensor([10, 11, 12, 20, 21])
        vectors = torch.tensor([[1., 0.], [.5, .5], [0., 1.], [1., 1.], [2., 2.]])
        labels = torch.tensor([0, 0, 0, 1, 1])
        selected = DataManager.herding_sample_multiclass(None, indices, vectors, labels, 1)
        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(int(index) in (10, 11, 12) for index in selected), 1)


class PreparedDatasetTests(unittest.TestCase):
    def test_pacs_original_split(self):
        from PIL import Image
        from my_dataset_utils.continual_datasets import PACS
        with tempfile.TemporaryDirectory() as directory:
            for domain in PACS.DOMAINS:
                for label in ('class0', 'class1'):
                    folder = Path(directory) / 'PACS/kfold' / domain / label
                    folder.mkdir(parents=True)
                    for index in range(5):
                        Image.new('RGB', (8, 8)).save(folder / (str(index) + '.png'))
            train = PACS(directory, train=True, seed=0).data
            proto = PACS(directory, train=True, seed=0).data
            test = PACS(directory, train=False, seed=0).data
            for left, reference, right in zip(train, proto, test):
                self.assertEqual(left.indices, reference.indices)
                self.assertFalse(set(left.indices) & set(right.indices))
                self.assertEqual(sorted(left.indices + right.indices), list(range(10)))
                self.assertEqual((len(left), len(right)), (8, 2))

    def test_iwildcam_fixed_manifest_and_loader(self):
        from PIL import Image
        from my_dataset_utils.iwildcam import build_iwildcam_tiny_dataloader
        manifest = json.loads((ROOT / 'protocols/iwildcam/manifest.json').read_text())
        train_ids = {row['image_id'] for row in manifest['train_examples']}
        test_ids = {row['image_id'] for row in manifest['test_examples']}
        self.assertEqual((len(train_ids), len(test_ids)), (274, 134))
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(manifest['tasks']), 7)
        self.assertEqual(len({task['domain_id'] for task in manifest['tasks']}), 6)
        self.assertGreater(len({len(task['class_domain_mask']) for task in manifest['tasks']}), 1)
        with tempfile.TemporaryDirectory() as directory:
            args = load_config(ROOT / 'experiments/supplemental/iWildCamTinyPlus.json', data_root=directory)
            train, test = build_iwildcam_tiny_dataloader(args)
            for stage, task in enumerate(manifest['tasks']):
                train.set_task_id(stage)
                test.set_task_id(stage)
                self.assertEqual(len(train), len(task['indices'][task['domain_name']]['train']))
                self.assertEqual(len(test), len(task['indices'][task['domain_name']]['test']))
                for split, dataset in [('train', train), ('test', test)]:
                    idx = task['indices'][task['domain_name']][split][0]
                    example = manifest[split + '_examples'][idx]
                    path = Path(directory) / 'iWildCamTinyPlus/images' / example['rel_path']
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (256, 256)).save(path)
                    image = dataset[0][0]
                    self.assertEqual(tuple(image.shape), (3, 224, 224))

    def test_prepared_domainnet_does_not_require_archives(self):
        from PIL import Image
        from my_dataset_utils.continual_datasets import DomainNet
        with tempfile.TemporaryDirectory() as directory:
            for split in ('train', 'test'):
                for domain in ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'):
                    folder = Path(directory) / 'DomainNet' / split / domain / 'class0'
                    folder.mkdir(parents=True)
                    Image.new('RGB', (8, 8)).save(folder / 'test.png')
            dataset = DomainNet(directory, download=False, mode='dicil')
            self.assertEqual(len(dataset.data), 6)
            self.assertEqual(len(dataset.data[0]), 1)

    def test_prepared_core50_does_not_trigger_download(self):
        from PIL import Image
        from my_dataset_utils.continual_datasets import CORe50
        with tempfile.TemporaryDirectory() as directory:
            for session in ('s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11'):
                folder = Path(directory) / 'core50_128x128/train' / session / 'o1'
                folder.mkdir(parents=True)
                Image.new('RGB', (8, 8)).save(folder / 'test.png')
            folder = Path(directory) / 'core50_128x128/test/o1'
            folder.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(folder / 'test.png')
            dataset = CORe50(directory, download=False, mode='dicil')
            self.assertEqual(len(dataset.data), 8)


if __name__ == '__main__':
    unittest.main()
