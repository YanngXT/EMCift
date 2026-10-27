import copy
import os
import random
import sys

import math
import numpy as np
import torch
import yaml
from torch.utils.data.dataset import Subset, ConcatDataset
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(root_dir)
from my_dataset_utils.continual_datasets import CORe50, DomainNet, MNISTM, SynDigit, MNIST_RGB, SVHN, PACS


class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes

    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_transform(is_train, args):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    return transform

def build_continual_dataloader(args):
    data_seed = args.get('data_seed', 0)
    os.environ['PYTHONHASHSEED'] = str(data_seed)
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(data_seed)

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args["joint_train"]:
        mode = 'joint'
    elif args["cross_domain_class_inc"]:
        mode = 'dicil'

    if mode in ['dicil']:
        if 'iDigits' in args["dataset"]:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            splited_dataset = dict()
            for i in range(len(dataset_list)):
                dataset_train, dataset_train_proto, dataset_val = get_dataset(
                    dataset=dataset_list[i],
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                splited_dataset[dataset_list[i]] = (dataset_train, dataset_train_proto, dataset_val)
        elif 'DomainNet' in args["dataset"]:
            dataset_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
            splited_dataset = dict()
            dataset_train, dataset_train_proto, dataset_val = get_dataset(
                dataset=args["dataset"],
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )
            for i, domain_name in enumerate(dataset_list):
                splited_dataset[domain_name] = (dataset_train[i], dataset_train_proto[i], dataset_val[i])
        elif 'CORe50' in args["dataset"]:
            dataset_list = ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']
            splited_dataset = dict()
            dataset_train, dataset_train_proto, dataset_val = get_dataset(
                dataset=args["dataset"],
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )
            for i, domain_name in enumerate(dataset_list):
                splited_dataset[domain_name] = (dataset_train[i], dataset_train_proto[i], dataset_val)
        elif 'PACS' in args["dataset"]:
            dataset_list = ['art_painting', 'cartoon', 'photo', 'sketch']
            splited_dataset = dict()
            dataset_train = PACS(args["data_path"], train=True,  transform=transform_train, seed=args.get('data_seed', 0)).data
            dataset_prot  = PACS(args["data_path"], train=True,  transform=transform_val,   seed=args.get('data_seed', 0)).data
            dataset_val   = PACS(args["data_path"], train=False, transform=transform_val,   seed=args.get('data_seed', 0)).data
            for i, domain_name in enumerate(dataset_list):
                splited_dataset[domain_name] = (dataset_train[i], dataset_prot[i], dataset_val[i])

    splited_dataset_indices, class_mask, domain_list, class_sample_counts = build_dicil_scenario(splited_dataset_by_domain=splited_dataset, args=args)
    Whole_dataset_train = WholeDataset(datasets=splited_dataset, indices=splited_dataset_indices, class_domain_mask=class_mask, domain_name_list=domain_list, mode='train', class_sample_counts=class_sample_counts)
    Whole_dataset_test = WholeDataset(datasets=splited_dataset, indices=splited_dataset_indices, class_domain_mask=class_mask, domain_name_list=domain_list, mode='test', class_sample_counts=class_sample_counts)
    return Whole_dataset_train, Whole_dataset_test

def get_dataset(dataset, transform_train, transform_val, mode, args):

    if dataset == 'MNIST':
        dataset_train = MNIST_RGB(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_train)
        dataset_prot = MNIST_RGB(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_val)
        dataset_val = MNIST_RGB(args["data_path"], train=False, download=args.get("download_data", False), transform=transform_val)

    elif dataset == 'SVHN':
        dataset_train = SVHN(args["data_path"], split='train', download=args.get("download_data", False), transform=transform_train)
        dataset_prot = SVHN(args["data_path"], split='train', download=args.get("download_data", False), transform=transform_val)
        dataset_val = SVHN(args["data_path"], split='test', download=args.get("download_data", False), transform=transform_val)

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_train, mode=mode).data
        dataset_prot = DomainNet(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_val, mode=mode).data
        dataset_val = DomainNet(args["data_path"], train=False, download=args.get("download_data", False), transform=transform_val, mode=mode).data

    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_train)
        dataset_prot = MNISTM(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_val)
        dataset_val = MNISTM(args["data_path"], train=False, download=args.get("download_data", False), transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_train)
        dataset_prot = SynDigit(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_val)
        dataset_val = SynDigit(args["data_path"], train=False, download=args.get("download_data", False), transform=transform_val)

    elif dataset == 'CORe50':
        dataset_train = CORe50(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_train, mode=mode).data
        dataset_prot = CORe50(args["data_path"], train=True, download=args.get("download_data", False), transform=transform_val, mode=mode).data
        dataset_val = CORe50(args["data_path"], train=False, download=args.get("download_data", False), transform=transform_val, mode=mode).data
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))

    return dataset_train, dataset_prot, dataset_val

def split_single_dataset(dataset_train, dataset_val, domain_id, args):
    """Partition one domain using the configured data seed."""
    config = yaml.safe_load(open(args["protocol_path"], "r", encoding="utf-8"))
    nb_classes = len(dataset_val.classes)
    labels = list(range(nb_classes))

    if True:
        random.shuffle(labels)

    train_indices_by_label = {label: [i for i, t in enumerate(dataset_train.targets) if int(t) == label] for label in labels}
    test_indices_by_label = {label: [i for i, t in enumerate(dataset_val.targets) if int(t) == label] for label in labels}

    visited = set()
    merged_mask_list, merged_train_indices_list, merged_test_indices_list = [], [], []
    class_sample_counts = []

    for label in labels:
        if label in visited:
                continue

        num_classes = random.randint(config['min_class_num'], config['max_class_num'])
        available = [l for l in labels if l not in visited]
        task_labels = [label] + random.sample([l for l in available if l != label], min(num_classes - 1, len(available) - 1))
        visited.update(task_labels)

        merged_train = sum([train_indices_by_label[l] for l in task_labels], [])
        merged_test = sum([test_indices_by_label[l] for l in task_labels], [])

        merged_mask_list.append([(l, f'D{domain_id}') for l in task_labels])
        merged_train_indices_list.append(merged_train)
        merged_test_indices_list.append(merged_test)

        task_class_count = {
            'train': {(l, f'D{domain_id}'): len(train_indices_by_label[l]) for l in task_labels},
            'test': {(l, f'D{domain_id}'): len(test_indices_by_label[l]) for l in task_labels}
        }
        class_sample_counts.append(task_class_count)

    return merged_mask_list, merged_train_indices_list, merged_test_indices_list, class_sample_counts

def build_dicil_scenario(splited_dataset_by_domain, args):
    """Build the seeded class-domain incremental stream."""
    datasets = list()
    class_mask = list()
    domain_list = list()
    class_sample_counts_all = list()

    for i, domain_name in enumerate(splited_dataset_by_domain.keys()):
        merged_mask_list, merged_train_indices_list, merged_test_indices_list, class_sample_counts = split_single_dataset(
            dataset_train=splited_dataset_by_domain[domain_name][0], dataset_val=splited_dataset_by_domain[domain_name][2], domain_id=i, args=args)
        for j in range(len(merged_mask_list)):
            class_mask.append(merged_mask_list[j])
            domain_list.append([item[1] for item in merged_mask_list[j]])
            datasets.append({domain_name: {'train': merged_train_indices_list[j], 'test': merged_test_indices_list[j]}})
            class_sample_counts_all.append(class_sample_counts[j])

    shuffled_indices = list(range(len(datasets)))
    random.shuffle(shuffled_indices)
    datasets = [datasets[i] for i in shuffled_indices]
    class_mask = [class_mask[i] for i in shuffled_indices]
    domain_list = [domain_list[i] for i in shuffled_indices]
    class_sample_counts_all = [class_sample_counts_all[i] for i in shuffled_indices]

    return datasets, class_mask, domain_list, class_sample_counts_all

class WholeDataset(torch.utils.data.Dataset): # type: ignore
    def __init__(self, datasets, indices, class_domain_mask, domain_name_list, mode, class_sample_counts=None):
        super(WholeDataset, self).__init__()
        self.datasets = datasets
        self.indices = indices
        self.class_domain_mask = class_domain_mask
        self.domain_name_list = domain_name_list
        self.task_id = 0
        self.mode = mode
        self.class_sample_counts = class_sample_counts or []
        self._cur_domain_name = list(self.indices[0].keys())[0] if len(self.indices) > 0 else None

    def __len__(self, task_id=None):
        task_id = task_id if task_id is not None else self.task_id
        if self.mode == 'train':
            return sum([len(self.indices[task_id][domain_name]['train']) for domain_name in self.indices[task_id].keys()])
        else:
            return sum([len(self.indices[task_id][domain_name]['test']) for domain_name in self.indices[task_id].keys()])

    def update_task(self):
        self.task_id += 1
        self._cur_domain_name = list(self.indices[self.task_id].keys())[0]

    def set_task_id(self, task_id):
        self.task_id = task_id
        self._cur_domain_name = list(self.indices[task_id].keys())[0]

    def __getitem__(self, idx):
        indices = self.indices[self.task_id]
        cur_domain_name = self._cur_domain_name
        cur_dataset = self.datasets[cur_domain_name]
        if self.mode == 'train':
            train_idx = indices[cur_domain_name]['train'][idx]
            train_data = cur_dataset[0][train_idx]
            train_proto_data = cur_dataset[1][train_idx]
            return train_data[0], train_proto_data[0], train_data[1]
        else:
            test_idx = indices[cur_domain_name]['test'][idx]
            test_data = cur_dataset[2][test_idx]
            return test_data[0], test_data[1]

    def analysis(self):
        seen_class_domain_list = []
        for i in range(len(self.indices)):
            cur_class_domain_list = self.class_domain_mask[i]
            seen_class_domain_list.extend(cur_class_domain_list)
            print('Task {}:'.format(i))
            print('Sample counts: {}'.format(self.__len__(i)))
            print('Current class_domain_list: {}'.format(set(cur_class_domain_list)))
            print('Seen class_domain_list: {}'.format(set(seen_class_domain_list)))

    def get_class_domain_num(self, class_id, domain_id):
        """Return the cached sample count for an observed class-domain pair."""
        total_samples = 0
        target_tuple = (class_id, domain_id)
        mode_key = 'train' if self.mode == 'train' else 'test'

        for task_idx, task_class_domain_mask in enumerate(self.class_domain_mask):
            if target_tuple in task_class_domain_mask and task_idx < len(self.class_sample_counts):
                total_samples += self.class_sample_counts[task_idx][mode_key].get(target_tuple, 0)

        return total_samples
