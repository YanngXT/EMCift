import logging
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class DataManager(object):
    def __init__(self, Whole_dataset_train, Whole_dataset_test, args=None):
        self._whole_dataset_train = Whole_dataset_train
        self._whole_dataset_test = Whole_dataset_test
        self._args = args
        self._setup_data(args)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataloader(self, task_id, source, mode, shuffle, batch_size, num_workers=4, pin_memory=True, with_domain=False):
        if source == "train" and mode == 'train':
            self._whole_dataset_train.set_task_id(task_id)
            dataloader = DataLoader(DummyDataset(self._whole_dataset_train, self.label_map, mode='train', with_domain=with_domain, domain=self.domain_map[self.domain_list[task_id]]), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        elif source == "train" and mode == 'test':
            self._whole_dataset_train.set_task_id(task_id)
            dataloader = DataLoader(DummyDataset(self._whole_dataset_train, self.label_map, mode='train_proto', with_domain=with_domain, domain=self.domain_map[self.domain_list[task_id]]), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        elif source == "cur_test" and mode == 'test':
            self._whole_dataset_test.set_task_id(task_id)
            dataloader = DataLoader(DummyDataset(self._whole_dataset_test, self.label_map, mode='test', with_domain=with_domain, domain=self.domain_map[self.domain_list[task_id]]), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        else:
            raise ValueError("Unknown data source {} or mode {}.".format(source, mode))
        return dataloader

    def _setup_data(self, args):
        self.class_mask = []
        self.domain_list = []
        for task_id in range(len(self._whole_dataset_train.class_domain_mask)):
            class_domain_list = self._whole_dataset_train.class_domain_mask[task_id]
            self.class_mask.append(list(set([item[0] for item in class_domain_list])))
            self.domain_list.append(list(set([item[1] for item in class_domain_list]))[0])

        global_classes = []
        for task_classes in self.class_mask:
            for cls in task_classes:
                if cls not in global_classes:
                    global_classes.append(cls)
        self._class_order = global_classes
        self.n_tot_cls = len(global_classes)

        self._increments = []
        seen_classes = set()
        for task_classes in self.class_mask:
            new_classes = [cls for cls in task_classes if cls not in seen_classes]
            self._increments.append(len(new_classes))
            seen_classes.update(new_classes)

        self.label_map = {cls: idx for idx, cls in enumerate(global_classes)}
        self.get_mapped_class_mask()

        seen_domains = []
        for domain_str in self.domain_list:
            if domain_str not in seen_domains:
                seen_domains.append(domain_str)
        self.domain_map = {domain_str: idx for idx, domain_str in enumerate(seen_domains)}

        logging.info('original class mask: {}, mapped class mask: {}'.format(self.class_mask, self.mapped_class_mask))
        logging.info('original domain list: {}, mapped domain list: {}'.format(self.domain_list, self.domain_map))


    def get_mapped_class_mask(self):
        self.mapped_class_mask = []
        for task_classes in self.class_mask:
            mapped_task = [self.label_map[cls] for cls in task_classes]
            self.mapped_class_mask.append(mapped_task)

    def add_memory(self, memory_indices:torch.Tensor, task_id:int):
        """Store exemplar indices rather than feature vectors."""
        if not hasattr(self, 'memory_indices'):
            self.memory_indices = {}

        self.memory_indices[task_id] = memory_indices.cpu()
        print(f"DEBUG: add_memory - stored {len(memory_indices)} indices for task {task_id}")

    def get_memory_dataset(self, task_id, mode='train'):
        """Return replay examples from tasks preceding task_id."""
        if not hasattr(self, 'memory_indices') or len(self.memory_indices) == 0:
            return None

        return MemoryDataset(self._whole_dataset_train, self.memory_indices, task_id, self.domain_list, self.label_map, self.domain_map, mode)

    def herding_sample_multiclass(
            self,
            idx_list: torch.Tensor,
            emb_list: torch.Tensor,
            label_list: torch.Tensor,
            num_samples_per_class: int
    ) -> torch.Tensor:
        assert idx_list.shape[0] == emb_list.shape[0] == label_list.shape[0], "Input tensors must have the same length."

        import time
        start_time = time.time()
        unique_labels = torch.unique(label_list)
        final_indices = []
        logging.info(f"Herding: Processing {len(unique_labels)} classes, selecting {num_samples_per_class} samples per class")

        for label in unique_labels:
            indices = torch.where(label_list == label)[0]
            if len(indices) == 0:
                continue

            current_indices = idx_list[indices]
            current_embeddings = emb_list[indices]

            if len(current_indices) <= num_samples_per_class:
                final_indices.append(current_indices)
                continue

            class_mean = current_embeddings.mean(dim=0)
            selected_positions = []

            for k in range(num_samples_per_class):
                if len(selected_positions) == 0:
                    selected_sum = torch.zeros_like(class_mean)
                else:
                    selected_sum = current_embeddings[selected_positions].sum(dim=0)


                num_selected = len(selected_positions)
                new_means = (selected_sum + current_embeddings) / (num_selected + 1)
                scores = torch.norm(new_means - class_mean, p=2, dim=1)

                if len(selected_positions) > 0:
                    scores[selected_positions] = float('inf')
                best_position = torch.argmin(scores).item()
                selected_positions.append(best_position)

            selected_indices = current_indices[selected_positions]
            final_indices.append(selected_indices)

        elapsed_time = time.time() - start_time
        logging.info(f"Herding completed in {elapsed_time:.2f}s, selected {sum(len(fi) for fi in final_indices)} samples total")
        return torch.cat(final_indices, dim=0)


class DummyDataset(Dataset):
    def __init__(self, origin_dataset, map_dict, mode='train', with_domain=False, domain=None):
        super(DummyDataset, self).__init__()
        self.origin_dataset = origin_dataset
        self.map_dict = map_dict
        self.mode = mode
        self._with_domain = with_domain
        self._domain = domain
        if with_domain:
            assert self._domain is not None, "domain is required"

    def __len__(self):
        return len(self.origin_dataset)

    def __getitem__(self, idx):
        data = self.origin_dataset[idx]
        if self.mode == 'train':
            image, _, label = data
        elif self.mode == 'train_proto':
            _, image, label = data
        elif self.mode == 'test':
            image, label = data

        label = self.map_dict.get(label, label)

        if self._with_domain:
            domain = self._domain
            return idx, image, label, domain
        else:
            return idx, image, label


class MemoryDataset(Dataset):
    """Replay images selected by their original dataset indices."""

    def __init__(self, whole_dataset_train, memory_indices, max_task_id,
                 domain_list, label_map, domain_map, mode='train'):
        self.memory_indices = memory_indices
        self.max_task_id    = max_task_id
        self.domain_list    = domain_list
        self.label_map      = label_map
        self.domain_map     = domain_map
        self.mode           = mode

        self.datasets = whole_dataset_train.datasets
        self.indices  = whole_dataset_train.indices

        self.index_map = []

        split_key = 'train' if mode in ('train', 'train_proto') else 'test'

        for t_id in range(max_task_id):
            if t_id not in memory_indices:
                continue
            task_indices = self.indices[t_id]

            for global_idx in memory_indices[t_id]:
                global_idx = int(global_idx)

                offset = 0
                resolved = False
                for domain_name, splits in task_indices.items():
                    domain_idx_list = splits[split_key]
                    n = len(domain_idx_list)
                    if global_idx < offset + n:
                        local_pos  = global_idx - offset
                        actual_idx = domain_idx_list[local_pos]
                        self.index_map.append((t_id, domain_name, actual_idx))
                        resolved = True
                        break
                    offset += n

                if not resolved:
                    first_domain = next(iter(task_indices))
                    fallback_list = task_indices[first_domain][split_key]
                    actual_idx = fallback_list[0] if fallback_list else 0
                    self.index_map.append((t_id, first_domain, actual_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        task_id, domain_name, actual_idx = self.index_map[idx]
        cur_dataset = self.datasets[domain_name]

        if self.mode == 'train':
            data  = cur_dataset[0][actual_idx]
            image = data[0]
            label = data[1]
        elif self.mode == 'train_proto':
            data  = cur_dataset[1][actual_idx]
            image = data[0]
            label = data[1]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        mapped_class_id  = self.label_map.get(label, label)
        domain_str       = self.domain_list[task_id]
        mapped_domain_id = self.domain_map[domain_str]

        return image, mapped_class_id, mapped_domain_id
