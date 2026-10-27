import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


class ManifestImageDataset(torch.utils.data.Dataset):
    def __init__(self, examples, data_root, transform=None):
        self.examples = examples
        self.data_root = Path(data_root)
        self.transform = transform
        self.targets = [int(row["category_id"]) for row in examples]
        self.classes = sorted(set(self.targets))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        path = self.data_root / row["rel_path"]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["category_id"])


class IWildCamTinyWholeDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, indices, class_domain_mask, domain_name_list, mode, class_sample_counts=None):
        self.datasets = datasets
        self.indices = indices
        self.class_domain_mask = class_domain_mask
        self.domain_name_list = domain_name_list
        self.task_id = 0
        self.mode = mode
        self.class_sample_counts = class_sample_counts or []
        self._cur_domain_name = list(self.indices[0].keys())[0] if self.indices else None

    def __len__(self, task_id=None):
        task_id = self.task_id if task_id is None else task_id
        split = "train" if self.mode == "train" else "test"
        return sum(len(v[split]) for v in self.indices[task_id].values())

    def update_task(self):
        self.task_id += 1
        self._cur_domain_name = list(self.indices[self.task_id].keys())[0]

    def set_task_id(self, task_id):
        self.task_id = task_id
        self._cur_domain_name = list(self.indices[task_id].keys())[0]

    def __getitem__(self, idx):
        task_indices = self.indices[self.task_id]
        domain_name = self._cur_domain_name
        cur_dataset = self.datasets[domain_name]
        if self.mode == "train":
            actual_idx = task_indices[domain_name]["train"][idx]
            train_data = cur_dataset[0][actual_idx]
            proto_data = cur_dataset[1][actual_idx]
            return train_data[0], proto_data[0], train_data[1]
        actual_idx = task_indices[domain_name]["test"][idx]
        test_data = cur_dataset[2][actual_idx]
        return test_data[0], test_data[1]

    def get_class_domain_num(self, class_id, domain_id):
        total = 0
        split = "train" if self.mode == "train" else "test"
        target = (class_id, domain_id)
        for task_idx, mask in enumerate(self.class_domain_mask):
            if target in mask and task_idx < len(self.class_sample_counts):
                total += self.class_sample_counts[task_idx][split].get(target, 0)
        return total


def build_transform(is_train):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def _counts_for_task(examples, indices, class_domain_mask):
    counts = {tuple(item): 0 for item in class_domain_mask}
    for idx in indices:
        row = examples[idx]
        key = (int(row["category_id"]), row["domain_id"])
        if key in counts:
            counts[key] += 1
    return counts


def build_iwildcam_tiny_dataloader(args):
    manifest_path = Path(args["protocol_path"])
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    data_root = Path(args["data_path"]) / "iWildCamTinyPlus" / manifest["data_root"]
    train_examples = manifest["train_examples"]
    test_examples = manifest["test_examples"]
    for row in train_examples + test_examples:
        row["category_id"] = int(row["category_id"])
        row["domain_id"] = manifest["domain_ids"][str(row["location"])]

    train_ds = ManifestImageDataset(train_examples, data_root, build_transform(True))
    proto_ds = ManifestImageDataset(train_examples, data_root, build_transform(False))
    test_ds = ManifestImageDataset(test_examples, data_root, build_transform(False))

    domain_names = sorted({task["domain_name"] for task in manifest["tasks"]})
    datasets = {domain_name: (train_ds, proto_ds, test_ds) for domain_name in domain_names}

    indices = []
    class_domain_mask = []
    domain_name_list = []
    class_sample_counts = []
    for task in manifest["tasks"]:
        task_indices = task["indices"]
        mask = [(int(cls), str(domain)) for cls, domain in task["class_domain_mask"]]
        indices.append(task_indices)
        class_domain_mask.append(mask)
        domain_name_list.append([task["domain_id"] for _ in mask])

        domain_name = task["domain_name"]
        train_idx = task_indices[domain_name]["train"]
        test_idx = task_indices[domain_name]["test"]
        class_sample_counts.append(
            {
                "train": _counts_for_task(train_examples, train_idx, mask),
                "test": _counts_for_task(test_examples, test_idx, mask),
            }
        )

    whole_train = IWildCamTinyWholeDataset(
        datasets=datasets,
        indices=indices,
        class_domain_mask=class_domain_mask,
        domain_name_list=domain_name_list,
        mode="train",
        class_sample_counts=class_sample_counts,
    )
    whole_test = IWildCamTinyWholeDataset(
        datasets=datasets,
        indices=indices,
        class_domain_mask=class_domain_mask,
        domain_name_list=domain_name_list,
        mode="test",
        class_sample_counts=class_sample_counts,
    )
    return whole_train, whole_test
