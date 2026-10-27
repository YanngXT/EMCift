from collections import defaultdict
import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from my_utils.inc_net import SimpleCosineIncrementalNet, MultiBranchCosineIncrementalNet_v5, SimpleVitNet
from models.base import BaseLearner
from my_utils.toolkit import tensor2numpy
from models.distribution_module import GradientReversalLayer
from torch.cuda.amp import autocast, GradScaler
import pickle
from itertools import cycle

num_workers = 8

class MultiPrototypeManager:
    def __init__(self):
        self.class_prototypes = {}
        self.class_domains = {}

    def add_prototype(self, class_index, prototype, domain=None):
        if class_index not in self.class_prototypes:
            self.class_prototypes[class_index] = []
            self.class_domains[class_index] = []
        self.class_prototypes[class_index].append(prototype)
        self.class_domains[class_index].append(domain)

    def get_prototypes(self, class_index):
        return self.class_prototypes.get(class_index, [])

    def get_domains(self, class_index):
        return self.class_domains.get(class_index, [])

    def get_vectors_for_draw(self):
        """Return prototype vectors and labels for visualization."""
        vectors = []
        y_true = []
        for class_index, prototypes in self.class_prototypes.items():
            for proto in prototypes:
                vectors.append(proto.cpu().numpy())
                y_true.append(class_index)
        return torch.tensor(vectors), torch.tensor(y_true)

    def has_prototype(self, class_index):
        return class_index in self.class_prototypes

    def get_all_prototypes(self):
        return self.class_prototypes

    def adjust_prototypes_by_memory_shift(self, memory_dataset, old_network, new_network, device):
        """Compensate stored prototypes using replay feature displacement."""
        if old_network is None or memory_dataset is None or len(memory_dataset) == 0:
            print("Skipping prototype adjustment: insufficient data or missing old model.")
            return

        old_network.eval()
        new_network.eval()

        all_features_old = []
        all_features_new = []
        all_labels = []
        all_domains = []

        data_loader = DataLoader(memory_dataset, batch_size=256, shuffle=False)

        print("Calculating feature space drift...")
        with torch.no_grad():
            for images, labels, domains in tqdm(data_loader, desc="Extracting features for drift calculation"):
                images = images.to(device)
                features_old = old_network(images)['features']
                features_new = new_network(images)['features']

                all_features_old.append(features_old.cpu())
                all_features_new.append(features_new.cpu())
                all_labels.append(labels)
                all_domains.append(domains)

        all_features_old = torch.cat(all_features_old, dim=0).to(device)
        all_features_new = torch.cat(all_features_new, dim=0).to(device)
        memory_labels = torch.cat(all_labels, dim=0).to(device)
        memory_domains = torch.cat(all_domains, dim=0).to(device)

        drift_vectors = defaultdict(dict)

        unique_classes = torch.unique(memory_labels)
        for class_idx in unique_classes:
            class_mask = (memory_labels == class_idx)
            domains_in_class = torch.unique(memory_domains[class_mask])

            for domain_idx in domains_in_class:
                mask = class_mask & (memory_domains == domain_idx)
                if mask.sum() > 0:
                    mean_feat_old = all_features_old[mask].mean(dim=0)
                    mean_feat_new = all_features_new[mask].mean(dim=0)

                    drift = mean_feat_new - mean_feat_old
                    drift_vectors[class_idx.item()][domain_idx.item()] = drift

        print("Applying drift to stored prototypes...")
        updated_count = 0
        for class_idx, protos in self.class_prototypes.items():
            domains = self.class_domains.get(class_idx, [])
            for i, domain_val in enumerate(domains):
                if class_idx in drift_vectors and domain_val in drift_vectors[class_idx]:
                    drift = drift_vectors[class_idx][domain_val]

                    original_proto = self.class_prototypes[class_idx][i]
                    self.class_prototypes[class_idx][i] = original_proto.to(device) + drift
                    updated_count += 1

        print(f"Prototype adjustment complete. Updated {updated_count} prototypes.")


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')

        if 'resnet' in args['convnet_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self.batch_size=128
            self.init_lr=args["init_lr"] if args["init_lr"] is not None else  0.01
        else:
            self._network = SimpleVitNet(args, True)
            self.batch_size= args["batch_size"]
            self.init_lr=args["init_lr"]

        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8

        self.args = args
        self.new_domain = None
        self.learned_domains = set()
        self.prototype_manager = MultiPrototypeManager()
        self.class_num_list = []
        self._global_domain_class_counts = defaultdict(lambda: defaultdict(int))

        self._old_network = None
        self.domain_classifier = None
        self.grl = None
        self.cosine_fc = None
        self.mlp_fc = None
        self.lambda_dann = args.get("lambda_dann", 0.1)
        self.lambda_metric = args.get("lambda_metric", 0.5)
        self.margin = args.get("margin", 0.5)  

        if args.get("full_triple_loss", False):
            self.miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets="semihard")
            self.triplet_loss_func = losses.TripletMarginLoss(margin=self.margin)
        else:
            self.contrastive_loss_func = losses.NTXentLoss(temperature=0.07)
            self.miner = None
            self.triplet_loss_func = None

        self.accumulation_steps = 1

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, ):
        embedding_list = self.cache_embedding_list
        label_list = self.cache_label_list

        class_list = np.unique(self.data_manager.mapped_class_mask[self.data_manager.task_id])
        logging.info('class_list {}'.format(class_list))
        for class_index in class_list:
            print('Replacing...', class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self.class_num_list.append(len(embedding))
            self.prototype_manager.add_prototype(class_index, proto, domain=self.cur_domain_name)
            try:
                self._network.fc.weight.data[class_index] = proto.to(self._device)
            except Exception:
                self._network.module.fc.weight.data[class_index] = proto.to(self._device)


    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        if self._cur_task > 0:
            if self._old_network is not None:
                del self._old_network
                torch.cuda.empty_cache()
            self._old_network = self._network.copy().to(self._device)
            self._old_network.eval()

        logging.info('before: new_domain {}, cur_domain {}, learned_domains {}'.format(self.new_domain, data_manager.domain_list[self._cur_task], self.learned_domains))
        self._total_classes = len(np.unique(sum(data_manager.mapped_class_mask[:self._cur_task + 1], [])))

        domain_str = data_manager.domain_list[self._cur_task]
        current_domain_info = data_manager.domain_map[domain_str]
        self.new_domain = current_domain_info not in self.learned_domains
        self.learned_domains.add(current_domain_info)
        self.cur_domain_name = current_domain_info

        should_init_domain_classifier = (self.args.get('traditional_dann', False) and self._cur_task > 0) or (self.new_domain and self._cur_task > 0)

        if should_init_domain_classifier:
            logging.info("Initializing Domain Classifier for DANN (traditional_dann={}, new_domain={}).".format(
                self.args.get('traditional_dann', False), self.new_domain))
            feature_dim = self._network._feature_dim
            num_domains = len(self.learned_domains)
            if self.domain_classifier is not None:
                del self.domain_classifier
                if self.grl is not None:
                    del self.grl
                torch.cuda.empty_cache()
            self.domain_classifier = nn.Linear(feature_dim, num_domains).to(self._device)
            if len(self._multiple_gpus) > 1:
                self.domain_classifier = nn.DataParallel(self.domain_classifier, self._multiple_gpus)
            self.grl = GradientReversalLayer(alpha=1.0)

        logging.info(
            'after: new_domain {}, cur_domain {}, learned_domains {}'.format(self.new_domain, self.cur_domain_name,
                                                                             self.learned_domains))

        print(self._network.device)
        self._network.update_fc(self._total_classes)
        if self._old_network is not None:
            self._old_network.update_fc(self._total_classes)

        self.train_loader = data_manager.get_dataloader(task_id=data_manager.task_id, source="train", mode="train", shuffle=True, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], pin_memory=self.args["pin_mem"], with_domain=True)
        self.train_loader_for_protonet = data_manager.get_dataloader(task_id=data_manager.task_id, source="train", mode="test", shuffle=True, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], pin_memory=self.args["pin_mem"], with_domain=True) 

        if self.args.get("rescale_ce_loss", False):
            logging.info(f"Collecting domain-class counts for task {self._cur_task}...")
            self._collect_domain_class_counts_optimized()

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _collect_domain_class_counts_optimized(self):
        current_task_class_domain_list = self.data_manager._whole_dataset_train.class_domain_mask[self._cur_task]
        for original_class_id, domain_str in current_task_class_domain_list:
            mapped_class_id = self.data_manager.label_map[original_class_id]
            mapped_domain_id = self.data_manager.domain_map[domain_str]
            count = self.data_manager._whole_dataset_train.get_class_domain_num(original_class_id, domain_str)
            self._global_domain_class_counts[mapped_domain_id][mapped_class_id] += count

        total_samples = sum(self._global_domain_class_counts[self.cur_domain_name].values())
        logging.info(f"Collected domain-class counts for domain {self.cur_domain_name}: {total_samples} samples, {len(self._global_domain_class_counts[self.cur_domain_name])} classes")

    def _train(self, train_loader, train_loader_for_protonet):
        self._network.to(self._device)

        if self._cur_task == 0:
            optimizer, scheduler = self.get_optimizer_scheduler()
            if self.args.get("use_mlp", False):
                if len(self._multiple_gpus) > 1:
                    self.mlp_fc = nn.Sequential(
                        nn.Linear(self._network.module.feature_dim, self._network.module.feature_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(self._network.module.feature_dim // 2, self._total_classes)
                    ).to(self._device)
                else:
                    self.mlp_fc = nn.Sequential(
                        nn.Linear(self._network.feature_dim, self._network.feature_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(self._network.feature_dim // 2, self._total_classes)
                    ).to(self._device)
            self._init_train(train_loader, optimizer, scheduler)
            self.construct_dual_branch_network()
        else:
            if len(self._multiple_gpus) > 1:
                self.cosine_fc = self._network.module.fc
            else:
                self.cosine_fc = self._network.fc

            if self.args.get("use_mlp", False):
                feature_dim = self._network._feature_dim if len(self._multiple_gpus) <= 1 else self._network.module._feature_dim
                self.mlp_fc = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(feature_dim // 2, self._total_classes)
                ).to(self._device)

            optimizer, scheduler = self.get_optimizer_scheduler()
            self.speed_domain_class_train(train_loader, optimizer, scheduler)

        self.extract_embedding(self._network)

        memory_indices = self.data_manager.herding_sample_multiclass(
            idx_list=self.cache_idx_list,
            emb_list=self.cache_embedding_list,
            label_list=self.cache_label_list,
            num_samples_per_class=self.args.get("memory_per_class", 1)
        )

        self.data_manager.add_memory(memory_indices, self._cur_task)

        self.replace_fc()

        if self._cur_task > 0 and self.args.get("shift", 1):
            memory_dataset = self.data_manager.get_memory_dataset(self._cur_task, mode='train_proto')
            if memory_dataset is not None and len(memory_dataset) > 0:
                self.prototype_manager.adjust_prototypes_by_memory_shift(
                    memory_dataset, self._old_network, self._network, self._device)

    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet_v5(self.args, True)
        if len(self._multiple_gpus) > 1:
            network.construct_dual_branch_network(self._network.module)
        else:
            network.construct_dual_branch_network(self._network)
        network.set_prototype_manager(self.prototype_manager)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            network = nn.DataParallel(network, self._multiple_gpus)
        self._network = network.to(self._device)

    def get_optimizer_scheduler(self):
        total_params = sum(p.numel() for p in self._network.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        params_to_optimize = list(self._network.parameters())
        if self.domain_classifier:
            params_to_optimize += list(self.domain_classifier.parameters())
            total_params += sum(p.numel() for p in self.domain_classifier.parameters())
            total_trainable_params += sum(p.numel() for p in self.domain_classifier.parameters() if p.requires_grad)
            print(f'{total_params:,} total parameters (including domain classifier).')
            print(f'{total_trainable_params:,} training parameters (including domain classifier).')

        if self.mlp_fc is not None:
            params_to_optimize += list(self.mlp_fc.parameters())
            total_params += sum(p.numel() for p in self.mlp_fc.parameters())
            total_trainable_params += sum(p.numel() for p in self.mlp_fc.parameters() if p.requires_grad)
            print(f'{total_params:,} total parameters (including MLP classifier).')
            print(f'{total_trainable_params:,} training parameters (including MLP classifier).')

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params_to_optimize, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(params_to_optimize, lr=self.init_lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError('Optimizer not implemented: {}'.format(self.args['optimizer']))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        return optimizer, scheduler

    def extract_embedding(self, model):
        """
        Extract embeddings from the model for the training data.
        This method is used to update the feature cache.
        """
        if self.cosine_fc is not None:
            if len(self._multiple_gpus) > 1:
                model.module.fc = self.cosine_fc
            else:
                model.fc = self.cosine_fc

        model = model.eval()
        idx_list = []
        embedding_list = []
        label_list = []
        with torch.no_grad():
            prog_bar = tqdm(enumerate(self.train_loader_for_protonet), total=len(self.train_loader_for_protonet))
            for i, batch in prog_bar:
                idx, data, label = batch[:3]
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model(data)['features']
                idx_list.append(idx.cpu())
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
                prog_bar.set_description('Extracting features for [replace fc] and [herding]')
        idx_list = torch.cat(idx_list, dim=0)
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        self.cache_idx_list = idx_list
        self.cache_embedding_list = embedding_list
        self.cache_label_list = label_list
        self.cache_domain_list = self.cur_domain_name * torch.ones_like(self.cache_label_list, dtype=torch.long).cpu()

    def _init_train(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            prog_bar2 = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, batch in prog_bar2:
                _, inputs, targets = batch[:3]
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self.args.get("use_mlp", False):
                    features = self._network(inputs)['features']
                    logits = self.mlp_fc(features) # type: ignore
                else:
                    logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                info = "Task {}, Epoch {}/{} Batch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1, self.args['tuned_epoch'],
                    i + 1, len(train_loader),
                    losses / (i + 1),
                    np.around(tensor2numpy(correct) * 100 / total, decimals=2),
                )
                prog_bar2.set_description(info)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1, self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)
            if losses / len(train_loader) < 1e-9:
                break
        logging.info(info)

    def speed_domain_class_train(self, train_loader, optimizer, scheduler):
        inc_epoch = self.args.get('inc_epoch', 10)  

        prog_bar = tqdm(range(inc_epoch), desc="Epoch Progress")

        global_domain_class_weights_map = None
        if self.args.get("rescale_ce_loss", False): 
            global_domain_class_weights_map = {}
            for d_id, class_counts_in_domain in self._global_domain_class_counts.items():
                for c_id, count in class_counts_in_domain.items():
                    if count > 0:
                        global_domain_class_weights_map[(d_id, c_id)] = 1.0 / count

            all_weights_values = torch.tensor(list(global_domain_class_weights_map.values()), device=self._device, dtype=torch.float)
            if all_weights_values.numel() > 0 and all_weights_values.mean() > 0:
                normalized_all_weights_values = all_weights_values / all_weights_values.mean()
                normalized_weights_list = normalized_all_weights_values.tolist()
                keys_list = list(global_domain_class_weights_map.keys())
                for i, key in enumerate(keys_list):
                    global_domain_class_weights_map[key] = normalized_weights_list[i]
            logging.info(f"Calculated Global Domain-Class Weights for Task {self._cur_task}: {global_domain_class_weights_map}")
            del all_weights_values
            torch.cuda.empty_cache()


        best_loss = float('inf')  
        best_train_acc = float('-inf')  
        best_ce = float('inf')  
        patience = self.args.get('patience', 3)  
        patience_counter = 0

        memory_loader = None
        if self._cur_task > 0:
            memory_dataset = self.data_manager.get_memory_dataset(self._cur_task)
            if memory_dataset is not None and len(memory_dataset) > 0:
                memory_loader = DataLoader(memory_dataset, batch_size=self.args["batch_size"], shuffle=True)

        for epoch in range(inc_epoch):
            self._network.train()
            if self.domain_classifier:
                self.domain_classifier.train()

            total_loss, ce_loss_sum, dann_loss_sum, metric_loss_sum = 0.0, 0.0, 0.0, 0.0
            correct, total = 0, 0

            if memory_loader:
                memory_loader = cycle(memory_loader)

            prog_bar1 = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for i, batch in prog_bar1:
                _, inputs, targets = batch[:3]
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                current_domain_labels_batch = torch.full((inputs.size(0),), self.cur_domain_name, dtype=torch.long, device=self._device)

                mem_inputs_batch, mem_targets_batch, mem_domains_batch = None, None, None
                if memory_loader is not None:
                    mem_inputs_batch, mem_targets_batch, mem_domains_batch = next(memory_loader) # type: ignore
                    mem_inputs_batch, mem_targets_batch, mem_domains_batch = mem_inputs_batch.to(self._device), mem_targets_batch.to(self._device), mem_domains_batch.to(self._device)
                if mem_inputs_batch is not None and mem_targets_batch is not None and mem_domains_batch is not None:
                    all_inputs = torch.cat([inputs, mem_inputs_batch])
                    all_targets = torch.cat([targets, mem_targets_batch])
                    all_domain_labels = torch.cat([current_domain_labels_batch, mem_domains_batch])
                else:
                    all_inputs = inputs
                    all_targets = targets
                    all_domain_labels = current_domain_labels_batch

                try:
                    if self.mlp_fc is not None:
                        backbone_output = self._network.extract_vector(all_inputs)
                        logits = self.mlp_fc(backbone_output)
                        output = {"logits": logits, "features": backbone_output}
                    else:
                        output = self._network(all_inputs)
                except Exception:
                    if self.mlp_fc is not None:
                        backbone_output = self._network.module.extract_vector(all_inputs)
                        logits = self.mlp_fc(backbone_output)
                        output = {"logits": logits, "features": backbone_output}
                    else:
                        output = self._network.module(all_inputs)

                logits = output["logits"]
                combined_features = output["features"]

                if self.args.get("rescale_ce_loss", False) and global_domain_class_weights_map is not None:
                    sample_weights_batch = torch.zeros_like(all_targets, dtype=torch.float, device=self._device)
                    for idx in range(len(all_targets)):
                        d_id = all_domain_labels[idx].item()
                        c_id = all_targets[idx].item()
                        sample_weights_batch[idx] = global_domain_class_weights_map.get((d_id, c_id), 1.0)

                    loss_ce = F.cross_entropy(logits, all_targets, reduction='none')
                    loss_ce = (loss_ce * sample_weights_batch).mean()
                else:
                    loss_ce = F.cross_entropy(logits, all_targets)

                loss_dann = torch.tensor(0.0).to(self._device)
                if (self.new_domain or self.args.get('traditional_dann', False)) and self.domain_classifier is not None and self.lambda_dann > 0:
                    if self.grl is not None:
                        reversed_features = self.grl(combined_features)
                        domain_logits = self.domain_classifier(reversed_features)
                        loss_dann = self.lambda_dann * F.cross_entropy(domain_logits, all_domain_labels)

                loss_metric = torch.tensor(0.0).to(self._device)
                if len(torch.unique(all_targets)) > 1 and self.lambda_metric > 0:
                    if self.args.get("full_triple_loss", False):
                        hard_pairs = self.miner(combined_features, all_targets) # type: ignore
                        loss_metric = self.triplet_loss_func(combined_features, all_targets, hard_pairs) # type: ignore
                    else:
                        loss_metric = self._compute_separation_triplet_loss(combined_features, all_targets)
                    loss_metric = self.lambda_metric * loss_metric


                loss = loss_ce + loss_dann + loss_metric
                loss = loss / self.accumulation_steps

                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()


                total_loss += loss.item()
                ce_loss_sum += loss_ce.item()
                dann_loss_sum += loss_dann.item()
                metric_loss_sum += loss_metric.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(all_targets.expand_as(preds)).cpu().sum()
                total += len(all_targets)

                prog_bar1.set_postfix({
                    'Loss': f'{total_loss / (i + 1):.3f}',
                    'CE': f'{ce_loss_sum / (i + 1):.3f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
                if self.args.get("rescale_ce_loss", False) and global_domain_class_weights_map is not None:
                    del sample_weights_batch
                del all_domain_labels, current_domain_labels_batch
                del all_inputs, all_targets, logits, combined_features
                torch.cuda.empty_cache()


            scheduler.step()

            avg_total_loss = total_loss / len(train_loader)
            avg_ce_loss = ce_loss_sum / len(train_loader)
            avg_dann_loss = dann_loss_sum / len(train_loader)
            avg_metric_loss = metric_loss_sum / len(train_loader)
            current_train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            logging.info(
                f"Task {self._cur_task}, Epoch {epoch + 1}/{inc_epoch} => "
                f"Total Loss: {avg_total_loss:.3f}, "
                f"CE Loss: {avg_ce_loss:.3f}, "
                f"DANN Loss: {avg_dann_loss:.3f}, "
                f"Metric Loss: {avg_metric_loss:.3f}, "
                f"Accy: {current_train_acc:.2f}%"
            )
            improved_acc = (current_train_acc > best_train_acc)
            improved_loss = (avg_total_loss < best_loss)
            improved_ce = (avg_ce_loss < best_ce)

            if improved_acc and (improved_loss or improved_ce):
                best_train_acc = current_train_acc
                best_loss = avg_total_loss
                best_ce = avg_ce_loss
                patience_counter = 0
                logging.info(f"Both training accuracy improved to {best_train_acc:.2f}% and (total loss improved to {best_loss:.3f} or ce loss improved to {best_ce:.3f})")
            else:
                patience_counter += 1
                logging.info(f"Early stopping patience counter: {patience_counter}/{patience}")
                if not improved_acc:
                    logging.info("Training accuracy did not improve in this epoch.")
                if not improved_loss:
                    logging.info("Total loss did not improve in this epoch.")
                if not improved_ce:
                    logging.info("CE loss did not improve in this epoch.")
            if patience_counter >= patience:
                logging.info(f"Early stopping! Neither training accuracy nor total loss has improved for {patience} epochs.")
                break
            if current_train_acc > 95:
                logging.info(f"Training accuracy improved to {current_train_acc:.2f}%")
                break
            prog_bar.set_description(
                f"Task {self._cur_task}, Epoch {epoch + 1}/{inc_epoch} => Total Loss: {total_loss / len(train_loader):.3f}, Accy: {current_train_acc:.2f}%"
            )

        if self.cosine_fc is not None:
            if len(self._multiple_gpus) > 1:
                self._network.module.fc = self.cosine_fc
            else:
                self._network.fc = self.cosine_fc
            logging.info("Restored CosineLinear classifier for feature extraction and prototype computation")

    def _compute_separation_triplet_loss(self, features, labels):
        """Compute the vectorized separation loss between different classes."""
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return torch.tensor(0.0).to(self._device)

        features= F.normalize(features, p=2, dim=1)
        dist_matrix = torch.cdist(features, features, p=2)

        labels_expanded1 = labels.unsqueeze(1)
        labels_expanded2 = labels.unsqueeze(0)
        negative_mask = (labels_expanded1 != labels_expanded2)

        hinge_losses = torch.clamp(self.margin - dist_matrix, min=0)

        hinge_losses = hinge_losses * negative_mask.float()

        num_negatives = negative_mask.sum(dim=1).float()
        num_negatives = torch.clamp(num_negatives, min=1.0)

        anchor_losses = hinge_losses.sum(dim=1) / num_negatives

        return anchor_losses.mean()
