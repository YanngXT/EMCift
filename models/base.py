# type: ignore
import copy
import logging
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from my_utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 1

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

        self.args = args
        self.data_manager = None
        self.prototype_manager = None
        self.cache_embedding_list = None
        self.cache_label_list = None
        self.cache_domain_list = None
        self.test_loader = None
        self.gen_loader = None
        self._class_means = None
        self.acc_matrix = None

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim


    def tsne(self, Normalize=False, task=-1):

        import umap.umap_ as umap
        import matplotlib.pyplot as plt
        import os
        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages

        print('now draw tsne results of extracted features.')
        tot_classes = self.data_manager.n_tot_cls

        val_vectors_list, val_y_true_list, val_domains_list = [], [], []
        for tid in range(len(self.data_manager.class_mask)):
            task_test_loader = self.data_manager.get_dataloader(task_id=tid, source='cur_test', mode='test', shuffle=False, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], pin_memory=self.args["pin_mem"], with_domain=True)
            task_vectors, task_y_true, task_domains = self._extract_vectors(task_test_loader, require_domain=True)
            val_vectors_list.append(task_vectors)
            val_y_true_list.append(task_y_true)
            val_domains_list.append(task_domains)

        val_vectors = np.concatenate(val_vectors_list, axis=0)
        val_y_true = np.concatenate(val_y_true_list, axis=0)
        val_domains = np.concatenate(val_domains_list, axis=0)
        num_val_samples = len(val_vectors)
        if np.isnan(val_vectors).any() or np.isinf(val_vectors).any():
            logging.info(f"TSNE Check: val_vectors contains NaN or Inf values after extraction.")
        else:
            logging.info(f"TSNE Check: val_vectors is clean (no NaN/Inf).")

        if (hasattr(self, 'cache_embedding_list') and self.cache_embedding_list is not None and
            hasattr(self, 'cache_label_list') and self.cache_label_list is not None and
            hasattr(self, 'cache_domain_list') and self.cache_domain_list is not None):
            train_vectors = self.cache_embedding_list.cpu().numpy() if isinstance(self.cache_embedding_list, torch.Tensor) else self.cache_embedding_list
            train_y_true = self.cache_label_list.cpu().numpy() if isinstance(self.cache_label_list, torch.Tensor) else self.cache_label_list
            train_domains = self.cache_domain_list.cpu().numpy() if isinstance(self.cache_domain_list, torch.Tensor) else self.cache_domain_list
            logging.info("TSNE: Using cached training embeddings.")
        else:
            logging.info("TSNE: Cache not available, extracting training embeddings from dataloader...")
            trainloader = self.data_manager.get_dataloader(task_id=self._cur_task, source='train', mode='test',
                                                           shuffle=False, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], pin_memory=self.args["pin_mem"], with_domain=True)
            train_vectors, train_y_true, train_domains = self._extract_vectors(trainloader, require_domain=True)
        num_train_samples = len(train_vectors)
        if np.isnan(train_vectors).any() or np.isinf(train_vectors).any():
            logging.info(f"TSNE Check: train_vectors contains NaN or Inf values after extraction.")
        else:
            logging.info(f"TSNE Check: train_vectors is clean (no NaN/Inf).")

        proto_vectors = np.array([])
        proto_y_true = np.array([])
        if hasattr(self, 'prototype_manager') and self.prototype_manager is not None:
            _proto_vectors_torch, _proto_y_true_torch = self.prototype_manager.get_vectors_for_draw()
            if len(_proto_vectors_torch) > 0:
                proto_vectors = _proto_vectors_torch.cpu().numpy()
                proto_y_true = _proto_y_true_torch.cpu().numpy()
                if np.isnan(proto_vectors).any() or np.isinf(proto_vectors).any():
                    logging.info(f"TSNE Check: proto_vectors contains NaN or Inf values from prototype_manager.")
                else:
                    logging.info(f"TSNE Check: proto_vectors is clean (no NaN/Inf).")

                print('val_vectors', len(val_vectors), val_vectors.shape, 'proto_vectors', len(proto_vectors),
                      proto_vectors.shape, 'proto class', np.unique(proto_y_true))

        current_domain_name = 'None'
        if task >= 0:
            try:
                tmp_loader_for_domain_name = self.data_manager.get_dataloader(task_id=task, source='train', mode='test', shuffle=False, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], pin_memory=self.args["pin_mem"], with_domain=True)
                first_batch = next(iter(tmp_loader_for_domain_name))
                current_domain_name = first_batch[-1][0]
            except StopIteration:
                print(f"Warning: Could not get domain name for task {task} from training loader.")

        result_root = f'results/{str(self.args["model_prefix"])}'
        if not os.path.exists(result_root):
            os.makedirs(result_root)

        pdf_path = f'{result_root}/{str(self.args["model_prefix"])}_total{tot_classes}_task{task}_tsne.pdf'
        png_dir = f'{result_root}/png_task{task}'
        os.makedirs(png_dir, exist_ok=True)

        with PdfPages(pdf_path) as pdf:
            combined_all_vectors = np.concatenate((val_vectors, train_vectors), axis=0)
            combined_all_y_true = np.concatenate((val_y_true, train_y_true), axis=0)
            combined_all_domains = np.concatenate((val_domains, train_domains), axis=0)
            num_combined_data_samples = num_val_samples + num_train_samples

            if len(proto_vectors) > 0:
                combined_all_vectors = np.concatenate((combined_all_vectors, proto_vectors), axis=0)
                combined_all_y_true = np.concatenate((combined_all_y_true, proto_y_true), axis=0)

            if Normalize:
                combined_all_vectors = combined_all_vectors / np.linalg.norm(combined_all_vectors, axis=1,
                                                                               keepdims=True)

            embedding_all: np.ndarray = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine').fit_transform(
                combined_all_vectors)


            del val_vectors, train_vectors
            torch.cuda.empty_cache()

            plt.figure(figsize=(8, 8))
            class_cmap = plt.cm.get_cmap("tab20", tot_classes)
            scatter = plt.scatter(embedding_all[:num_val_samples, 0],
                                  embedding_all[:num_val_samples, 1],
                                  c=combined_all_y_true[:num_val_samples], s=0.5, alpha=0.5,
                                  cmap=class_cmap,
                                  vmin=0, vmax=tot_classes - 1)
            if len(proto_vectors) > 0:
                plt.scatter(
                    embedding_all[num_combined_data_samples:, 0],
                    embedding_all[num_combined_data_samples:, 1],
                    c=combined_all_y_true[num_combined_data_samples:],
                    s=200,
                    marker='*',
                    cmap=class_cmap,
                    vmin=0,
                    vmax=tot_classes - 1,
                    edgecolors='black',
                    linewidths=0.8
                )
            plt.title(f"TSNE by Class (Validation Data) Task {task}")
            plt.legend(*scatter.legend_elements(), title="Class", loc="best", fontsize='small')
            pdf.savefig()
            plt.savefig(f'{png_dir}/fig1_val_by_class.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 8))
            unique_val_domains = sorted(list(set(combined_all_domains[:num_val_samples])))
            domain2idx_val = {d: i for i, d in enumerate(unique_val_domains)}
            domain_idx_val = np.array([domain2idx_val[d] for d in combined_all_domains[:num_val_samples]])
            domain_cmap_val = plt.cm.get_cmap("tab10", len(unique_val_domains))
            scatter = plt.scatter(embedding_all[:num_val_samples, 0],
                                  embedding_all[:num_val_samples, 1],
                                  c=domain_idx_val, s=0.5, alpha=0.5, cmap=domain_cmap_val)
            plt.title(f"TSNE by Domain (Validation Data) Task {task}")
            handles, _ = scatter.legend_elements()
            labels = [str(d) for d in unique_val_domains]
            plt.legend(handles, labels, title="Domain", loc="best", fontsize='small')
            plt.text(0.05, 0.95, f"Current Domain: {current_domain_name}", transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.5))
            pdf.savefig()
            plt.savefig(f'{png_dir}/fig2_val_by_domain.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 8))
            scatter = plt.scatter(embedding_all[:num_combined_data_samples, 0],
                                  embedding_all[:num_combined_data_samples, 1],
                                  c=combined_all_y_true[:num_combined_data_samples], s=0.5, alpha=0.5, cmap=class_cmap,
                                  vmin=0, vmax=tot_classes - 1)
            if len(proto_vectors) > 0:
                plt.scatter(
                    embedding_all[num_combined_data_samples:, 0],
                    embedding_all[num_combined_data_samples:, 1],
                    c=combined_all_y_true[num_combined_data_samples:],
                    s=200,
                    marker='*',
                    cmap=class_cmap,
                    vmin=0,
                    vmax=tot_classes - 1,
                    edgecolors='black',
                    linewidths=0.8
                )
            plt.title(f"TSNE by Class (Validation + Training Data) Task {task}")
            plt.legend(*scatter.legend_elements(), title="Class", loc="best", fontsize='small')
            pdf.savefig()
            plt.savefig(f'{png_dir}/fig3_combined_by_class.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 8))
            unique_combined_domains = sorted(list(set(combined_all_domains)))
            domain2idx_combined = {d: i for i, d in enumerate(unique_combined_domains)}
            domain_idx_combined = np.array([domain2idx_combined[d] for d in combined_all_domains])
            domain_cmap_combined = plt.cm.get_cmap("tab10", len(unique_combined_domains))
            scatter = plt.scatter(embedding_all[:num_combined_data_samples, 0],
                                  embedding_all[:num_combined_data_samples, 1],
                                  c=domain_idx_combined, s=0.5, alpha=0.5, cmap=domain_cmap_combined)
            plt.title(f"TSNE by Domain (Validation + Training Data) Task {task}")
            handles, _ = scatter.legend_elements()
            labels = [str(d) for d in unique_combined_domains]
            plt.legend(handles, labels, title="Domain", loc="best", fontsize='small')
            plt.text(0.05, 0.95, f"Current Domain: {current_domain_name}", transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.5))
            pdf.savefig()
            plt.savefig(f'{png_dir}/fig4_combined_by_domain.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()

        del embedding_all
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"TSNE plots saved to {pdf_path} and {png_dir}/*.png")

    def save_checkpoint(self, filename):
        original_device = next(self._network.parameters()).device

        try:
            self._network.cpu()
            save_dict = {
                "tasks": self._cur_task,
                "model_state_dict": self._network.state_dict(),
                "total_classes": self._total_classes,
                "model_type": type(self._network).__name__,
            }

            if hasattr(self, 'prototype_manager') and self.prototype_manager is not None:
                if hasattr(self.prototype_manager, 'get_state_dict'):
                    save_dict["prototype_manager_state"] = self.prototype_manager.get_state_dict()
                    logging.info("Prototype manager state saved to checkpoint")
                else:
                    save_dict["prototype_manager_state"] = {
                        'class_prototypes': self.prototype_manager.class_prototypes,
                        'class_domains': getattr(self.prototype_manager, 'class_domains', {}),
                        'class_counts': getattr(self.prototype_manager, 'class_counts', {})
                    }
                    logging.info("Prototype manager core data saved to checkpoint")

            torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))
        finally:
            self._network.to(original_device)

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        """Evaluate cumulative accuracy, reusing cached predictions when available."""
        if hasattr(self, '_cached_y_pred_list') and len(self._cached_y_pred_list) > 0:
            y_pred = np.concatenate(self._cached_y_pred_list, axis=0)
            y_true = np.concatenate(self._cached_y_true_list, axis=0)
        else:
            logging.warning('No cached y_pred_list and y_true_list')
            y_pred_list, y_true_list = [], []
            for task_id in range(self._cur_task + 1):
                test_loader = self.data_manager.get_dataloader(task_id=task_id, source='cur_test', mode='test', shuffle=False, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], pin_memory=self.args["pin_mem"], with_domain=False)
                task_y_pred, task_y_true = self._eval_cnn(test_loader)
                y_pred_list.append(task_y_pred)
                y_true_list.append(task_y_true)
            y_pred = np.concatenate(y_pred_list, axis=0)
            y_true = np.concatenate(y_true_list, axis=0)

        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, '_cached_y_pred_list'):
            del self._cached_y_pred_list
            del self._cached_y_true_list

        return cnn_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, batch in enumerate(loader):
            _, inputs, targets = batch[:3]
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, batch in enumerate(loader):
                _, inputs, targets = batch[:3]
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    outputs = self._network(inputs)["logits"]
                predicts = torch.topk(
                    outputs, k=self.topk, dim=1, largest=True, sorted=True
                )[
                    1
                ]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  
        scores = dists.T

        return np.argsort(scores, axis=1)[:, : self.topk], y_true

    def _extract_vectors(self, loader, require_domain=False):
        self._network.eval()
        self._network = self._network.to(self._device)
        if require_domain:
            with torch.no_grad():
                vectors, targets, domains = [], [], []
                prog_bar = tqdm(enumerate(loader), total=len(loader))
                for _, batch in prog_bar:
                    _, _inputs, _targets, _domains = batch
                    try:
                        _targets = _targets.numpy()
                    except Exception:
                        print(Exception)
                        print(type(_targets), print(len(_targets)), print(_targets))
                    if isinstance(self._network, nn.DataParallel):
                        _vectors = tensor2numpy(
                            self._network.module.extract_vector(_inputs.to(self._device))
                        )
                    else:
                        _vectors = tensor2numpy(
                            self._network.extract_vector(_inputs.to(self._device))
                        )

                    vectors.append(_vectors)
                    targets.append(_targets)
                    domains.append(_domains)
                    prog_bar.set_description('Extracting features for plot(eval)')
        else:
            vectors, targets = [], []
            prog_bar = tqdm(enumerate(loader), total=len(loader))
            with torch.no_grad():
                for _, batch in prog_bar:
                    _, _inputs, _targets = batch[:3]
                    try:
                        _targets = _targets.numpy()
                    except Exception:
                        print(Exception)
                        print(type(_targets), print(len(_targets)), print(_targets))
                    if isinstance(self._network, nn.DataParallel):
                        _vectors = tensor2numpy(
                            self._network.module.extract_vector(_inputs.to(self._device))
                        )
                    else:
                        _vectors = tensor2numpy(
                            self._network.extract_vector(_inputs.to(self._device))
                        )

                    vectors.append(_vectors)
                    targets.append(_targets)
                    prog_bar.set_description('Extracting features for plot(eval)')
        if require_domain:
            return np.concatenate(vectors), np.concatenate(targets), np.concatenate(domains)
        else:
            return np.concatenate(vectors), np.concatenate(targets)

    def eval_forgetting(self, current_task, acc_matrix=None):
        """Compute final-stage forgetting from the task accuracy matrix."""
        if acc_matrix is None:
            if not hasattr(self, 'acc_matrix'):
                logging.warning("No acc_matrix found, cannot calculate forgetting")
                return None
            acc_matrix = self.acc_matrix

        if current_task == 0:
            return {
                'bwt_per_task': np.array([]),
                'avg_forgetting': 0.0,
                'max_forgetting': 0.0,
                'backward_transfer': 0.0
            }

        bwt_per_task = []

        for i in range(current_task):
            bwt_sum = 0.0
            count = 0

            for j in range(i + 1, current_task + 1):
                bwt_sum += (acc_matrix[i, j] - acc_matrix[i, i])
                count += 1

            if count > 0:
                bwt_i = bwt_sum / count
                bwt_per_task.append(bwt_i)
            else:
                bwt_per_task.append(0.0)

        avg_forgetting = np.mean(bwt_per_task) if bwt_per_task else 0.0

        max_forgetting = np.mean([np.max(acc_matrix[i, :current_task+1]) - acc_matrix[i, current_task]
                                for i in range(current_task)])

        diagonal = np.array([acc_matrix[i, i] for i in range(current_task)])

        backward_from_init = np.mean([acc_matrix[i, current_task] - acc_matrix[i, i]
                                     for i in range(current_task)])

        best_acc = np.max(acc_matrix[:current_task, :current_task+1], axis=1)
        backward_from_best = np.mean(acc_matrix[:current_task, current_task] - best_acc)

        forgetting = np.mean(best_acc - acc_matrix[:current_task, current_task])

        upper_vals = [acc_matrix[i,j] for i in range(len(acc_matrix)) for j in range(len(acc_matrix)) if i<j]
        mean_val = np.mean(upper_vals)
        print(f'均值（AA）为 {mean_val}')
        logging.info(f'均值（AA）为 {mean_val}')
        return {
            'bwt_per_task': np.array(bwt_per_task),
            'avg_forgetting': avg_forgetting,
            'max_forgetting': max_forgetting,
            'forgetting': forgetting,
            'backward_transfer': backward_from_init,
            'backward_from_init': backward_from_init,
            'backward_from_best': backward_from_best,
            'AA': mean_val
        }

    def evaluate_till_now(self, current_task):
        """Evaluate seen tasks and update the accuracy-matrix column."""
        if self.acc_matrix is None:
            self.acc_matrix = np.zeros((self.data_manager.nb_tasks, self.data_manager.nb_tasks))

        self._cached_y_pred_list = []
        self._cached_y_true_list = []

        for task_id in range(current_task + 1):
            test_loader = self.data_manager.get_dataloader(task_id=task_id, source='cur_test', mode='test', shuffle=False, batch_size=self.args.get("test_batch_size", 256), num_workers=self.args["num_workers"], pin_memory=self.args["pin_mem"], with_domain=False)
            tmp_y_pred, tmp_y_true = self._eval_cnn(test_loader)
            tmp_cnn_accy = self._evaluate(tmp_y_pred, tmp_y_true)

            self.acc_matrix[task_id, current_task] = tmp_cnn_accy['top1']

            self._cached_y_pred_list.append(tmp_y_pred)
            self._cached_y_true_list.append(tmp_y_true)

        forgetting_metrics = self.eval_forgetting(current_task, self.acc_matrix)

        avg_acc = np.mean(self.acc_matrix[:current_task+1, current_task])
        current_task_acc = self.acc_matrix[current_task, current_task]

        if hasattr(self, 'args'):
            checkpoint_dir = "checkpoints/{}/{}/{}".format(
                self.args["model_name"], 
                self.args["dataset"], 
                self.args["data_seed"]
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            model_prefix = self.args.get("model_prefix", "{}_{}".format(self.args["prefix"], self.args["convnet_type"]))
            acc_matrix_filename = "{}/{}_task{}_seed{}_acc_matrix.npy".format(
                checkpoint_dir,
                model_prefix,
                current_task,
                self.args["seed"]
            )
            np.save(acc_matrix_filename, self.acc_matrix)

            txt_filename = acc_matrix_filename.replace('.npy', '.txt')
            with open(txt_filename, 'w') as f:
                f.write(f"# ACC Matrix for Task {current_task}\n")
                f.write(f"# Rows: test tasks, Columns: training stages\n")
                f.write(f"# Shape: {self.acc_matrix.shape}\n")
                f.write(f"# Average Accuracy: {avg_acc:.4f}\n")
                f.write(f"# Current Task Acc: {current_task_acc:.4f}\n")
                if current_task > 0 and forgetting_metrics:
                    f.write(f"# Forgetting: {forgetting_metrics['forgetting']:.4f}\n")
                    f.write(f"# Backward Transfer (from init): {forgetting_metrics['backward_from_init']:.4f}\n")
                    f.write(f"# Backward Transfer (from best): {forgetting_metrics['backward_from_best']:.4f}\n")
                    f.write(f"# Average Forgetting (AF): {forgetting_metrics['avg_forgetting']:.4f}\n")
                    f.write(f"# Max Forgetting: {forgetting_metrics['max_forgetting']:.4f}\n")
                f.write("\n# ACC Matrix:\n")
                np.savetxt(f, self.acc_matrix, fmt='%.4f', delimiter='\t')

            logging.info(f"ACC_matrix saved to: {acc_matrix_filename} and {txt_filename}")

        result_str = f"Learned [Average accuracy till task{current_task+1}]\tAcc@1: {avg_acc:.4f}\tCurrent Task Acc: {current_task_acc:.4f}"

        if current_task > 0:
            result_str += f"\tForgetting: {forgetting_metrics['forgetting']:.4f}"
            result_str += f"\tBWT(init): {forgetting_metrics['backward_from_init']:.4f}"
            result_str += f"\tBWT(best): {forgetting_metrics['backward_from_best']:.4f}"
            result_str += f"\tAF: {forgetting_metrics['avg_forgetting']:.4f}"

        logging.info(result_str)

        return {
            'acc_matrix': self.acc_matrix,
            'avg_accuracy': avg_acc,
            'current_task_accuracy': current_task_acc,
            'forgetting_metrics': forgetting_metrics
        }
