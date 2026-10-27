import sys
import logging
import copy
import time
from datetime import datetime

import torch
from my_utils import factory
from my_utils.data_manager import DataManager
from my_utils.inc_net import MultiBranchCosineIncrementalNet_v5
from my_utils.toolkit import count_parameters
import os
import numpy as np
from my_dataset_utils.my_datasets import build_continual_dataloader


def load_checkpoint(checkpoint_path, args, device):
    """Load a model and task index from a checkpoint."""
    checkpoint_data = torch.load(checkpoint_path, map_location=device)

    if 'model_checkpoint' in checkpoint_data:
        checkpoint = checkpoint_data['model_checkpoint']
        loaded_task = checkpoint_data['task_id']
        total_classes = checkpoint_data['total_classes']
        model_type = checkpoint_data['model_type']
    else:
        checkpoint = checkpoint_data
        loaded_task = checkpoint["tasks"]
        total_classes = checkpoint.get("total_classes", None)
        model_type = checkpoint.get("model_type", "SimpleVit")

    model = factory.get_model(args["model_name"], args)
    network = MultiBranchCosineIncrementalNet_v5(model.args, True) # type: ignore
    network.construct_dual_branch_network(model._network) # type: ignore
    model._network = network.to(device) # type: ignore

    if hasattr(model, 'prototype_manager') and model.prototype_manager is not None: # type: ignore
        model._network.set_prototype_manager(model.prototype_manager) # type: ignore

    if total_classes is not None:
        model._total_classes = total_classes # type: ignore
        model._network.update_fc(total_classes) # type: ignore
        logging.info(f"Updated model fc to {total_classes} classes based on checkpoint")
    else:
        if 'fc.weight' in checkpoint["model_state_dict"]:
            fc_weight_shape = checkpoint["model_state_dict"]['fc.weight'].shape
            inferred_classes = fc_weight_shape[0]
            model._total_classes = inferred_classes
            model._network.update_fc(inferred_classes)
            logging.info(f"Inferred {inferred_classes} classes from fc.weight shape and updated model")

    try:
        model._network.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logging.info("Model state loaded successfully (with strict=False)")
    except Exception as e:
        logging.warning(f"Failed to load model state with strict=False: {e}")
        model_state = model._network.state_dict()
        checkpoint_state = checkpoint["model_state_dict"]

        for key in model_state.keys():
            if key in checkpoint_state:
                if model_state[key].shape == checkpoint_state[key].shape:
                    model_state[key] = checkpoint_state[key]
                else:
                    logging.warning(f"Shape mismatch for {key}: model {model_state[key].shape} vs checkpoint {checkpoint_state[key].shape}")

        model._network.load_state_dict(model_state)
        logging.info("Model state loaded with manual parameter matching")

    model._cur_task = loaded_task

    if "prototype_manager_state" in checkpoint and hasattr(model, 'prototype_manager') and model.prototype_manager is not None: # type: ignore
        if hasattr(model.prototype_manager, 'load_state_dict'): # type: ignore
            model.prototype_manager.load_state_dict(checkpoint["prototype_manager_state"], device) # type: ignore
            logging.info("Prototype manager state loaded from checkpoint")
        else:
            proto_state = checkpoint["prototype_manager_state"]
            model.prototype_manager.class_prototypes = proto_state.get('class_prototypes', {}) # type: ignore
            if hasattr(model.prototype_manager, 'class_domains'): # type: ignore
                model.prototype_manager.class_domains = proto_state.get('class_domains', {}) # type: ignore
            if hasattr(model.prototype_manager, 'class_counts'): # type: ignore
                model.prototype_manager.class_counts = proto_state.get('class_counts', {}) # type: ignore
            logging.info("Prototype manager core data loaded from checkpoint")
    for class_index in model.prototype_manager.class_prototypes.keys(): # type: ignore
        model._network.fc.weight.data[class_index] = model.prototype_manager.class_prototypes[class_index][0] # type: ignore
    logging.info("Checkpoint loaded successfully from: {}".format(checkpoint_path))
    logging.info("Loaded model from task: {}, total_classes: {}, model_type: {}".format(loaded_task, total_classes, model_type))
    return loaded_task, model

def get_checkpoint_path(args, task_id):
    """Return the checkpoint path for a task."""
    checkpoint_dir = "checkpoints/{}/{}/{}/{}/".format(args["model_name"], args["dataset"], args["data_seed"], args["model_prefix"])
    checkpoint_filename = "{}/task{}_seed{}.pkl".format(checkpoint_dir, task_id, args["seed"])
    return checkpoint_filename


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    now = datetime.now()
    time_str = now.strftime("%Y-%m%d-%H:%M:%S")
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], args["data_seed"], time_str)

    if not os.path.exists(logs_name):
        os.makedirs(logs_name, exist_ok=True)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        args["data_seed"],
        time_str,
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info('Current pid is: {}'.format(os.getpid()))
    print_args(args)
    Whole_dataset_train, Whole_dataset_test = build_continual_dataloader(args)
    data_manager = DataManager(Whole_dataset_train, Whole_dataset_test, args=args)

    init_cls = len(data_manager.class_mask[0])
    args["init_cls"] = init_cls
    _set_random(args["seed"])
    _set_device(args)

    model = factory.get_model(args["model_name"], args)
    cnn_curve = {"top1": []}

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        data_manager.task_id = task # type: ignore
        model.incremental_train(data_manager)

        model.evaluate_till_now(task)

        cnn_accy = model.eval_task()

        model.after_task()

        if args["plot"] and (task == data_manager.nb_tasks - 1): 
            model.tsne(True, task)

        if args.get("save_checkpoint", True) and (task == data_manager.nb_tasks - 1):
            checkpoint_dir = "checkpoints/{}/{}/{}/{}/".format(args["model_name"], args["dataset"], args["data_seed"], args["model_prefix"])
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_filename = "{}/task{}_seed{}".format(
                checkpoint_dir,
                task,
                args["seed"]
            )

            cache_attrs = ['cache_embedding_list', 'cache_label_list', 'cache_domain_list', 'cache_image_list']
            for attr in cache_attrs:
                if hasattr(model, attr):
                    logging.info(f"Clearing cache: {attr}")
                    delattr(model, attr)

            model.save_checkpoint(checkpoint_filename)
            logging.info("Model checkpoint saved: {}/task{}.pkl".format(checkpoint_filename, task))

        logging.info("No NME accuracy.")
        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))

        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
