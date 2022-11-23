import math
import os
import random
import sys
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, recall_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

# cur_dir = os.path.dirname(__file__)
# sys.path.insert(0, '/'.join(cur_dir.split('/')[:-1]))
sys.path.append("./")
import json
import logging
import time

from BREM.common.config import config
from BREM.common.thumos_dataset import (THUMOS_Dataset,
                                        detection_collate)
from BREM.thumos14.BaselineModel import BaseModel
from BREM.thumos14.BEM import BEMModule
from apex import amp
from torch.utils.data import DataLoader

time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + "_" + config.output
config["training"]["checkpoint_path"] = os.path.join(
    config["training"]["checkpoint_path"], time_str
)
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
backbone_lr = learning_rate * config["bb_rate"]
weight_decay = config["training"]["weight_decay"]
max_epoch = config["training"]["max_epoch"]
num_classes = config["dataset"]["num_classes"]
checkpoint_path = config["training"]["checkpoint_path"]
focal_loss = config["training"]["focal_loss"]
random_seed = config["training"]["random_seed"]

train_state_path = os.path.join(checkpoint_path, "training")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config["training"]["resume"]
config["training"]["ssl"] = 0.1


def get_logger(log_path="./log/"):
    # time_str = time.strftime("%Y-%m-%d|%H:%M:%S", time.localtime())  # 2016-03-20|11:45:39
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", filename=log_path
    )
    logger = logging.getLogger()
    return logger


log_path = "./log/log_" + time_str + ".txt"
logger = get_logger(log_path)
"""
Distributed training
"""
world_size, local_rank, rank = None, None, None
if config["dist"]:
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    local_rank = config["local_rank"]
    rank = dist.get_rank()
    print("world size: ", world_size, "\trank: ", rank)


def print_training_info():
    print("batch size: ", batch_size)
    print("learning rate: ", learning_rate)
    print("backbone learning rate: ", backbone_lr)
    print("weight decay: ", weight_decay)
    print("max epoch: ", max_epoch)
    print("checkpoint path: ", checkpoint_path)
    print("loc weight: ", config["training"]["lw"])
    print("cls weight: ", config["training"]["cw"])
    print("iou weight: ", config["training"]["piou"])
    print("ssl weight: ", config["training"]["ssl"])
    print("piou:", config["training"]["piou"])
    print("resume: ", resume)
    print("Num of GPUs: {}".format(world_size if config["dist"] else 1))
    config_info = json.dumps(config, indent=4, sort_keys=True)
    print(config_info)
    logger.info(config_info)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


GLOBAL_SEED = 1


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def save_model(epoch, model, optimizer):
    torch.save(
        model.module.state_dict(),
        os.path.join(checkpoint_path, "checkpoint-{}.ckpt".format(epoch)),
    )
    torch.save(
        {"optimizer": optimizer.state_dict(), "state": get_rng_states()},
        os.path.join(train_state_path, "checkpoint_{}.ckpt".format(epoch)),
    )

def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(
            checkpoint_path, "checkpoint-{}.ckpt".format(resume)
        )
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(
            train_state_path, "checkpoint_{}.ckpt".format(resume)
        )
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict["optimizer"])
        set_rng_state(state_dict["state"])
    return start_epoch


def forward_one_epoch(
    net, clips, targets, training=True, epoch=None
) -> Dict[str, torch.Tensor]:
    """
    Returns
    ---
        loss_l: iou loss
        loss_c: focal loss
        loss_prop_l: L1 loss
        loss_prop_c: focal loss
        loss_ct: BCE loss, quality, (center-ness) 
        loss_start, loss_end: BCE loss, Activation guided Learning, l_{act} in the paper.
    """
    clips = clips.cuda()  # for RGB, [1, 3, 256, 96, 96], 3 RGE, 256 frames
    targets = [t.cuda() for t in targets]

    if training:
        loss_dict = net(clips, targets, return_loss=True)
    else:
        with torch.no_grad():
            loss_dict = net(clips, targets, return_loss=True)

    return loss_dict


def run_one_epoch(
    epoch,
    net,
    optimizer,
    data_loader,
    epoch_step_num,
    training=True,
    warm_scheduler=None,
):
    if training:
        net.train()
    else:
        net.eval()

    loss_val = OrderedDict()
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets) in enumerate(pbar):
            """
            Flow: first dim is batch_size for Tensor, length is batch_size for numpy array.
                clips: [1, 2, 256, 96, 96]
                targets: list, [1, 3], 3:(start, end, class_label)
                scores: [1, 2, 256], 2-dim is start and end, respectively.
                ssl_clips: [1, 2, 256, 96, 96]
                ssl_targets: list, [3, 2], used to partition action and background.
            """
            if warm_scheduler is not None:
                warm_scheduler.step()
            loss_dict = forward_one_epoch(
                net, clips, targets, training=training, epoch=epoch,
            )

            cost = loss_dict["loss"]

            if training:
                optimizer.zero_grad()
                if config["fp16"]:
                    with amp.scale_loss(cost, optimizer) as scaled_cost:
                        scaled_cost.backward()
                else:
                    cost.backward()
                optimizer.step()

            for key in loss_dict.keys():
                if key not in loss_val:
                    loss_val[key] = loss_dict[key].cpu().detach().numpy()
                else:
                    loss_val[key] += loss_dict[key].cpu().detach().numpy()

            pbar.set_postfix(
                dict(loss="{:.5f}".format(float(cost.cpu().detach().numpy())))
            )

    for key_val, loss in loss_val.items():
        loss_val[key_val] = loss_val[key_val] / (n_iter + 1)

    if training:
        prefix = "Train"
        if not config["dist"] or dist.get_rank() == 0:
            save_model(epoch, net, optimizer)
    else:
        prefix = "Val"

    loss_log = ""
    for key_val, loss in loss_val.items():
        loss_log += key_val + " - {:.5f}, ".format(loss)

    plog = "Epoch-{} {} ".format(epoch, prefix,) + loss_log
    print(plog)
    logger.info(plog + "\n")


def devide_param(net):
    backbone_param = list(net.backbone.parameters())
    backbone_param_id = list(map(id, backbone_param))
    rest_param = filter(lambda x: id(x) not in backbone_param_id, net.parameters())
    return backbone_param, rest_param


if __name__ == "__main__":
    if not config["dist"] or dist.get_rank() == 0:
        print_training_info()
    set_seed(random_seed)
    """
    Setup model
    """
    net = globals()[config.model.model_type](config)
    """
    Setup loss
    """
    piou = config["training"]["piou"]
    # CPD_Loss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)
    """
    Setup dataloader
    """
    train_dataset = THUMOS_Dataset()
    """
    Setup model
    """
    use_distributed = config["dist"]
    if use_distributed:
        torch.cuda.set_device(local_rank)
        train_sampler = DistributedSampler(train_dataset)
        learning_rate = learning_rate * world_size
        backbone_lr = learning_rate * config["bb_rate"]
        config["training"]["learning_rate"] = learning_rate
        print(
            "Learning rate: {}, backbone learning rate: {}, because of distributed training.".format(
                learning_rate, backbone_lr
            )
        )
        print("Distributed training")
    else:
        train_sampler = None

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(not use_distributed),
        num_workers=30,
        worker_init_fn=worker_init_fn,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    if use_distributed:
        epoch_step_num = math.ceil(len(train_dataset) / batch_size / world_size)
    else:
        epoch_step_num = math.ceil(len(train_dataset) / batch_size)
    net.cuda()
    if use_distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    """
    Setup optimizer
    """
    backbone_param, rest_param = devide_param(net)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_param, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": rest_param, "lr": learning_rate, "weight_decay": weight_decay},
        ]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    """
    fp16
    """
    if config["fp16"]:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

    if use_distributed:
        net = DistributedDataParallel(
            net,
            device_ids=[local_rank],
            broadcast_buffers=True,
            output_device=local_rank,
        )
    else:
        net = nn.DataParallel(net, device_ids=[0])
    """
    Start training
    """
    start_epoch = resume_training(resume, net, optimizer)

    for i in range(start_epoch, max_epoch + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(i)
        run_one_epoch(
            i,
            net,
            optimizer,
            train_data_loader,
            epoch_step_num
        )
        scheduler.step()
"""
testing
"""
if not config["dist"] or dist.get_rank() == 0:
    from Baseline_BREM.BREM.thumos14.test import test_pai

    del net
    del optimizer
    del train_dataset
    del train_data_loader
    log_file_name = time_str

    test_pai(log_file_name)
