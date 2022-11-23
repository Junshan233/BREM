import math
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm

sys.path.append('./')
import json
import logging
import time

from BREM.anet.BaselineModel import BaseModel
from BREM.anet.BEM import BEMModule
from BREM.common.anet_dataset import ANET_Dataset, detection_collate
from BREM.common.config import config
from apex import amp
from torch.utils.data import DataLoader

time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + "_" + config.output
config['training']['checkpoint_path'] = os.path.join(config['training']['checkpoint_path'], time_str)
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
backbone_lr = learning_rate * config["bb_rate"]
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = 2
checkpoint_path = config['training']['checkpoint_path']
focal_loss = config['training']['focal_loss']
random_seed = config['training']['random_seed']
ngpu = config['ngpu']

if config.dist:
    raise ValueError("Not support distributed training.")
else:
    print('Not using distributed mode')
    config.distributed = False

train_state_path = os.path.join(checkpoint_path, 'training')
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config['training']['resume']
config['training']['ssl'] = 0.1

def get_logger(log_path = './log/'):
    # time_str = time.strftime("%Y-%m-%d|%H:%M:%S", time.localtime())  # 2016-03-20|11:45:39
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(message)s', filename=log_path)
    logger = logging.getLogger()
    return logger

log_path = "./log/log_" + time_str + '.txt'
logger = get_logger(log_path)


def print_training_info():
    command = "command: " + " ".join(sys.argv)
    print(command)
    logger.info(command)
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('piou: ', config['training']['piou'])
    print('resume: ', resume)
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
    torch.save(model.module.state_dict(),
               os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(epoch)))

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


def forward_one_epoch(net, clips, targets, training=True):
    clips = clips.cuda()
    targets = [t.cuda() for t in targets]

    if training:
        loss_dict = net(clips, targets, return_loss=True)
    else:
        with torch.no_grad():
            loss_dict = net(clips, targets, return_loss=True)

    return loss_dict


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True, warm_scheduler=None,):
    if training:
        net.train()
    else:
        net.eval()

    loss_val = OrderedDict()
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets) in enumerate(pbar):

            if warm_scheduler is not None:
                warm_scheduler.step()
            loss_dict = forward_one_epoch(
                net, clips, targets, training=training,
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
        prefix = 'Train'
        save_model(epoch, net, optimizer)
    else:
        prefix = 'Val'

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


if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)
    """
    Setup model
    """
    net = globals()[config.model.model_type](config)
    net = nn.DataParallel(net, device_ids=list(range(ngpu))).cuda()

    """
    Setup optimizer
    """
    backbone_param, rest_param = devide_param(net.module)
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_param, "lr": learning_rate * 0.1, "weight_decay": weight_decay},
            {"params": rest_param, "lr": learning_rate, "weight_decay": weight_decay},
        ]
    )

    """
    Setup dataloader
    """
    train_dataset = ANET_Dataset(config,
                                 config['dataset']['training']['video_info_path'],
                                 config['dataset']['training']['video_mp4_path'],
                                 config['dataset']['training']['clip_length'],
                                 config['dataset']['training']['crop_size'],
                                 config['dataset']['training']['clip_stride'],
                                 channels=config['model']['in_channels'],
                                 binary_class=True)

    """
    Setup model
    """
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler,
                                                        batch_size,
                                                        drop_last=True)
    
    train_data_loader = DataLoader(train_dataset,
                                   batch_sampler=batch_sampler_train,
                                   num_workers=56, worker_init_fn=worker_init_fn,
                                   collate_fn=detection_collate, pin_memory=True)
    epoch_step_num = math.ceil(len(train_dataset) / batch_size)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.training.lr_step_size, gamma=0.1
    )
    """
    fp16
    """
    if config['fp16']:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    """
    Start training
    """
    start_epoch = resume_training(resume, net, optimizer)

    for i in range(start_epoch, max_epoch + 1):
        run_one_epoch(i, net, optimizer, train_data_loader, epoch_step_num)
        scheduler.step()

"""
testing
"""
from Baseline_BREM.BREM.anet.test import test_func

del net
del optimizer
del train_dataset
del train_data_loader
log_file_name = time_str
epoch_list = [5, 6, 7, 8, 9, 10, 11, 12]

test_func(config, log_file_name, epoch_list=epoch_list, thread_num=1)

# python3 BREM/anet/train.py configs/anet.yaml --batch_size 1 --learning_rate 1e-4
