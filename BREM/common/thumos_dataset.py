import copy
import io
import math
import multiprocessing as mp
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import tqdm
from BREM.common import videotransforms
from BREM.common.config import config
from torch.utils.data import DataLoader, Dataset


def get_class_index_map(class_info_path="thumos_annotations/Class Index_Detection.txt"):
    txt = np.loadtxt(class_info_path, dtype=str)
    originidx_to_idx = {}
    idx_to_class = {}
    for idx, l in enumerate(txt):
        originidx_to_idx[int(l[0])] = idx + 1
        idx_to_class[idx + 1] = l[1]
    return originidx_to_idx, idx_to_class


def get_video_info(video_info_path):
    df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
    video_infos = {}
    for info in df_info:
        video_infos[info[0]] = {
            "fps": info[1],
            "sample_fps": info[2],
            "count": info[3],
            "sample_count": info[4],
        }
    return video_infos


def get_video_anno(video_infos, video_anno_path):
    df_anno = pd.DataFrame(pd.read_csv(video_anno_path)).values[:]
    originidx_to_idx, idx_to_class = get_class_index_map()
    video_annos = {}
    for anno in df_anno:
        video_name = anno[0]
        originidx = anno[2]
        start_frame = anno[-2]
        end_frame = anno[-1]
        count = video_infos[video_name]["count"]
        sample_count = video_infos[video_name]["sample_count"]
        ratio = sample_count * 1.0 / count
        start_gt = start_frame * ratio
        end_gt = end_frame * ratio
        class_idx = originidx_to_idx[originidx]
        if video_annos.get(video_name) is None:
            video_annos[video_name] = [[start_gt, end_gt, class_idx]]
        else:
            video_annos[video_name].append([start_gt, end_gt, class_idx])
    return video_annos


def annos_transform(annos, clip_length):
    res = []
    for anno in annos:
        res.append([anno[0] * 1.0 / clip_length, anno[1] * 1.0 / clip_length, anno[2]])
    return res


def split_videos(
    video_infos,
    video_annos,
    clip_length=config["dataset"]["training"]["clip_length"],
    stride=config["dataset"]["training"]["clip_stride"],
):
    # video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    # video_annos = get_video_anno(video_infos,
    #                              config['dataset']['training']['video_anno_path'])
    training_list = []
    for video_name in video_annos.keys():
        sample_count = video_infos[video_name]["sample_count"]
        annos = video_annos[video_name]  # [[start_gt, end_gt, class_idx]]
        if sample_count <= clip_length:
            offsetlist = [0]
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]
        for offset in offsetlist:
            left, right = offset + 1, offset + clip_length  # 闭区间
            cur_annos = []
            save_offset = False
            for anno in annos:
                max_l = max(left, anno[0])
                min_r = min(right, anno[1])
                ioa = (min_r - max_l) * 1.0 / (anno[1] - anno[0])
                if ioa >= 1.0:
                    save_offset = True
                if ioa > 0:
                    cur_annos.append(
                        [
                            anno[0] - offset,
                            anno[1] - offset,
                            anno[2],
                        ]
                    )
            if save_offset:
                training_list.append(
                    {
                        "video_name": video_name,
                        "offset": offset,
                        "annos": cur_annos,
                    }
                )
    return training_list


def load_video_data(video_infos, npy_data_path):
    data_dict = {}
    print("loading video frame data ...")
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
        data = np.load(os.path.join(npy_data_path, video_name + ".npy"))
        data = np.transpose(data, [3, 0, 1, 2])
        data_dict[video_name] = data
    return data_dict


class THUMOS_Dataset(Dataset):
    def __init__(
        self,
        clip_length=config["dataset"]["training"]["clip_length"],
        crop_size=config["dataset"]["training"]["crop_size"],
        stride=config["dataset"]["training"]["clip_stride"],
        npy_data_path=config["dataset"]["training"]["video_data_path"],
        num_class=config["dataset"]["num_classes"],
        video_info_path=config["dataset"]["training"]["video_info_path"],
        video_anno_path=config["dataset"]["training"]["video_anno_path"],
        rgb_norm=True,
        training=True,
    ):
        video_infos = get_video_info(video_info_path)
        video_annos = get_video_anno(video_infos, video_anno_path)
        self.training_list = split_videos(
            video_infos, video_annos, clip_length, stride
        )  # self.th is a dict whith saves min video clips length of every video.
        # np.random.shuffle(self.training_list)
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.random_crop = videotransforms.RandomCrop(crop_size)
        self.random_flip = videotransforms.RandomHorizontalFlip(p=0.5)
        self.center_crop = videotransforms.CenterCrop(crop_size)
        self.rgb_norm = rgb_norm
        self.training = training
        self.num_class = num_class
        self.npy_data_path = npy_data_path
        self.data_dict = load_video_data(
            video_infos,
            npy_data_path
        )

    def __len__(self):
        return len(self.training_list)

    def _get_video_data(self, video_name, offset):
        if self.data_dict is None:
            video_data = np.load(
                os.path.join(self.npy_data_path, video_name + ".npy")
            )
            video_data = np.transpose(video_data, [3, 0, 1, 2])
        else:
            video_data = self.data_dict[video_name]
        input_data = video_data[:, offset : offset + self.clip_length]
        c, t, h, w = input_data.shape
        if t < self.clip_length:
            # padding t to clip_length
            pad_t = self.clip_length - t
            zero_clip = np.zeros([c, pad_t, h, w], input_data.dtype)
            input_data = np.concatenate([input_data, zero_clip], 1)
        return input_data

    def __getitem__(self, idx):
        sample_info = self.training_list[idx]
        offset = sample_info["offset"]
        annos = sample_info["annos"]
        input_data = self._get_video_data(sample_info["video_name"], offset)

        # random crop and flip
        if self.training:
            input_data = self.random_flip(self.random_crop(input_data))
        else:
            input_data = self.center_crop(input_data)

        # import pdb;pdb.set_trace()
        input_data = torch.from_numpy(input_data).float()
        if self.rgb_norm:
            input_data = (input_data / 255.0) * 2.0 - 1.0
        annos = annos_transform(annos, self.clip_length)
        target = np.stack(annos, 0)

        return (
            input_data,
            target,
        )


def detection_collate(batch):
    targets = []
    clips = []

    for sample in batch:
        clips.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return (
        torch.stack(clips, 0),
        targets,
    )
