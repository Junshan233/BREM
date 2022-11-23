import io
import json
import math
import os
import pickle
import random

import decord
import numpy as np
import torch
from BREM.common import videotransforms
from torch.utils.data import Dataset


def load_json(file):
    """
    :param file: json file path
    :return: data of json
    """
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def video2npy(file):
    """
    file: file-like object
    """
    container = decord.VideoReader(file)
    imgs = []
    for i in range(len(container)):
        imgs.append(container[i].asnumpy())
    imgs = np.stack(imgs)
    imgs = imgs[:768]
    return imgs


def annos_transform(annos, clip_length):
    res = []
    for anno in annos:
        res.append([anno[0] * 1.0 / clip_length, anno[1] * 1.0 / clip_length, anno[2]])
    return res


def get_video_info(video_info_path, subset="training"):
    json_data = load_json(video_info_path)
    video_info = {}
    video_list = list(json_data.keys())
    for video_name in video_list:
        tmp = json_data[video_name]
        if tmp["subset"] == subset:
            video_info[video_name] = tmp
    return video_info


def split_videos(video_info, clip_length, stride, binary_class=False):
    training_list = []
    for video_name in list(video_info.keys())[:]:
        frame_num = min(video_info[video_name]["frame_num"], clip_length)
        annos = []
        for anno in video_info[video_name]["annotations"]:
            if binary_class:
                anno["label_id"] = 1 if anno["label_id"] > 0 else 0
            if anno["end_frame"] <= anno["start_frame"]:
                continue
            annos.append([anno["start_frame"], anno["end_frame"], anno["label_id"]])
        if len(annos) == 0:
            continue

        offsetlist = [0]

        for offset in offsetlist:
            cur_annos = []
            save_offset = True
            for anno in annos:
                cur_annos.append([anno[0], anno[1], anno[2]])
            if save_offset:
                start = np.zeros([clip_length])
                end = np.zeros([clip_length])
                action = np.zeros([clip_length])
                for anno in cur_annos:
                    s, e, id = anno
                    d = max((e - s) / 10.0, 2.0)
                    act_s = np.clip(int(round(s)), 0, clip_length - 1)
                    act_e = np.clip(int(round(e)), 0, clip_length - 1) + 1
                    action[act_s:act_e] = id
                    start_s = np.clip(int(round(s - d / 2)), 0, clip_length - 1)
                    start_e = np.clip(int(round(s + d / 2)), 0, clip_length - 1) + 1
                    start[start_s:start_e] = id
                    end_s = np.clip(int(round(e - d / 2)), 0, clip_length - 1)
                    end_e = np.clip(int(round(e + d / 2)), 0, clip_length - 1) + 1
                    end[end_s:end_e] = id

                training_list.append(
                    {
                        "video_name": video_name,
                        "offset": offset,
                        "annos": cur_annos,
                        "frame_num": frame_num,
                        "start": start,
                        "end": end,
                        "action": action,
                    }
                )
    return training_list


def detection_collate(batch):
    targets = []
    clips = []

    for sample in batch:
        clips.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    return torch.stack(clips, 0), targets


class ANET_Dataset(Dataset):
    def __init__(
        self,
        cfg,
        video_info_path,
        video_dir,
        clip_length,
        crop_size,
        stride,
        channels=3,
        rgb_norm=True,
        training=True,
        binary_class=False,
    ):
        self.training = training
        subset = "training" if training else "validation"
        video_info = get_video_info(video_info_path, subset)
        self.training_list = split_videos(video_info, clip_length, stride, binary_class)
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.rgb_norm = rgb_norm
        self.video_dir = video_dir
        self.channels = channels
        self.cfg = cfg

        self.random_crop = videotransforms.RandomCrop(crop_size)
        self.random_flip = videotransforms.RandomHorizontalFlip(p=0.5)
        self.center_crop = videotransforms.CenterCrop(crop_size)

    def __len__(self):
        return len(self.training_list)

    def __getitem__(self, idx):
        sample_info = self.training_list[idx]
        video_name = sample_info["video_name"]
        offset = sample_info["offset"]
        annos = sample_info["annos"]
        frame_num = sample_info["frame_num"]
        data = np.load(os.path.join(self.video_dir, video_name + ".npy"))
        start = offset
        end = min(offset + self.clip_length, frame_num)
        frames = data[start:end]
        frames = np.transpose(frames, [3, 0, 1, 2]).astype(np.float)

        c, t, h, w = frames.shape
        if t < self.clip_length:
            pad_t = self.clip_length - t
            zero_clip = np.ones([c, pad_t, h, w], dtype=frames.dtype) * 127.5
            frames = np.concatenate([frames, zero_clip], 1)

        # random crop and flip
        if self.training:
            frames = self.random_flip(self.random_crop(frames))
        else:
            frames = self.center_crop(frames)

        input_data = torch.from_numpy(frames.copy()).float()
        if self.rgb_norm:
            input_data = (input_data / 255.0) * 2.0 - 1.0
        annos = annos_transform(annos, self.clip_length)
        target = np.stack(annos, 0)

        return input_data, target
