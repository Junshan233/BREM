import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
import copy
import pickle
import io
import sys
sys.path.append('./')
from BREM.common import videotransforms
from BREM.common.anet_dataset import get_video_info, load_json
from BREM.anet.BaselineModel import BaseModel
from BREM.anet.BEM import BEMModule
from BREM.common.segment_utils import softnms_v2
from BREM.common.config import config
from BREM.common.anet_dataset import video2npy
from BREM.evaluation.eval_detection import ANETdetection
import multiprocessing as mp
import threading

num_classes = 2
iter_start = 1
score_func = torch.nn.Softmax(-1)

conf_thresh = config['testing']['conf_thresh']
top_k = config['testing']['top_k']
nms_thresh = config['testing']['nms_thresh']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']
crop_size = config['dataset']['testing']['crop_size']
rgb_checkpoint_path = 'models/anet/checkpoint-10.ckpt'
flow_checkpoint_path = 'models/anet_flow/checkpoint-6.ckpt'
json_name = config['testing']['output_json']
output_path = config['testing']['output_path']
ngpu = config['ngpu']
softmax_func = True
if not os.path.exists(output_path):
    os.makedirs(output_path)

if config.model.head.type == "2stage_g":
    g_args = config.model.prop_g_args
    use_offset = g_args.use_offset
    use_ext_cls = g_args.use_confs
    use_ext_quality = g_args.use_q
else:
    use_offset = False
    use_ext_cls = False
    use_ext_quality = False

thread_num = ngpu
global result_dict
result_dict = mp.Manager().dict()

processes = []
lock = threading.Lock()

video_infos = get_video_info(config['dataset']['testing']['video_info_path'],
                             subset='validation')
rgb_mp4_data_path = 'datasets/activitynet/train_val_npy_112'
flow_mp4_data_path = 'datasets/activitynet/flow/train_val_npy_112'

if softmax_func:
    score_func = nn.Softmax(dim=-1)
else:
    score_func = nn.Sigmoid()

centor_crop = videotransforms.CenterCrop(crop_size)

video_list = list(video_infos.keys())
video_num = len(video_list)
per_thread_video_num = video_num // thread_num

cuhk_data = load_json('cuhk-val/cuhk_val_simp_share.json')
cuhk_data_score = cuhk_data["results"]
cuhk_data_action = cuhk_data["class"]

def sub_processor(lock, pid, video_list, data_set):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm.tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(0)
    rgb_net = globals()[config.model.model_type](config, training=False)
    flow_config = copy.deepcopy(config)
    flow_config.model.in_channels = 2
    flow_net = globals()[config.model.model_type](flow_config, training=False)
    if calcute_score:
        rgb_net.load_state_dict(torch.load(rgb_model_path))
        flow_net.load_state_dict(torch.load(flow_model_path))
    rgb_net.eval().cuda()
    flow_net.eval().cuda()

    for video_name in video_list:
        cuhk_score = data_set.cuhk_data_score[video_name[2:]]
        cuhk_class_1 = data_set.cuhk_data_action[np.argmax(cuhk_score)]
        cuhk_score_1 = max(cuhk_score)

        sample_count = data_set.video_infos[video_name]['frame_num']
        sample_fps = data_set.video_infos[video_name]['fps']
        duration = data_set.video_infos[video_name]['duration']

        offsetlist = [0]

        if calcute_score:
            rgb_data, flow_data = data_set.get_data(video_name)
            rgb_data = torch.from_numpy(rgb_data.copy())
            flow_data = torch.from_numpy(flow_data.copy())

        output = []
        for cl in range(num_classes):
            output.append([])
        res = torch.zeros(num_classes, top_k, 3)

        for offset in offsetlist:
            if calcute_score:
                rgb_clip = rgb_data[:, offset: offset + clip_length]
                rgb_clip = rgb_clip.float()

                flow_clip = flow_data[:, offset: offset + clip_length]
                flow_clip = flow_clip.float()

                if rgb_clip.size(1) < clip_length:
                    rgb_tmp = torch.ones(
                        [rgb_clip.size(0), clip_length - rgb_clip.size(1), crop_size, crop_size]).float() * 127.5
                    flow_tmp = torch.ones(
                        [flow_clip.size(0), clip_length - flow_clip.size(1), crop_size, crop_size]).float() * 127.5
                    rgb_clip = torch.cat([rgb_clip, rgb_tmp], dim=1)
                    flow_clip = torch.cat([flow_clip, flow_tmp], dim=1)
                rgb_clip = rgb_clip.unsqueeze(0).cuda()
                flow_clip = flow_clip.unsqueeze(0).cuda()
                rgb_clip = (rgb_clip / 255.0) * 2.0 - 1.0
                flow_clip = (flow_clip / 255.0) * 2.0 - 1.0

                with torch.no_grad():
                    rgb_output_data = rgb_net(rgb_clip, return_loss=False)
                    flow_output_data = flow_net(flow_clip, return_loss=False)

                score_save = dict(
                    rgb_output_data=rgb_output_data,
                    flow_output_data=flow_output_data,
                )
                if video_name not in score_cache.keys():
                    score_cache[video_name] = {}
                score_cache[video_name][offset] = score_save
            else:
                score_cur = score_cache[video_name][offset]
                rgb_output_data = score_cur["rgb_output_data"]
                flow_output_data = score_cur["flow_output_data"]
                rgb_head_out, priors = rgb_output_data[:2]
                flow_head_out, _ = flow_output_data[:2]
                if config.model.model_type == "BEMModule":
                    rgb_bem_out_dict = rgb_output_data[2]
                    flow_bem_out_dict = flow_output_data[2]

                loc = (rgb_head_out["locs"] + flow_head_out["locs"])[0] / 2.0
                conf = (rgb_head_out["confs"] + flow_head_out["confs"])[0] / 2.0
                conf = score_func(conf)
                quality = (rgb_head_out["quality"] + flow_head_out["quality"])[0] / 2.0

                if use_offset:
                    offset_p = (rgb_head_out["offset"] + flow_head_out["offset"])[0] / 2.0
                    pre_loc_w = loc[:, :1] + loc[:, 1:]
                    loc = 0.5 * pre_loc_w * offset_p + loc

                decoded_segments = torch.cat(
                    [priors[:, :1] * clip_length - loc[:, :1],
                    priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
                decoded_segments.clamp_(min=0, max=clip_length - 1)

                if use_ext_cls:
                    ext_confs = (rgb_head_out["ext_confs"] + flow_head_out["ext_confs"])[0] / 2.0
                    ext_confs = score_func(ext_confs)
                    conf = (conf + ext_confs) / 2.0

                if use_ext_quality:
                    ext_quality = (rgb_head_out["ext_quality"] + flow_head_out["ext_quality"])[0] / 2.0
                    quality = (quality + ext_quality) / 2.0
                quality = quality.sigmoid_()
                quality = quality.mean(1, keepdim=True)
                conf = conf * quality

                if config.model.model_type == "BEMModule":
                    bem_out_dict = {}
                    for k in rgb_bem_out_dict.keys():
                        bem_out_dict[k] = (rgb_bem_out_dict[k] + flow_bem_out_dict[k]) / 2.0

                    boundary_conf, decoded_segments = flow_net.bem.inference(
                        bem_out_dict, decoded_segments, conf, low=0.3)
                    conf = boundary_conf * conf

                conf = conf.view(-1, num_classes).transpose(1, 0)
                conf_scores = conf.clone()

                for cl in range(iter_start, num_classes):
                    c_mask = conf_scores[cl] > 0
                    scores = conf_scores[cl][c_mask]
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                    segments = decoded_segments[l_mask].view(-1, 2)
                    segments = (segments + offset) / sample_fps
                    segments = torch.cat([segments, scores.unsqueeze(1)], -1)

                    output[cl].append(segments)
        
        if not calcute_score:
            sum_count = 0
            for cl in range(iter_start, num_classes):
                if len(output[cl]) == 0:
                    continue
                tmp = torch.cat(output[cl], 0)
                tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k, score_threshold=1e-18)
                res[cl, :count] = tmp
                sum_count += count

            flt = res.contiguous().view(-1, 3)
            flt = flt.view(num_classes, -1, 3)
            proposal_list = []
            for cl in range(iter_start, num_classes):
                class_name = cuhk_class_1
                tmp = flt[cl].contiguous()
                tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
                if tmp.size(0) == 0:
                    continue
                tmp = tmp.detach().cpu().numpy()
                for i in range(tmp.shape[0]):
                    tmp_proposal = {}
                    start_time = max(0, float(tmp[i, 0]))
                    end_time = min(duration, float(tmp[i, 1]))
                    if end_time <= start_time:
                        continue

                    tmp_proposal['label'] = class_name
                    tmp_proposal['score'] = float(tmp[i, 2]) * cuhk_score_1
                    tmp_proposal['segment'] = [start_time, end_time]
                    proposal_list.append(tmp_proposal)

            result_dict[video_name[2:]] = proposal_list
        with lock:
            progress.update(1)
    with lock:
        progress.close()

global rgb_model_path
global flow_model_path

def test_func(cfg, rgb_log_file, flow_log_file, rgb_epoch, flow_epoch, thread_num=2):
    global rgb_model_path
    global flow_model_path
    processes = []
    lock = threading.Lock()
    data_set = Dataset(cfg)
    task_ID = 'none'
    json_name = f'fusion.json'
    rgb_model_path = f'models/anet/{rgb_log_file}/checkpoint-{rgb_epoch}.ckpt'
    flow_model_path = f'models/anet/{flow_log_file}/checkpoint-{flow_epoch}.ckpt'

    video_list = data_set.video_name
    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num

    if thread_num == 1:
        sub_processor(lock, 0, video_list, data_set)
    else:
        for i in range(thread_num):
            if i == thread_num - 1:
                sub_video_list = video_list[i * per_thread_video_num:]
            else:
                sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
            p = mp.Process(target=sub_processor, args=(lock, i, sub_video_list, data_set))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    if not calcute_score:
        output_dict = {"version": "ActivityNet-v1.3", "results": dict(result_dict), "external_data": {}}

        with open(os.path.join(output_path, json_name), "w") as out:
            json.dump(output_dict, out)

        print(f"RGB: {rgb_log_file}, {rgb_epoch}")
        print(f"Flow: {flow_log_file}, {flow_epoch}")

        tious = np.linspace(0.5, 0.95, 10)
        anet_detection = ANETdetection(
            ground_truth_filename='anet_annotations/activity_net_1_3_new.json',
            prediction_filename=os.path.join(output_path, json_name),
            subset='validation', tiou_thresholds=tious)
        mAPs, average_mAP, ap = anet_detection.evaluate()
        for (tiou, mAP) in zip(tious, mAPs):
            print("mAP at tIoU {} is {}".format(tiou, mAP))
        print('Average mAP:', average_mAP)
    else:
        print("score calculation finished.")

class ActivityNetTestDataset():
    def __init__(self, cfg, video_dir) -> None:
        super().__init__()
        self.video_infos = get_video_info(cfg['dataset']['testing']['video_info_path'],
                             subset='validation')
        self.video_name = list(self.video_infos.keys())
        cuhk_data = load_json('cuhk-val/cuhk_val_simp_share.json')
        self.cuhk_data_score = cuhk_data["results"]
        self.cuhk_data_action = cuhk_data["class"]
        # self.video_dir = cfg['dataset']['training']['video_mp4_path']
        self.video_dir = video_dir
        self.centor_crop = videotransforms.CenterCrop(crop_size)

    def get_data(self, video_name):
        video_info = self.video_infos[video_name]
        data = np.load(os.path.join(self.video_dir, video_name + '.npy'))
        data = np.transpose(data, [3, 0, 1, 2])
        data = self.centor_crop(data)

        return data

class Dataset():
    def __init__(self, cfg) -> None:
        rgb_mp4_data_path = 'datasets/activitynet/train_val_npy_112'
        flow_mp4_data_path = 'datasets/activitynet/flow/train_val_npy_112'
        self.rgb_dataset = ActivityNetTestDataset(cfg, rgb_mp4_data_path)
        self.flow_dataset = ActivityNetTestDataset(cfg, flow_mp4_data_path)
        self.video_name = self.rgb_dataset.video_name
        self.cuhk_data_score = self.rgb_dataset.cuhk_data_score
        self.cuhk_data_action = self.rgb_dataset.cuhk_data_action
        self.video_infos = self.rgb_dataset.video_infos
    
    def get_data(self, video_name):
        rgb_data = self.rgb_dataset.get_data(video_name)
        flow_data = self.flow_dataset.get_data(video_name)
        return rgb_data, flow_data

global calcute_score
global score_cache


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python3 BREM/anet/test_fusion_dev.py  configs/anet.yaml --nms_sigma=0.85
    """
    # Step 1
    set 'calcute_score = True', and run the command:
    $ python3 BREM/anet/test_fusion_dev.py  configs/anet.yaml --nms_sigma=0.85
    the output of model will be saved on 'score_save_path'
    # Step 2
    set 'calcute_score = False', and run the command:
    $ python3 BREM/anet/test_fusion_dev.py  configs/anet.yaml --nms_sigma=0.85
    you will get the results:
        mAP at tIoU 0.5 is 0.5221759897420912
        mAP at tIoU 0.55 is 0.4930783642756433
        mAP at tIoU 0.6 is 0.4629986406804226
        mAP at tIoU 0.65 is 0.43059293811275867
        mAP at tIoU 0.7 is 0.3973296222262041
        mAP at tIoU 0.75 is 0.3539615200564451
        mAP at tIoU 0.8 is 0.3060505836467371
        mAP at tIoU 0.85 is 0.24584946036698174
        mAP at tIoU 0.8999999999999999 is 0.16940782697765605
        mAP at tIoU 0.95 is 0.05131819960460767
        Average mAP: 0.3432763145689548
    """

    calcute_score = False
    score_save_path = "./output/anet_score/"
    score_file_name = "fusion_anet1.pt"
    score_cache = {}
    if not calcute_score:
        score_cache = torch.load(os.path.join(score_save_path, score_file_name))
    test_func(
        config,
        rgb_log_file="2022-04-04_19:34:32_fa2",
        flow_log_file="2022-04-04_19:34:33_fa4",
        rgb_epoch=8,
        flow_epoch=9,
        thread_num=1
    )
    if calcute_score:
        torch.save(score_cache, os.path.join(score_save_path, score_file_name))

    