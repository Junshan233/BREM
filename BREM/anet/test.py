import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from torch.utils.data import DataLoader, Dataset
import pickle
import sys
sys.path.insert(0, "./")
from BREM.common import videotransforms
from BREM.common.anet_dataset import get_video_info, load_json
from BREM.anet.BaselineModel import BaseModel
from BREM.anet.BEM import BEMModule
from BREM.common.segment_utils import softnms_v2
from BREM.common.config import config
from BREM.evaluation.eval_detection import ANETdetection
import io
from BREM.common.anet_dataset import video2npy

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
checkpoint_path = config['testing']['checkpoint_path']
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

global result_dict
result_dict = mp.Manager().dict()

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
    net = globals()[config.model.model_type](config, training=False)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval().cuda()

    for video_name in video_list:
        cuhk_score = data_set.cuhk_data_score[video_name[2:]]
        cuhk_class_1 = data_set.cuhk_data_action[np.argmax(cuhk_score)]
        cuhk_score_1 = max(cuhk_score)

        sample_count = data_set.video_infos[video_name]['frame_num']
        sample_fps = data_set.video_infos[video_name]['fps']
        duration = data_set.video_infos[video_name]['duration']

        offsetlist = [0]

        # data = np.load(os.path.join(mp4_data_path, video_name + '.npy'))
        data = data_set.get_data(video_name)
        data = torch.from_numpy(data.copy())

        output = []
        for cl in range(num_classes):
            output.append([])
        res = torch.zeros(num_classes, top_k, 3)

        for offset in offsetlist:
            clip = data[:, offset: offset + clip_length]
            clip = clip.float()
            if clip.size(1) < clip_length:
                tmp = torch.ones(
                    [clip.size(0), clip_length - clip.size(1), crop_size, crop_size]).float() * 127.5
                clip = torch.cat([clip, tmp], dim=1)
            clip = clip.unsqueeze(0).cuda()
            clip = (clip / 255.0) * 2.0 - 1.0
            with torch.no_grad():
                output_data = net(clip, return_loss=False)

            if config.model.model_type == "BaseModel":
                head_out, priors = output_data
            if config.model.model_type == "BEMModule":
                head_out, priors, bem_out_dict = output_data
            loc = head_out["locs"]
            conf = head_out["confs"]
            quality = head_out["quality"]


            loc = loc[0]
            conf = conf[0]
            conf = score_func(conf)

            if use_offset:
                offset_p = head_out["offset"][0]
                pre_loc_w = loc[:, :1] + loc[:, 1:]
                loc = 0.5 * pre_loc_w * offset_p + loc
            decoded_segments = torch.cat(
                [priors[:, :1] * clip_length - loc[:, :1],
                 priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
            decoded_segments.clamp_(min=0, max=clip_length - 1)

            if use_ext_cls:
                ext_confs = head_out["ext_confs"][0]
                ext_confs = score_func(ext_confs)
                conf = (conf + ext_confs) / 2.0

            quality = quality[0]
            if use_ext_quality:
                ext_quality = head_out["ext_quality"][0]
                quality = (quality + ext_quality) / 2.0
            quality = quality.sigmoid_()
            quality = quality.mean(1, keepdim=True)
            conf = conf * quality

            if config.model.model_type == "BEMModule":
                boundary_conf, decoded_segments = net.bem.inference(
                    bem_out_dict, decoded_segments, conf)
                conf = boundary_conf * conf

            conf = conf.view(-1, num_classes).transpose(1, 0)
            conf_scores = conf.clone()

            for cl in range(iter_start, num_classes):
                c_mask = conf_scores[cl] > 1e-9
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                segments = decoded_segments[l_mask].view(-1, 2)
                segments = (segments + offset) / sample_fps
                segments = torch.cat([segments, scores.unsqueeze(1)], -1)

                output[cl].append(segments)

        sum_count = 0
        for cl in range(iter_start, num_classes):
            if len(output[cl]) == 0:
                continue
            tmp = torch.cat(output[cl], 0)
            tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k, score_threshold=1e-9)
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


def test_func(cfg, log_file, epoch_list = [16], thread_num=2):
    global checkpoint_path
    processes = []
    lock = threading.Lock()
    data_set = ActivityNetTestDataset(cfg)
    for epoch_id in epoch_list:
        task_ID = 'none'
        json_name = f'task_{task_ID}_{epoch_id}.json'
        checkpoint_path = f'models/anet/{log_file}/checkpoint-{epoch_id}.ckpt'
        # checkpoint_path = f'models/anet_flow/{log_file}/checkpoint-{epoch_id}.ckpt'

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

        output_dict = {"version": "ActivityNet-v1.3", "results": dict(result_dict), "external_data": {}}

        with open(os.path.join(output_path, json_name), "w") as out:
            json.dump(output_dict, out)

        print(f"\n# epoch {epoch_id}")

        tious = np.linspace(0.5, 0.95, 10)
        anet_detection = ANETdetection(
            ground_truth_filename='anet_annotations/activity_net_1_3_new.json',
            prediction_filename=os.path.join(output_path, json_name),
            subset='validation', tiou_thresholds=tious)
        mAPs, average_mAP, ap = anet_detection.evaluate()
        for (tiou, mAP) in zip(tious, mAPs):
            print("mAP at tIoU {} is {}".format(tiou, mAP))
        print('Average mAP:', average_mAP)



class ActivityNetTestDataset():
    def __init__(self, cfg) -> None:
        super().__init__()
        self.video_infos = get_video_info(config['dataset']['testing']['video_info_path'],
                             subset='validation')
        self.video_name = list(self.video_infos.keys())
        cuhk_data = load_json('cuhk-val/cuhk_val_simp_share.json')
        self.cuhk_data_score = cuhk_data["results"]
        self.cuhk_data_action = cuhk_data["class"]
        self.video_dir = config['dataset']['training']['video_mp4_path']
        self.cfg = cfg
        self.centor_crop = videotransforms.CenterCrop(crop_size)

    def get_data(self, video_name):
        video_info = self.video_infos[video_name]
        data = np.load(os.path.join(self.video_dir, video_name + '.npy'))
        data = np.transpose(data, [3, 0, 1, 2])
        data = self.centor_crop(data)

        return data

if __name__ == "__main__":
    # python3 BREM/anet/test_pai.py configs/anet.yaml --nms_sigma=0.85
    """
    # RGB
    $ python3 BREM/anet/test_pai.py configs/anet.yaml --nms_sigma=0.85
    args:
        log_file="2022-04-04_19:34:32_fa2"
        epoch_list=[8]
    # Flow
    $ python3 BREM/anet/test_pai.py configs/anet_flow.yaml --nms_sigma=0.85
    args:
        log_file="2022-04-04_19:34:33_fa4"
        epoch_list=[9]

    !!!Warning: Please check the checkpoint path.
    """

    test_func(config, log_file="2022-04-04_19:34:32_fa2", epoch_list=[8])

    """Results
    RGB:
        mAP at tIoU 0.5 is 0.5070598047604052
        mAP at tIoU 0.55 is 0.47833479040731236
        mAP at tIoU 0.6 is 0.4463752980422788
        mAP at tIoU 0.65 is 0.41395058365272175
        mAP at tIoU 0.7 is 0.377489092989468
        mAP at tIoU 0.75 is 0.33668195430763637
        mAP at tIoU 0.8 is 0.2871111789812648
        mAP at tIoU 0.85 is 0.2314419459294733
        mAP at tIoU 0.8999999999999999 is 0.15763249098032098
        mAP at tIoU 0.95 is 0.05176766456939601
        Average mAP: 0.32878448046202774
    Flow:
        mAP at tIoU 0.5 is 0.5132005907866605
        mAP at tIoU 0.55 is 0.48603157447319456
        mAP at tIoU 0.6 is 0.45204337083884893
        mAP at tIoU 0.65 is 0.4207662475315999
        mAP at tIoU 0.7 is 0.38524357047882957
        mAP at tIoU 0.75 is 0.3440885063196428
        mAP at tIoU 0.8 is 0.2920261376386212
        mAP at tIoU 0.85 is 0.235606619553773
        mAP at tIoU 0.8999999999999999 is 0.15425498781260327
        mAP at tIoU 0.95 is 0.04339506667012618
        Average mAP: 0.33266566721039
    
    """