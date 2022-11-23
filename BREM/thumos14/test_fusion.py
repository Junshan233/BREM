import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
import sys
import copy
sys.path.append('./')
from BREM.common import videotransforms
from BREM.common.thumos_dataset import get_video_info, get_class_index_map
from BREM.common.segment_utils import softnms_v2, softnms_v3
from BREM.common.config import config
from BREM.evaluation.eval_detection import ANETdetection
from BREM.thumos14.BaselineModel import BaseModel
from BREM.thumos14.BEM import BEMModule
from torch.utils.data import DataLoader, Dataset
import io

num_classes = config['dataset']['num_classes']
conf_thresh = config['testing']['conf_thresh']
top_k = config['testing']['top_k']
nms_thresh = config['testing']['nms_thresh']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']
checkpoint_path = config['testing']['checkpoint_path']
json_name = config['testing']['output_json']
output_path = config['testing']['output_path']
softmax_func = False
if not os.path.exists(output_path):
    os.makedirs(output_path)
fusion = config['testing']['fusion']

if config.model.head.type == "2stage_g":
    g_args = config.model.prop_g_args
    use_offset = g_args.use_offset
    use_ext_cls = g_args.use_confs
    use_ext_quality = g_args.use_q
else:
    use_offset = False
    use_ext_cls = False
    use_ext_quality = False

# getting path for fusion
rgb_data_path = config['testing'].get('rgb_data_path',
                                      './datasets/thumos14/test_npy/')
flow_data_path = config['testing'].get('flow_data_path',
                                       './datasets/thumos14/test_flow_npy/')
rgb_checkpoint_path = config['testing'].get('rgb_checkpoint_path',
                                            './models/thumos14/checkpoint-15.ckpt')
flow_checkpoint_path = config['testing'].get('flow_checkpoint_path',
                                             './models/thumos14_flow/checkpoint-16.ckpt')

def get_data_sample(video_data: torch.Tensor, offset):
    input_data = video_data[:, offset: offset \
        + clip_length]
    c, t, h, w = input_data.shape
    if t < clip_length:
        pad_t = clip_length - t
        zero_clip = video_data.new_zeros((c, pad_t, h, w))
        input_data = torch.cat([input_data, zero_clip], dim=1)
    return input_data


def main(data_loader, json_name):
    video_infos = get_video_info(config['dataset']['testing']['video_info_path'])
    originidx_to_idx, idx_to_class = get_class_index_map()
    assert fusion, "only suppert fusion."
    assert config.model.model_type in ["BEMModule", "BaseModel"], "only support BEMModule | BaseModel."

    if fusion:
        cfg_rgb = copy.deepcopy(config)
        cfg_rgb.model.in_channels = 3
        cfg_flow = copy.deepcopy(config)
        cfg_flow.model.in_channels = 2
        rgb_net = globals()[config.model.model_type](cfg_rgb, training=False)
        flow_net = globals()[config.model.model_type](cfg_flow, training=False)
        if calcute_score:
            rgb_net.load_state_dict(torch.load(rgb_checkpoint_path))
            flow_net.load_state_dict(torch.load(flow_checkpoint_path))
        rgb_net.eval().cuda()
        flow_net.eval().cuda()
        net = rgb_net
    else:
        net = globals()[config.model.model_type](config, training=False)
        net.load_state_dict(torch.load(checkpoint_path))
        net.eval().cuda()

    result_dict = {}
    for video_name, data, flow_data in tqdm.tqdm(data_loader, ncols=0):
        video_name = video_name[0]
        if calcute_score:
            data = data.squeeze(0)
            if not isinstance(flow_data, list):
                flow_data = flow_data.squeeze(0)
        sample_count = video_infos[video_name]['sample_count']
        sample_fps = video_infos[video_name]['sample_fps']
        if sample_count < clip_length:
            offsetlist = [0]
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]

        output = []
        for cl in range(num_classes):
            output.append([])
        res = torch.zeros(num_classes, top_k, 3)

        # print(video_name)
        for offset in offsetlist:
            if calcute_score:
                clip = get_data_sample(data, offset)
                clip = clip.float()
                clip = (clip / 255.0) * 2.0 - 1.0
                clip = clip.unsqueeze(0).cuda()
                if fusion:
                    flow_clip = get_data_sample(flow_data, offset)
                    flow_clip = flow_clip.float()
                    flow_clip = (flow_clip / 255.0) * 2.0 - 1.0
                    flow_clip = flow_clip.unsqueeze(0).cuda()
                # clip = torch.from_numpy(clip).float()

                with torch.no_grad():
                    output_data = net(clip, return_loss=False)
                    if fusion:
                        flow_output_data = flow_net(flow_clip, return_loss=False)

                rgb_head_out, priors = output_data[:2]
                flow_head_out, priors = flow_output_data[:2]
                score_save = dict(
                    rgb_head_out=rgb_head_out,
                    flow_head_out=flow_head_out,
                    priors=priors,
                )
                if config.model.model_type == "BEMModule":
                    rgb_bem_out_dict = output_data[2]
                    flow_bem_out_dict = flow_output_data[2]
                    score_save.update(dict(
                        rgb_bem_out_dict=rgb_bem_out_dict,
                        flow_bem_out_dict=flow_bem_out_dict,
                    ))
                if video_name not in score_cache.keys():
                    score_cache[video_name] = {}
                score_cache[video_name][offset] = score_save
            else:
                score_cur = score_cache[video_name][offset]
                rgb_head_out = score_cur["rgb_head_out"]
                flow_head_out = score_cur["flow_head_out"]
                priors = score_cur["priors"]
                if config.model.model_type == "BEMModule":
                    rgb_bem_out_dict = score_cur["rgb_bem_out_dict"]
                    flow_bem_out_dict = score_cur["flow_bem_out_dict"]

                loc = (rgb_head_out["locs"] + flow_head_out["locs"])[0] / 2.0
                conf = (rgb_head_out["confs"].sigmoid() + flow_head_out["confs"].sigmoid())[0] / 2.0
                quality = (rgb_head_out["quality"] + flow_head_out["quality"])[0] / 2.0

                # bem_out_dict = dict()
                # for key in rgb_bem_out_dict.keys():
                #     bem_out_dict[key] = (rgb_bem_out_dict[key] + flow_bem_out_dict[key]) / 2.0

                if use_offset:
                    offset_p = (rgb_head_out["offset"] + flow_head_out["offset"])[0] / 2.0
                    pre_loc_w = loc[:, :1] + loc[:, 1:]
                    loc = 0.5 * pre_loc_w * offset_p + loc
                if config.model.head.type == "align_head":
                    offset_p = (rgb_head_out["offset"] + flow_head_out["offset"])[0] / 2.0
                    pre_loc_w = loc[:, :1] + loc[:, 1:]
                    loc = 0.5 * pre_loc_w * offset_p + loc
                    confs_2 = (rgb_head_out["confs_2"].sigmoid() + flow_head_out["confs_2"].sigmoid())[0] / 2.0
                    conf = (conf + confs_2) / 2.0
            
                decoded_segments = torch.cat(
                    [priors[:, :1] * clip_length - loc[:, :1],
                    priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
                decoded_segments.clamp_(min=0, max=clip_length - 1)

                if use_ext_cls:
                    ext_confs = (rgb_head_out["ext_confs"].sigmoid() + flow_head_out["ext_confs"].sigmoid())[0] / 2.0
                    conf = (conf + ext_confs) / 2.0
            # prop_conf = score_func(prop_conf)
            # center = center.sigmoid()
                if use_ext_quality:
                    ext_quality = (rgb_head_out["ext_quality"] + flow_head_out["ext_quality"])[0] / 2.0
                    quality = (quality + ext_quality) / 2.0
                quality = quality.sigmoid_()


                conf[(decoded_segments[:, 0] <= 0) | (decoded_segments[:, 1] >= clip_length)] = 0.
                # conf = (conf * fitness + prop_conf) / 2.0
                # conf = (conf + prop_conf) / 2.0
                quality = quality.mean(1, keepdim=True)
                conf = conf * quality

                if config.model.model_type == "BEMModule":
                    rgb_boundary_conf, decoded_segments = net.bem.inference(
                        rgb_bem_out_dict, decoded_segments, conf, low=0.4)
                    flow_boundary_conf, decoded_segments = net.bem.inference(
                        flow_bem_out_dict, decoded_segments, conf, low=0.4)
                    boundary_conf = (rgb_boundary_conf + flow_boundary_conf) / 2
                    conf = boundary_conf * conf
            
                conf = conf.view(-1, num_classes).transpose(1, 0)
                conf_scores = conf.clone()

                for cl in range(0, num_classes):
                    c_mask = conf_scores[cl] > conf_thresh
                    scores = conf_scores[cl][c_mask]
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)

                    segments = decoded_segments[l_mask].view(-1, 2)
                    segments = (segments + offset) / sample_fps  # 对应于原视频的开始和结束时间
                    segments = torch.cat([segments, scores.unsqueeze(1)], -1)  # shape: [number, 3]; 3: start time, end time, class confidence

                    output[cl].append(segments)

        if not calcute_score:
            # print(output[1][0].size(), output[2][0].size())
            sum_count = 0
            for cl in range(0, num_classes):
                if len(output[cl]) == 0:
                    continue
                tmp = torch.cat(output[cl], 0)
                tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k)
                res[cl, :count] = tmp
                sum_count += count

            sum_count = min(sum_count, top_k)
            flt = res.contiguous().view(-1, 3)
            flt = flt.view(num_classes, -1, 3)
            proposal_list = []
            for cl in range(0, num_classes):
                class_name = idx_to_class[cl + 1]
                tmp = flt[cl].contiguous()
                tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
                if tmp.size(0) == 0:
                    continue
                tmp = tmp.detach().cpu().numpy()
                for i in range(tmp.shape[0]):
                    tmp_proposal = {}
                    tmp_proposal['label'] = class_name
                    tmp_proposal['score'] = float(tmp[i, 2])
                    tmp_proposal['segment'] = [float(tmp[i, 0]),
                                            float(tmp[i, 1])]
                    proposal_list.append(tmp_proposal)

            result_dict[video_name] = proposal_list

    if not calcute_score:
        output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}

        with open(os.path.join(output_path, json_name), "w") as out:
            json.dump(output_dict, out)

        gt_json = './thumos_annotations/thumos_gt.json'
        output_json = os.path.join(output_path, json_name)
        tious = [0.3, 0.4, 0.5, 0.6, 0.7]
        anet_detection = ANETdetection(
            ground_truth_filename=gt_json,
            prediction_filename=output_json,
            subset='test', tiou_thresholds=tious)
        mAPs, average_mAP, ap = anet_detection.evaluate()
        for (tiou, mAP) in zip(tious, mAPs):
            print("mAP at tIoU {} is {}".format(tiou, mAP))
        print('Average mAP:', average_mAP)
    else:
        print("score calculation finished.")


class TestDataset(Dataset):
    def __init__(self, npy_data_path, flow_data_path) -> None:
        super(TestDataset, self).__init__()
        self.video_infos = get_video_info(config['dataset']['testing']['video_info_path'])
        self.video_name = list(self.video_infos.keys())
        self.npy_data_path = npy_data_path
        self.flow_data_path = flow_data_path
        self.centor_crop = videotransforms.CenterCrop(config['dataset']['testing']['crop_size'])
        if "flow" in config['dataset']['training']['video_data_path']:
            self.npy_data_path = self.flow_data_path

    def __len__(self) -> int:
        return len(self.video_name)

    def __getitem__(self, index: int):
        video_n = self.video_name[index]
        if not calcute_score:
            return video_n, [], []
        data, flow_data = [], []
        data = np.load(os.path.join(self.npy_data_path, video_n + '.npy'))
        data = np.transpose(data, [3, 0, 1, 2])
        data = self.centor_crop(data)
        data = torch.from_numpy(data)
        if fusion:
            flow_data = np.load(os.path.join(self.flow_data_path, video_n + '.npy'))
            flow_data = np.transpose(flow_data, [3, 0, 1, 2])
            flow_data = self.centor_crop(flow_data)
            flow_data = torch.from_numpy(flow_data)
        return video_n, data, flow_data

rgb_model_path = ''
flow_model_path = ''
rgb_weight = 0.5
flow_weight = 0.5


def test_pai():
    global rgb_model_path
    global flow_model_path
    global conf_thresh
    test_dataset = TestDataset(
        npy_data_path='./datasets/thumos14/test_npy/',
        flow_data_path='./datasets/thumos14/test_flow_npy/'
        )
    data_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        num_workers=40, 
        drop_last=False,
        shuffle=False)

    if fusion:
        global rgb_weight
        global flow_weight
        rgb_weight = 0.5
        flow_weight = 1 - rgb_weight
        rgb_model_path = f'models/thumos14/{rgb_log_file}/checkpoint-{rgb_epoch}.ckpt'
        flow_model_path = f'models/thumos14_flow/{flow_log_file}/checkpoint-{flow_epoch}.ckpt'
        json_name = "fusion.json"
        main(data_loader, json_name)

global rgb_log_file, flow_log_file, rgb_epoch, flow_epoch
global calcute_score
global score_cache

if __name__ == "__main__":
    # $ CUDA_VISIBLE_DEVICES=1 python3 BREM/thumos14/test_fusion.py configs/thumos14.yaml

    """
    # Step 1
    set 'calcute_score = True', and run the command:
    $ python3 BREM/thumos14/test_fusion.py configs/thumos14.yaml
    the output of model will be saved on 'score_save_path'
    # Step 2
    set 'calcute_score = False', and run the command:
    $ python3 BREM/thumos14/test_fusion.py configs/thumos14.yaml
    you will get the results:
        mAP at tIoU 0.3 is 0.706749662331388
        mAP at tIoU 0.4 is 0.6609676448700232
        mAP at tIoU 0.5 is 0.6000991848265967
        mAP at tIoU 0.6 is 0.5005502616477655
        mAP at tIoU 0.7 is 0.3643073137022799
        Average mAP: 0.5665348134756107
    """

    # args
    rgb_log_file = "2022-04-04_19:13:57_ad52"
    rgb_epoch = 14
    flow_log_file = "2022-04-04_19:34:31_fn2"
    flow_epoch = 16
    conf_thresh = 0.001

    calcute_score = True
    fusion = True
    score_save_path = "./output/thumos_score/"
    score_file_name = "test1.pt"
    score_cache = {}
    if not calcute_score:
        score_cache = torch.load(os.path.join(score_save_path, score_file_name))
    
    test_pai()

    if calcute_score:
        torch.save(score_cache, os.path.join(score_save_path, score_file_name))
