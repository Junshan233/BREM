import math
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from BREM.anet.BaselineModel import BaseLoss, BaseModel
from BREM.anet.BEM_1window import BEM_Loss_test, BEM_test
from BREM.anet.module import BEM, BEM_Loss
from BREM.common.config import config
from BREM.common.layers import Unit1D, Unit3D
from torch import Tensor


class IndexMap(nn.Module):
    def __init__(self, windows_num=22, map_type="contiguous") -> None:
        super().__init__()
        self.windows_num = windows_num
        if map_type[:8] == "adaptive":
            max_windows_size = int(map_type[-3:])
            windows_size = self._get_map_v1(max_windows_size)
        elif map_type[:5] == "range":
            size = map_type.split("_")[1:]
            min_size = int(size[0])
            max_size = int(size[1])
            windows_size = self._get_map_v2(min_size, max_size)
        else:
            raise ValueError("Unsupported map type.")
        self.windows_size = nn.Parameter(
            torch.Tensor(windows_size), requires_grad=False
        )

    def _get_map_v1(self, max_windows_size):
        return torch.linspace(1, max_windows_size, steps=self.windows_num)

    def _get_map_v2(self, min_size, max_size):
        return torch.linspace(min_size, max_size, steps=self.windows_num)

    def get_windows_size(self):
        return self.windows_size.clone()

    def index_to_windows(self, index: int):
        return self.windows_size[index].clone()

    def windows_to_index(self, windows_size_pred: Tensor):
        """
        args:
            window_size: (K,)
        """
        windows_size_pred = windows_size_pred.view(-1, 1)
        windows_size = self.windows_size.view(1, -1)
        error = (windows_size_pred - windows_size).abs()
        _, best_index = error.min(-1)
        return best_index

    def windows_to_index_interpolation(self, windows_size_pred: Tensor):
        """
        args:
            windows_size_pred: (K,)
        """
        windows_size_pred = windows_size_pred.view(-1, 1)
        windows_size = self.windows_size.view(1, -1)
        error = (windows_size_pred - windows_size).abs()
        _, neighbor_index = error.topk(k=2, dim=-1, largest=False)
        neighbor = windows_size.view(-1)[neighbor_index.view(-1)].view_as(
            neighbor_index
        )
        neighbor_index = neighbor_index.float()
        index = (
            (windows_size_pred.view(-1) - neighbor[:, 0])
            / (neighbor[:, 1] - neighbor[:, 0])
        ) * (neighbor_index[:, 1] - neighbor_index[:, 0]) + neighbor_index[:, 0]
        return index


class BEMModule(BaseModel):
    def __init__(self, cfg, training=True):
        self.bem_param_init = cfg.model.BEM.param_init
        super().__init__(cfg, training=training)

    def _buile_module(self):
        self.index_map = IndexMap(
            windows_num=self.cfg.model.BEM.windows_num,
            map_type=self.cfg.model.BEM.map_type,
        )
        self._buile_pyramid_module()
        self._build_bem_model()

    def _build_loss(self):
        return BEMModuleLoss(self.cfg, index_map=self.index_map)

    def _reset_bias(self, bias_value):
        self.bem._reset_bias(bias_value)
        super()._reset_bias(bias_value)

    def _build_bem_model(self):
        conv_channels = self.cfg.model.BEM.dim_conv
        self.bem = BEM(conv_channels, index_map=self.index_map)
        self._build_frame_level_feat_branch()

    def _build_frame_level_feat_branch(self):
        conv_channels = 512
        feat_channels_in = self.feat_channels
        feat_channels_out = self.cfg.model.BEM.dim_fpn
        out_channels = conv_channels
        self.layer1 = nn.Sequential(
            Unit3D(
                in_channels=feat_channels_in[0],
                output_channels=feat_channels_out,
                kernel_shape=[1, 6, 6],
                padding="spatial_valid",
                use_batch_norm=False,
                use_bias=True,
                activation_fn=None,
            ),
            nn.GroupNorm(32, feat_channels_out),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            Unit3D(
                in_channels=feat_channels_in[1],
                output_channels=feat_channels_out,
                kernel_shape=[1, 3, 3],
                use_batch_norm=False,
                padding="spatial_valid",
                use_bias=True,
                activation_fn=None,
            ),
            nn.GroupNorm(32, feat_channels_out),
            nn.ReLU(inplace=True),
        )
        self.in_conv = nn.Sequential(
            Unit1D(
                in_channels=feat_channels_out,
                output_channels=out_channels,
                kernel_shape=1,
                stride=1,
                use_bias=True,
                activation_fn=None,
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 1, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def _frame_level_feat_forward(self, feat_dict):
        x2 = feat_dict["Mixed_5c"]
        x1 = feat_dict["Mixed_4f"]
        x0 = self.layer1(x1)
        x0 = x0.squeeze(-1).squeeze(-1)
        x = self.layer2(x2)
        x = x.squeeze(-1).squeeze(-1)
        y = F.interpolate(x, x0.size()[2:], mode="nearest")
        base_feature = x0 + y

        base_feature = self.in_conv(base_feature)
        frame_level_feat = base_feature.unsqueeze(-1)
        frame_level_feat = F.interpolate(frame_level_feat, [self.frame_num, 1]).squeeze(
            -1
        )
        frame_level_feat = self.deconv(frame_level_feat)
        return frame_level_feat

    def _bem_forward(self, frame_level_feat):
        out_dict = dict()
        out_dict_ = self.bem(frame_level_feat)
        out_dict.update(out_dict_)
        return out_dict

    def _forward(self, x: Tensor):
        feat_dict = self._backbone_forward(x)
        frame_level_feat = self._frame_level_feat_forward(feat_dict)
        out_dict = self._bem_forward(frame_level_feat)
        head_out, priors = self._pyramid_forward(feat_dict, frame_level_feat)
        return head_out, priors, out_dict

    def forward_train(self, x, annos):
        output = self._forward(x)
        loss_dict = self.loss(output, annos)
        return loss_dict


class BEMModuleLoss(BaseLoss):
    def __init__(self, cfg, index_map) -> None:
        super().__init__(cfg)
        self.bem_weight = 1
        self.bem_args = self.cfg.model.BEM
        self.bem_loss = BEM_Loss(self.clip_length, index_map=index_map)

    def forward(self, predictions: List[Tensor], targets: List[Tensor]):
        head_out, priors, bem_out_dict = predictions
        loc_data = head_out["locs"]
        conf_data = head_out["confs"]
        quality = head_out["quality"]

        loc_t, conf_t, quality_t, loc_se_mask = self._calc_pyramid_target(
            targets, loc_data, priors
        )

        # bounding regression loss (iou loss)
        pos = conf_t > 0
        loss_l_iou = self._se_loss(
            loc_data, loc_t, pos, loc_se_mask, self.clip_length
        )

        # quality loss
        pos_idx = pos.unsqueeze(-1).expand_as(quality)
        quality_p = quality[pos_idx]
        quality_t_p = quality_t[pos_idx]
        if quality_p.numel() > 0:
            loss_q = F.binary_cross_entropy_with_logits(
                quality_p, quality_t_p, reduction="mean"
            )
        else:
            loss_q = quality_p.sum()

        # stage1 loss softmax focal loss
        conf_data = conf_data.view(-1, self.num_classes)
        targets_conf = conf_t.view(-1, 1)
        conf_data = F.softmax(conf_data, dim=1)
        loss_c = self.focal_loss(conf_data, targets_conf)

        loss_bem_pak = self.bem_loss(bem_out_dict, targets)
        (
            loss_bem,
            reg_loss_iou,
        ) = loss_bem_pak

        N = max(pos.sum(), 1)
        loss_c /= N

        loss_l_iou *= self.iou_weight
        loss_c *= self.cls_weight
        loss_q *= self.qualtity_weight
        loss_bem *= self.bem_weight
        loss = loss_l_iou + loss_c + loss_q + loss_bem
        loss_dict = OrderedDict(
            loss=loss,
            loss_loc_iou=loss_l_iou,
            loss_cls=loss_c,
            loss_quality=loss_q,
            loss_bem=loss_bem,
            reg_loss_iou=reg_loss_iou,
        )
        if self.cfg.model.head.type == "2stage_g":
            loss_ext_confs, loss_offset, loss_q_ext = self._calc_2nd_stage_loss(
                head_out, targets, priors
            )
            loss_ext_confs *= self.cls_weight
            loss_q_ext *= self.qualtity_weight
            loss_offset *= self.offset_weight
            if isinstance(loss_ext_confs, Tensor):
                loss_dict["loss_ext_confs"] = loss_ext_confs
            if isinstance(loss_offset, Tensor):
                loss_dict["loss_offset"] = loss_offset
            if isinstance(loss_q_ext, Tensor):
                loss_dict["loss_ext_quality"] = loss_q_ext
            loss_dict["loss"] += loss_ext_confs + loss_offset + loss_q_ext
        return loss_dict


def test_inference(repeats=3, clip_frames=256):
    model = BEMModule(config, training=False)
    model.eval()
    model.cuda()
    import time

    run_times = []
    x = torch.randn([1, 3, clip_frames, 96, 96]).cuda()
    warmup_times = 2
    for i in range(repeats + warmup_times):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            y = model(x, return_loss=False)
        torch.cuda.synchronize()
        run_times.append(time.time() - start)

    infer_time = np.mean(run_times[warmup_times:])
    infer_fps = clip_frames * (1.0 / infer_time)
    print("inference time (ms):", infer_time * 1000)
    print("infer_fps:", int(infer_fps))
    # print(y['loc'].size(), y['conf'].size(), y['priors'].size())


if __name__ == "__main__":
    from BREM.common.config import config

    # python BREM/thumos14/BEM.py configs/thumos14.yaml
    test_inference(20, 256)
