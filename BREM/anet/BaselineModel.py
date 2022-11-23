import math
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from BREM.anet.module import FocalLoss_Ori
from BREM.common.layers import Unit1D, Unit3D
from BREM.common.loss import focal_loss, iou_loss, iou_loss_v2, sigmoid_focal_loss

from BREM.common.module import I3D_BackBone, ScaleExp
from BREM.common.head import Head
from torch import Tensor

num_classes = 2


class BaseModel(nn.Module):
    def __init__(self, cfg, training=True):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.conv_channels = 512
        self.layer_num = 6
        self.feat_t = 768 // 8
        self.roi_length = 10
        self.frame_num = 768
        self.feat_channels = [832, 1024]
        self.fpn_strides = [8, 16, 32, 64, 128, 256]
        self.num_classes = 2
        self.in_channels = cfg.model.in_channels
        self.backbone_model = cfg.model.backbone_model
        self._training = training

        self._buile_module()
        self.reset_params()

        self.backbone = I3D_BackBone(in_channels=self.in_channels)
        self._training = training

        if self._training:
            if self.backbone_model is None:
                self.backbone.load_pretrained_weight()
            else:
                self.backbone.load_pretrained_weight(self.backbone_model)

        if cfg.freeze_backbone:
            for name, parameter in self.backbone.named_parameters():
                if parameter.requires_grad:
                    parameter.requires_grad = False
        self.loss = self._build_loss()

    def _build_loss(self):
        return BaseModleLoss(self.cfg)

    def _buile_module(self):
        self._buile_pyramid_module()

    def _buile_pyramid_module(self):
        out_channels = self.conv_channels
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.frame_num = self.frame_num

        self.pyramids.append(
            nn.Sequential(
                Unit3D(
                    in_channels=self.feat_channels[1],
                    output_channels=out_channels,
                    kernel_shape=[1, 3, 3],
                    use_batch_norm=False,
                    padding="spatial_valid",
                    use_bias=True,
                    activation_fn=None,
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
        )
        for i in range(1, self.layer_num):
            self.pyramids.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=2,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.priors = []
        t = self.feat_t
        for i in range(self.layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t, i] for c in range(t)]).view(-1, 2)
            )
            t = t // 2

        self.head = Head(
            self.cfg,
            in_channel=out_channels,
            mid_channel=self.cfg.model.head.channel,
            num_classes=self.num_classes,
            priors=self.priors,
            length=self.frame_num,
            training=self._training,
            fpn_strides=self.fpn_strides,
        )

    def _forward(self, x: Tensor):
        feat_dict = self._backbone_forward(x)
        out = self._pyramid_forward(feat_dict)
        return out

    def _backbone_forward(self, x):
        feat_dict = self.backbone(x)
        return feat_dict

    def _pyramid_forward(
        self, feat_dict: Dict[str, Tensor], frame_level_feat: Optional[Tensor] = None
    ):
        pyramid_feats = []
        x1 = feat_dict["Mixed_5c"]
        for i, conv in enumerate(self.pyramids):
            if i == 0:
                x = conv(x1)
                x = x.squeeze(-1).squeeze(-1)
            else:
                x = conv(x)
            pyramid_feats.append(x)

        head_out = self.head(pyramid_feats, self.loc_heads, frame_level_feat)

        priors = torch.cat(self.priors, 0).cuda()
        return head_out, priors

    def forward_train(self, x, annos):
        out = self._forward(x)
        loss_dict = self.loss(out, annos)
        return loss_dict

    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if (
            isinstance(m, nn.Conv1d)
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Conv3d)
            or isinstance(m, nn.ConvTranspose3d)
        ):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
        for layer in self.head.modules():
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
        if hasattr(self, "bem") and self.bem_param_init == "normal":
            for layer in self.bem.modules():
                if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        prior_prob = 0.02
        bias_value = float(-np.log((1 - prior_prob) / prior_prob))
        self._reset_bias(bias_value)

    def _reset_bias(self, bias_value):
        pass

    def forward(self, x: Tensor, annos: List[Tensor] = None, return_loss=True):
        if return_loss:
            loss_dict = self.forward_train(x, annos)
            return loss_dict
        else:
            return self._forward(x)


class BaseLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.clip_length = cfg.dataset.training.clip_length
        self.iou_weight = 1
        self.cls_weight = 1
        self.qualtity_weight = 1
        self.num_classes = 2
        self.focal_loss = FocalLoss_Ori(
            self.num_classes, balance_index=0, size_average=False, alpha=0.25
        )
        self.offset_weight = 1
        self.cfg = cfg
        self.quality_num = 2
        self.bounds = [[0, 30], [15, 60], [30, 120], [60, 240], [96, 768], [256, 768]]
        self.prior_lb = self.prior_rb = None
        self.piou = self.cfg.training.piou

    def gen_bounds(self, priors):
        K = priors.size(0)
        prior_lb = priors[:, 1].clone()
        prior_rb = priors[:, 1].clone()
        for i in range(K):
            prior_lb[i] = self.bounds[int(prior_lb[i])][0]
            prior_rb[i] = self.bounds[int(prior_rb[i])][1]
        prior_lb = prior_lb.unsqueeze(1)
        prior_rb = prior_rb.unsqueeze(1)
        return prior_lb, prior_rb

    @staticmethod
    def _se_loss(loc_data, loc_t, pos, loc_se_mask, clip_length, weight=None):
        # pos = pos & loc_se_mask[..., 0] & loc_se_mask[..., 1]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 2)  # 第一阶段的边框回归的 iou loss，只计算动作帧的 loss
        loc_target = loc_t[pos_idx].view(-1, 2)
        if loc_p.numel() > 0:
            loss_l_iou = iou_loss(loc_p, loc_target, loss_type="giou", reduction="none")
        else:
            loss_l_iou = loc_p.sum()
        if weight is None:
            loss_l_iou = loss_l_iou.mean()
        else:
            weight = weight[pos]
            if weight.numel() > 0:
                loss_l_iou = (loss_l_iou * weight).sum() / weight.sum()
            else:
                loss_l_iou = loss_l_iou.mean()
        return loss_l_iou

    def _calc_quality_target_2_1(self, gt, pre_loc, priors):
        start_gt = gt[:, 0]
        end_gt = gt[:, 1]
        decoded = torch.stack(
            [
                priors[:, 0] * self.clip_length - pre_loc[:, 0],
                priors[:, 0] * self.clip_length + pre_loc[:, 1],
            ],
            dim=-1,
        )
        pre_segment = torch.stack(
            [
                torch.stack([decoded[:, 0], end_gt], dim=-1),
                torch.stack([start_gt, decoded[:, 1]], dim=-1),
            ],
            dim=0,
        )  # 2, num_priors, 2
        segment = gt.unsqueeze(0).expand_as(pre_segment).contiguous()
        ious = iou_loss_v2(
            pre_segment.view(-1, 2), segment.view(-1, 2), loss_type="calc iou"
        ).clamp_(min=0)
        return ious.view(2, -1).permute(1, 0)

    @torch.no_grad()
    def _calc_pyramid_target(self, targets: List[Tensor], loc_data, priors):
        num_batch = loc_data.size(0)
        num_priors = priors.size(0)

        # match priors and ground truth segments
        loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
        conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)
        quality_t = torch.Tensor(num_batch, num_priors, self.quality_num).to(
            loc_data.device
        )
        loc_se_mask = torch.BoolTensor(num_batch, num_priors, 2).to(loc_data.device)

        for idx in range(num_batch):
            truths = targets[idx][:, :-1]  # [N, 2], N is the number of action in clip
            labels = targets[idx][:, -1]
            pre_loc = loc_data[idx]
            """
            match gt
            """
            K = priors.size(0)
            N = truths.size(0)
            center = priors[:, 0].unsqueeze(1).expand(K, N)
            left = (center - truths[:, 0].unsqueeze(0).expand(K, N)) * self.clip_length
            right = (truths[:, 1].unsqueeze(0).expand(K, N) - center) * self.clip_length
            max_dis = torch.max(left, right)
            if self.prior_lb is None or self.prior_rb is None:
                self.prior_lb, self.prior_rb = self.gen_bounds(priors)
            l_bound = self.prior_lb.expand(K, N)
            r_bound = self.prior_rb.expand(K, N)
            area = left + right
            maxn = self.clip_length * 2
            area[left < 0] = maxn
            area[right < 0] = maxn
            area[max_dis <= l_bound] = maxn
            area[max_dis > r_bound] = maxn
            best_truth_area, best_truth_idx = area.min(1)

            loc_t[idx][:, 0] = (
                priors[:, 0] - truths[best_truth_idx, 0]
            ) * self.clip_length
            loc_t[idx][:, 1] = (
                truths[best_truth_idx, 1] - priors[:, 0]
            ) * self.clip_length
            conf = labels[best_truth_idx]
            conf[best_truth_area >= maxn] = 0
            conf_t[idx] = conf

            loc_se_mask[idx][:, 0] = True
            loc_se_mask[idx][:, 1] = True

            gt = truths[best_truth_idx, :] * self.clip_length
            quality_t[idx] = self._calc_quality_target_2_1(gt, pre_loc, priors)
        return loc_t, conf_t, quality_t, loc_se_mask

    @torch.no_grad()
    def _calc_offset_target(self, loc_t: Tensor, loc_p: Tensor, pos: Tensor):
        prop_w = loc_p[:, :, 0] + loc_p[:, :, 1]
        prop_loc_t = (loc_t - loc_p) / (0.5 * prop_w.unsqueeze(-1))
        iou = iou_loss(loc_p.view(-1, 2), loc_t.view(-1, 2), loss_type="calc iou")
        iou = iou.view(loc_t.size()[:2])
        offset_pos = pos & (iou > self.piou)
        return prop_loc_t, offset_pos

    @staticmethod
    def _offset_loss_l1(offset_p: Tensor, offset_t: Tensor, pos: Tensor):
        pos_prop = pos.unsqueeze(-1).expand_as(offset_p)
        offset_p = offset_p[pos_prop]
        offset_t = offset_t[pos_prop]
        if offset_p.numel() > 0:
            loss = F.smooth_l1_loss(offset_p, offset_t, reduction="sum")
        else:
            loss = offset_p.sum()
        loss /= max(pos.sum(), 1)
        return loss

    def _ext_confs_loss(self, conf, conf_t):
        conf_data = conf.view(-1, self.num_classes)
        targets_conf = conf_t.view(-1, 1)
        conf_data = F.softmax(conf_data, dim=1)
        loss_c = self.focal_loss(conf_data, targets_conf)
        return loss_c / max((conf_t > 0).sum(), 1)

    def _calc_2nd_stage_loss(self, head_out, targets, priors):
        offset_p = head_out["offset"]
        ext_confs = head_out["ext_confs"]
        ext_quality = head_out["ext_quality"]
        pse_proposal = head_out["pse_proposal"]
        loss_ext_confs, loss_offset, loss_q_ext = 0, 0, 0
        # quality label
        g_args = self.cfg.model.prop_g_args
        if g_args.use_offset:
            prop_pre_w = (pse_proposal[:, :, 0] + pse_proposal[:, :, 1]).unsqueeze(-1)
            loc_data_ = 0.5 * prop_pre_w * offset_p + pse_proposal
        else:
            loc_data_ = pse_proposal
        loc_t, conf_t, quality_t, _ = self._calc_pyramid_target(targets, loc_data_, priors)
        pos = conf_t > 0
        if g_args.use_q:
            pos_idx = pos.unsqueeze(-1).expand_as(ext_quality)
            quality_p = ext_quality[pos_idx]
            quality_t_p = quality_t[pos_idx]
            if quality_p.numel() > 0:
                loss_q_ext = F.binary_cross_entropy_with_logits(
                    quality_p, quality_t_p, reduction="mean"
                )
            else:
                loss_q_ext = quality_p.sum()
        # offset
        offset_t, offset_pos = self._calc_offset_target(loc_t, pse_proposal, pos)
        if g_args.use_offset:
            loss_offset = self._offset_loss_l1(offset_p, offset_t, offset_pos)
        # confs
        if g_args.use_confs:
            ext_conf_t = conf_t.clone()
            ext_conf_t[~offset_pos] = 0
            loss_ext_confs = self._ext_confs_loss(ext_confs, ext_conf_t)
        return loss_ext_confs, loss_offset, loss_q_ext


class BaseModleLoss(BaseLoss):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg

    def forward(
        self,
        predictions: List[Tensor],
        targets: List[Tensor],
    ):
        head_out, priors = predictions
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

        # stage1: loss softmax focal loss
        conf_data = conf_data.view(-1, self.num_classes)
        targets_conf = conf_t.view(-1, 1)
        conf_data = F.softmax(conf_data, dim=1)
        loss_c = self.focal_loss(conf_data, targets_conf)

        N = max(pos.sum(), 1)
        loss_c /= N
        loss_l_iou *= self.iou_weight
        loss_c *= self.cls_weight
        loss_q *= self.qualtity_weight
        loss = loss_l_iou + loss_c + loss_q
        loss_dict = OrderedDict(
            loss=loss,
            loss_loc_iou=loss_l_iou,
            loss_cls=loss_c,
            loss_quality=loss_q,
        )
        if self.cfg.model.head.type == "2stage_g":
            loss_ext_confs, loss_offset, loss_q_ext = self._calc_2nd_stage_loss(
                head_out, targets, priors
            )
            loss_ext_confs *= self.cls_weight
            loss_q_ext *= self.qualtity_weight
            loss_offset *= self.offset_weight
            loss_dict["loss_ext_confs"] = loss_ext_confs
            loss_dict["loss_offset"] = loss_offset
            loss_dict["loss_ext_quality"] = loss_q_ext
            loss_dict["loss"] += loss_ext_confs + loss_offset + loss_q_ext
        return loss_dict


def test_inference(repeats=3, clip_frames=256):
    model = BaseModel(config, training=False)
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

    test_inference(20, 256)
