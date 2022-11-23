import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from BREM.common.config import config
from BREM.common.layers import Unit1D
from BREM.common.module import (
    DeformConv1D
)
from torch import Tensor


class BaseHead(nn.Module):
    def __init__(
        self, in_channel, mid_channel, num_classes, fpn_strides, *args, **kwargs
    ):
        super().__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.fpn_strides = fpn_strides
        self.use_quality_1 = config.model.use_quality_1
        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=in_channel if i == 0 else mid_channel,
                        output_channels=mid_channel,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, mid_channel),
                    nn.ReLU(inplace=True),
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=in_channel if i == 0 else mid_channel,
                        output_channels=mid_channel,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, mid_channel),
                    nn.ReLU(inplace=True),
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=mid_channel,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None,
        )
        if self.use_quality_1:
            self.quality_num = 2
            in_channel_q = mid_channel
            self.quality = Unit1D(
                in_channels=in_channel_q,
                output_channels=self.quality_num,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None,
            )
        self.conf_head = Unit1D(
            in_channels=mid_channel,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None,
        )

    def _reset_bias(self, bias_value):
        nn.init.constant_(self.conf_head.conv1d.bias, bias_value)

    def forward(self, features: List[Tensor], scales, *args, **kwargs):
        locs_ = []
        confs_ = []
        quality_ = []
        for i, (feat, scale) in enumerate(zip(features, scales)):
            loc_feat = self.loc_tower(feat)
            conf_feat = self.conf_tower(feat)

            loc_se = self.loc_head(loc_feat)
            locs = scale(loc_se) * self.fpn_strides[i]
            locs = locs.permute(0, 2, 1).contiguous()

            if self.use_quality_1:
                quality_se = self.quality(loc_feat)
                quality = quality_se.permute(0, 2, 1).contiguous()
                quality_.append(quality)

            confs = self.conf_head(conf_feat).permute(0, 2, 1).contiguous()

            locs_.append(locs)
            confs_.append(confs)
        return locs_, confs_, quality_


class BaseHead2Stage(nn.Module):
    def __init__(
        self, in_channel, mid_channel, num_classes, fpn_strides, *args, **kwargs
    ):
        super().__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.fpn_strides = fpn_strides
        self.use_quality_1 = config.model.use_quality_1
        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=in_channel if i == 0 else mid_channel,
                        output_channels=mid_channel,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, mid_channel),
                    nn.ReLU(inplace=True),
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=in_channel if i == 0 else mid_channel,
                        output_channels=mid_channel,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, mid_channel),
                    nn.ReLU(inplace=True),
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=mid_channel,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None,
        )
        in_channel_q = mid_channel
        self.quality_num = 2
        if self.use_quality_1:
            self.quality = Unit1D(
                in_channels=in_channel_q,
                output_channels=self.quality_num,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None,
            )
        self.conf_head = Unit1D(
            in_channels=mid_channel,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None,
        )
        self.quality_ext = nn.Conv1d(in_channel_q, self.quality_num, 3, padding=1)
        self.stride = config.model.featmap_strides
        self.priors = kwargs["priors"]
        self.offset_subbranch = BoundarySubBranch(in_channel, out_conv=True)
        self.ext_confs_subbranch = BoundarySubBranch(
            in_channel,
            mid_channel,
            num_classes,
            conv_layer=1,
        )


    def _reset_bias(self, bias_value):
        nn.init.constant_(self.conf_head.conv1d.bias, bias_value)
        nn.init.constant_(self.ext_confs_subbranch.out_head.conv1d.bias, bias_value)

    def forward(self, features: List[Tensor], scales, *args, **kwargs):
        locs_ = []
        confs_ = []
        quality_ = []
        offset_ = []
        ext_confs_ = []
        quality_ext_ = []
        pse_proposal_ = []
        for i, (feat, scale) in enumerate(zip(features, scales)):
            loc_feat = self.loc_tower(feat)
            conf_feat = self.conf_tower(feat)

            loc_se = self.loc_head(loc_feat)
            locs = scale(loc_se) * self.fpn_strides[i]
            pse_proposal = locs
            self.priors = [x.type_as(feat) for x in self.priors]
            offset, offset_feat = self.offset_subbranch(
                feat,
                pse_proposal,
                self.stride[i]
            )
            offset_.append(offset.permute(0, 2, 1).contiguous())
            ext_confs, _ = self.ext_confs_subbranch(feat, pse_proposal, self.stride[i])
            ext_confs_.append(ext_confs.permute(0, 2, 1).contiguous())
            locs = locs.permute(0, 2, 1).contiguous()

            if self.use_quality_1:
                quality_se = self.quality(loc_feat)
                quality = quality_se.permute(0, 2, 1).contiguous()
                quality_.append(quality)

            quality_ext = self.quality_ext(offset_feat)
            quality_ext = quality_ext.permute(0, 2, 1).contiguous()
            quality_ext_.append(quality_ext)

            confs = self.conf_head(conf_feat).permute(0, 2, 1).contiguous()

            locs_.append(locs)
            confs_.append(confs)
            pse_proposal_.append(pse_proposal.permute(0, 2, 1).contiguous().detach())
        others = dict(quality_ext=quality_ext_, pse_proposal=pse_proposal_)
        return locs_, confs_, quality_, offset_, ext_confs_, others


class BoundarySubBranch(nn.Module):
    def __init__(
        self, in_channel, hid_channel=None, out_channel=2, conv_layer=1, out_conv=True
    ) -> None:
        super().__init__()
        if hid_channel is None:
            hid_channel = in_channel
        self.in_channel = in_channel
        self.conv_layer = conv_layer
        self.hid_channel = hid_channel
        self.out_channel = out_channel
        self.grad_mul = config.model.deform_conv_grad_mul
        self.out_conv = out_conv
        conv = []
        for i in range(conv_layer):
            conv.append(
                nn.Conv1d(
                    in_channel if i == 0 else hid_channel, hid_channel, 3, padding=1
                )
            )
            conv.append(nn.GroupNorm(32, hid_channel))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)
        self.deform_1d = DeformConv1D(hid_channel, hid_channel, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if self.out_conv:
            self.out_head = Unit1D(hid_channel, out_channel, 3, activation_fn=None)

    def _get_dcn_offset(self, bbox_pred: Tensor, gradient_mul: float, stride: int):
        bbox_pred_grad_mul = (
            1 - gradient_mul
        ) * bbox_pred.detach() + gradient_mul * bbox_pred
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, T = bbox_pred.size()
        y1 = bbox_pred_grad_mul[:, 0]
        y2 = bbox_pred_grad_mul[:, 1]
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(N, 2 * 3, T)
        bbox_pred_grad_mul_offset[:, 0] = -1.0 * y1
        bbox_pred_grad_mul_offset[:, 4] = y2
        return bbox_pred_grad_mul_offset.unsqueeze(-1)

    def forward(self, feat: Tensor, locs: Tensor, stride: int, *args, **kwargs):
        offset_feat = self.conv(feat)
        dcn_offset = self._get_dcn_offset(locs, self.grad_mul, stride)
        offset_feat = self.relu(self.deform_1d(offset_feat, dcn_offset))
        if self.out_conv:
            offset = self.out_head(offset_feat)
        else:
            offset = None
        return offset, offset_feat


class Head(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__()
        _type = cfg.model.head.type
        self._type = _type
        self.use_quality_1 = config.model.use_quality_1
        if _type == "base":
            self.head = BaseHead(*args, **kwargs)
        elif _type == "2stage_g":
            self.head = BaseHead2Stage(*args, **kwargs)
        else:
            raise ValueError("Unsurported 'head type'")

    def _reset_bias(self, bias_value):
        self.head._reset_bias(bias_value)

    def forward(self, *input, **kwargs):
        func = lambda x: torch.cat(x, dim=1)
        out_dict = {}
        head_out = self.head(*input, **kwargs)
        locs, confs, quality = head_out[:3]
        locs = func(locs)
        confs = func(confs)
        if self.use_quality_1:
            quality = func(quality)
        out_dict.update(dict(locs=locs, confs=confs, quality=quality))
        if self._type == "2stage_g":
            offset, ext_confs, others = head_out[3:]
            quality_ext = others["quality_ext"]
            pse_proposal = others["pse_proposal"]
            offset = func(offset) if len(offset) > 0 else offset
            ext_confs = func(ext_confs) if len(ext_confs) > 0 else ext_confs
            ext_quality = func(quality_ext) if len(quality_ext) > 0 else quality_ext
            pse_proposal = func(pse_proposal)
            out_dict.update(dict(
                offset=offset,
                ext_confs=ext_confs,
                ext_quality=ext_quality,
                pse_proposal=pse_proposal,
            ))
        return out_dict
