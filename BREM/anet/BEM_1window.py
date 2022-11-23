import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from BREM.common.config import config
from BREM.common.layers import Unit1D
from BREM.common.loss import iou_loss_v2
from torch import Tensor

num_classes = 1  # sigmoid


class ReductionLayer_test(nn.Module):
    """For one window"""

    def __init__(self, in_dim, hid_dim, out_dim, number, _type="fc") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.number = number
        self._type = _type
        if _type == "fc":
            self._build_model_fc()
        elif _type == "mean_max":
            self._build_model_mean_max()
        else:
            raise ValueError("not supported _type, (fc|mean_max).")

    def _build_model_fc(self):
        self.layer = nn.Sequential(
            nn.Conv3d(
                self.in_dim,
                self.hid_dim,
                kernel_size=(self.number, 1, 1),
                stride=(self.number, 1, 1),
            ),
            nn.GroupNorm(32, self.hid_dim),
            nn.ReLU(inplace=True),
        )
        self.trans_layer = nn.Sequential(
            nn.Conv1d(self.hid_dim, self.out_dim, kernel_size=1),
            nn.GroupNorm(32, self.out_dim),
            nn.ReLU(inplace=True),
        )

    def _forward_fc(self, x: Tensor):
        y = self.layer(x).squeeze(2).squeeze(-1)
        y = self.trans_layer(y)
        return y

    def _build_model_mean_max(self):
        self.mean_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.max_pool = nn.AdaptiveMaxPool3d((1, None, None))
        self.trans_layer = nn.Sequential(
            nn.Conv1d(self.in_dim * 2, self.out_dim, kernel_size=1),
            nn.GroupNorm(32, self.out_dim),
            nn.ReLU(inplace=True),
        )

    def _forward_mean_max(self, x: Tensor):
        x_mean = self.mean_pool(x)
        x_max = self.max_pool(x)
        y = torch.cat([x_mean, x_max], dim=1).squeeze(2).squeeze(-1)
        y = self.trans_layer(y)
        return y

    def forward(self, x: Tensor):
        """
        args:
            x: (B, C, N, T, D)
        returns:
            Tensor, (B, C, T, D)
        """
        if self._type == "fc":
            return self._forward_fc(x)
        elif self._type == "mean_max":
            return self._forward_mean_max(x)


class BEM_test(nn.Module):
    def __init__(
        self,
        dim,
        tscale=768,
        num_sample_perbin=3,
        hid_dim_2d=128,
    ) -> None:
        super().__init__()
        self.windows_zize = config.model.BEM.one_window_size # e.g. 16
        self.dim = dim
        self.tscale = tscale
        self.num_sample_perbin = num_sample_perbin
        self.num_sample = config.model.BEM.num_sample
        self.windows_num = 1
        self.hid_dim_1d = 256
        self.hid_dim_2d = hid_dim_2d
        self.hid_dim_3d = 512
        self.windows_ratio = config.model.BEM.windows_ratio
        self.reduction_type = config.model.BEM.reduction_type

        self._get_interp1d_mask()
        self.conv_1 = nn.Sequential(
            Unit1D(dim, self.hid_dim_1d, kernel_shape=3, activation_fn=None),
            nn.GroupNorm(32, self.hid_dim_1d),
            nn.ReLU(inplace=True),
        )
        self.reduction_layer = ReductionLayer_test(
            self.hid_dim_1d,
            self.hid_dim_3d,
            self.hid_dim_2d,
            self.num_sample,
            _type=self.reduction_type,
        )
        self.conv_start = nn.Sequential(
            nn.Conv1d(self.hid_dim_2d, self.hid_dim_2d, kernel_size=3, padding=1),
            nn.GroupNorm(32, self.hid_dim_2d),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hid_dim_2d, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv_end = nn.Sequential(
            nn.Conv1d(self.hid_dim_2d, self.hid_dim_2d, kernel_size=3, padding=1),
            nn.GroupNorm(32, self.hid_dim_2d),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hid_dim_2d, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def _reset_bias(self):
        pass

    def forward(self, x: Tensor):
        """
        args:
            x: Tensor, (B, C, T)
        returns:
            confidence_start, confidence_end: (B, 2, T, D)
            confidence_cls: (B, 20, T, D)
        """
        x = self.conv_1(x)
        x_interpolate = self.sample(x)
        feat_map = self.reduction_layer(x_interpolate)
        confidence_start = self.conv_start(feat_map)
        confidence_end = self.conv_end(feat_map)
        confidence_map = {
            "confidence_start": confidence_start.unsqueeze(-1),
            "confidence_end": confidence_end.unsqueeze(-1)
        }
        return confidence_map

    def sample(self, feat_map):
        B, C, _ = feat_map.size()
        if self.sample_mask.device is not feat_map.device:
            self.sample_mask = self.sample_mask.to(feat_map.device)
        feat_map = torch.matmul(feat_map, self.sample_mask).reshape(
            B, C, self.num_sample, self.tscale, self.windows_num
        )
        return feat_map

    def _get_interp1d_bin_mask(
        self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin
    ):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[
                idx * num_sample_perbin : (idx + 1) * num_sample_perbin
            ]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []

        w_size = self.windows_zize
        w_size = w_size + w_size * self.windows_ratio
        mask_mat_vector = []
        for center_t in range(self.tscale):
            d = 1.0 * w_size * 0.5
            start_index = center_t - d
            end_index = center_t + d
            if start_index <= end_index:
                sample_xmin = start_index
                sample_xmax = end_index
                p_mask = self._get_interp1d_bin_mask(
                    sample_xmin,
                    sample_xmax,
                    self.tscale,
                    self.num_sample,
                    self.num_sample_perbin,
                )
            else:
                p_mask = np.zeros([self.tscale, self.num_sample])  # (T, N)
            mask_mat_vector.append(p_mask)
        mask_mat_vector = np.stack(mask_mat_vector, axis=2)  # (T, N, T)
        mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)  # (T, N, T, D)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = torch.tensor(mask_mat).reshape(self.tscale, -1)

    def inference(
        self,
        confidence_map: Dict[str, Tensor],
        segments: Tensor,
        *args,
        **kwargs
    ):
        """
        batch size = 1
        D = 1
        """
        confidence_start = confidence_map["confidence_start"][0]  # (2, T, D)
        confidence_end = confidence_map["confidence_end"][0]  # (2, T, D)
        confidence_start = (confidence_start[0]).unsqueeze(-1)
        confidence_end = (confidence_end[0]).unsqueeze(-1)
        confidence_start = 0.5 * confidence_start + 0.5
        confidence_end = 0.5 * confidence_end + 0.5

        start_index = segments[:, 0].long()
        end_index = segments[:, 1].long()

        conf_start = confidence_start[start_index, 0]
        conf_end = confidence_end[end_index, 0]

        conf = torch.pow(conf_start * conf_end, 0.5)
        return conf, segments


class BEM_Loss_test(nn.Module):
    """仅有一个长度的窗"""

    def __init__(self, clip_length):
        super().__init__()
        self.windows_zize = config.model.BEM.one_window_size # e.g. 16
        self.tscale = clip_length
        self.clip_length = clip_length
        self.windows_num = 1
        self._lambda_1 = config.model.BEM.lambda_1

        self.priors = torch.arange(0, self.clip_length, dtype=torch.float).view(-1, 1)

    def _get_iou_heatmap(self, gt: Tensor, duration: Tensor, segment: Tensor):
        N = gt.size(0)
        gt = gt.view(-1, 1)
        duration = duration.view(1, -1)
        segment_gt = torch.stack(
            [gt - duration / 2, gt + duration / 2], dim=-1
        )  # (N, D, 2)
        segment_gt = segment_gt.clamp_(0, self.clip_length)
        segment_gt = segment_gt.unsqueeze(0).expand(self.tscale, N, self.windows_num, 2)
        segment = segment.unsqueeze(1).expand(self.tscale, N, self.windows_num, 2)
        iou = iou_loss_v2(
            segment.reshape(-1, 2), segment_gt.reshape(-1, 2), loss_type="calc iou"
        )  # (T, N, D)
        iou = iou.view(self.tscale, N, self.windows_num).clamp_(0)
        iou_max = iou.max(1)[0]
        return iou_max

    def forward(self, conf_dict: Dict[str, Tensor], annos: List[Tensor]):
        confidence_start, confidence_end = (
            conf_dict["confidence_start"],
            conf_dict["confidence_end"],
        )  # (B, 2, T, D)
        device = confidence_start.device
        num_batch = len(annos)
        iou_target_s = torch.Tensor(num_batch, self.tscale, self.windows_num).to(device)
        iou_target_e = torch.Tensor(num_batch, self.tscale, self.windows_num).to(device)
        if self.priors.device is not device:
            self.priors = self.priors.to(device)

        with torch.no_grad():
            for idx in range(num_batch):
                truths = annos[idx][:, :-1] * self.clip_length
                N = truths.size(0)
                duration = torch.tensor(
                    [self.windows_zize], device=device, dtype=torch.float
                ).view(-1)
                center_t = torch.arange(
                    0, self.tscale, device=device, dtype=torch.float
                )
                segment_grid = torch.stack(
                    torch.meshgrid(
                        center_t,
                        duration,
                    ),
                    dim=-1,
                )  # (T, D, 2)
                segment = segment_grid.new_empty(*segment_grid.size())
                segment[..., 0] = segment_grid[..., 0] - segment_grid[..., 1] / 2
                segment[..., 1] = segment_grid[..., 0] + segment_grid[..., 1] / 2
                segment = segment.clamp_(0, self.clip_length)
                iou_target_s[idx] = self._get_iou_heatmap(
                    truths[:, 0], duration, segment
                )
                iou_target_e[idx] = self._get_iou_heatmap(
                    truths[:, 1], duration, segment
                )

        reg_loss_iou = 0
        reg_loss_iou += self._reg_loss_func(confidence_start[:, 0], iou_target_s)
        reg_loss_iou += self._reg_loss_func(confidence_end[:, 0], iou_target_e)

        reg_loss_iou = self._lambda_1 * reg_loss_iou / 2
        loss = reg_loss_iou
        return loss, reg_loss_iou

    @staticmethod
    def _reg_loss_func(pred_score, gt_iou_map):
        u_hmask = (gt_iou_map > 0.7).float()
        u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
        u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.0)).float()
        # u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / num_m
        u_smmask = torch.rand_like(gt_iou_map)
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > (1.0 - r_m)).float()

        r_l = num_h / num_l
        u_slmask = torch.rand_like(gt_iou_map)
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1.0 - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights, reduction="sum")
        loss = 0.5 * loss / max(torch.sum(weights), 1)
        # loss = F.l1_loss(pred_score * weights, gt_iou_map * weights, reduction="sum")
        # loss = loss / torch.sum(weights)
        return loss
