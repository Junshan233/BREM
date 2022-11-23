import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from BREM.common.config import config
from BREM.common.i3d_backbone import InceptionI3d
from BREM.common.layers import Unit1D
from BREM.common.loss import iou_loss_v2
from torch import Tensor

freeze_bn = config["model"]["freeze_bn"]
freeze_bn_affine = config["model"]["freeze_bn_affine"]


class I3D_BackBone(nn.Module):
    def __init__(
        self,
        final_endpoint="Mixed_5c",
        name="inception_i3d",
        in_channels=3,
        freeze_bn=freeze_bn,
        freeze_bn_affine=freeze_bn_affine,
    ):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(
            final_endpoint=final_endpoint, name=name, in_channels=in_channels
        )
        self._model.build()
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine
        if self._freeze_bn:
            # print('freeze all BatchNorm3d in I3D backbone.')
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    # print('freeze {}.'.format(name))
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def load_pretrained_weight(self, model_path="models/i3d_models/rgb_imagenet.pt"):
        self._model.load_state_dict(torch.load(model_path), strict=False)

    def train(self, mode=True):
        super(I3D_BackBone, self).train(mode)
        if self._freeze_bn and mode:
            # print('freeze all BatchNorm3d in I3D backbone.')
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    # print('freeze {}.'.format(name))
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def forward(self, x):
        return self._model.extract_features(x)


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)


def temporal_iop(proposal: Tensor, gt: Tensor) -> Tensor:
    """Compute IoP score between a groundtruth bbox and the proposals.

    Compute the IoP which is defined as the overlap ratio with
    groundtruth proportional to the duration of this proposal.

    Args:
        proposal: Tensor, (N, 2)
        gt: Tensor, (N, 2)

    Returns:
        Tensor, (N,)
    """
    len_anchors = proposal[:, 1] - proposal[:, 0]
    tmin = torch.max(proposal[:, 0], gt[:, 0])
    tmax = torch.min(proposal[:, 1], gt[:, 1])
    inter_len = torch.clamp(tmax - tmin, 0.0)
    scores = torch.div(inter_len, len_anchors)
    return scores


class ReductionLayer(nn.Module):
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
        elif _type in ["mean", "max"]:
            self._build_model_mean_or_max()
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
            nn.Conv2d(self.hid_dim, self.out_dim, kernel_size=1),
            nn.GroupNorm(32, self.out_dim),
            nn.ReLU(inplace=True),
        )

    def _forward_fc(self, x: Tensor):
        y = self.layer(x).squeeze(2)
        y = self.trans_layer(y)
        return y

    def _build_model_mean_max(self):
        self.mean_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.max_pool = nn.AdaptiveMaxPool3d((1, None, None))
        self.trans_layer = nn.Sequential(
            nn.Conv2d(self.in_dim * 2, self.out_dim, kernel_size=1),
            nn.GroupNorm(32, self.out_dim),
            nn.ReLU(inplace=True),
        )

    def _build_model_mean_or_max(self):
        if self._type == "mean":
            self.pooling_layer = nn.AdaptiveAvgPool3d((1, None, None))
        elif self._type == "max":
            self.pooling_layer = nn.AdaptiveMaxPool3d((1, None, None))
        else:
            raise ValueError("not support type: {}.".format(self._type))
        self.trans_layer = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1),
            nn.GroupNorm(32, self.out_dim),
            nn.ReLU(inplace=True),
        )

    def _forward_mean_max(self, x: Tensor):
        x_mean = self.mean_pool(x)
        x_max = self.max_pool(x)
        y = torch.cat([x_mean, x_max], dim=1).squeeze(2)
        y = self.trans_layer(y)
        return y

    def _forward_mean_or_max(self, x: Tensor):
        x = self.pooling_layer(x)
        y = x.squeeze(2)
        y = self.trans_layer(y)
        return y

    def forward(self, x: Tensor):
        """
        args:
            x: Tensor | List[Tensor], (B, C, N, T, D)
        returns:
            Tensor, (B, C, T, D)
        """
        if self._type == "fc":
            return self._forward_fc(x)
        elif self._type == "mean_max":
            return self._forward_mean_max(x)
        elif self._type in ["mean", "max"]:
            return self._forward_mean_or_max(x)


class BEM(nn.Module):
    """Boundary Evaluation Module

    ref BMN，https://github.com/JJBOY/BMN-Boundary-Matching-Network
    """

    def __init__(
        self,
        dim,
        index_map,
        num_classes=config.dataset.num_classes,
        tscale=256,
        num_sample_perbin=3,
        hid_dim_2d=128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.index_map = index_map
        self.tscale = tscale
        self.num_sample_perbin = num_sample_perbin
        self.num_sample = config.model.BEM.num_sample
        self.windows_num = self.index_map.windows_num
        self.max_windows_size = self.index_map.index_to_windows(self.windows_num - 1)
        self.hid_dim_1d = 256
        self.hid_dim_2d = hid_dim_2d
        self.hid_dim_3d = 512
        self.windows_ratio = config.model.BEM.windows_ratio
        self.reduction_type = config.model.BEM.reduction_type
        self.in_plen_r = config.model.BEM.in_ratio
        self.out_plen_r = config.model.BEM.out_ratio
        self.num_classes = num_classes

        self._get_interp1d_mask()
        self.conv_1 = nn.Sequential(
            Unit1D(dim, self.hid_dim_1d, kernel_shape=3, activation_fn=None),
            nn.GroupNorm(32, self.hid_dim_1d),
            nn.ReLU(inplace=True),
        )
        self.reduction_layer = ReductionLayer(
            self.hid_dim_1d,
            self.hid_dim_3d,
            self.hid_dim_2d,
            self.num_sample,
            _type=self.reduction_type,
        )
        out_number = 1
        self.conv_start = nn.Sequential(
            nn.Conv2d(self.hid_dim_2d, self.hid_dim_2d, kernel_size=3, padding=1),
            nn.GroupNorm(32, self.hid_dim_2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid_dim_2d, out_number, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv_end = nn.Sequential(
            nn.Conv2d(self.hid_dim_2d, self.hid_dim_2d, kernel_size=3, padding=1),
            nn.GroupNorm(32, self.hid_dim_2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid_dim_2d, out_number, kernel_size=1),
            nn.Sigmoid(),
        )

    def _reset_bias(self, bias_value):
        pass

    def forward(self, x: Tensor, *args, **kwargs):
        """
        args:
            x: Tensor, (B, C, T)
        returns:
            confidence_start, confidence_end: (B, 2, T, D)
        """
        x = self.conv_1(x)
        x_interpolate = self.sample(x)
        feat_map = self.reduction_layer(x_interpolate)
        confidence_start = self.conv_start(feat_map)
        confidence_end = self.conv_end(feat_map)
        confidence_map = {
            "confidence_start": confidence_start,
            "confidence_end": confidence_end,}
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
        for w_size_idx in range(0, self.windows_num):
            w_size = self.index_map.index_to_windows(w_size_idx)
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
        fpn_conf: Tensor,
        low: float = 0.5
    ):
        """batch size = 1"""
        confidence_start = confidence_map["confidence_start"][0]  # (2, T, D)
        confidence_end = confidence_map["confidence_end"][0]  # (2, T, D)
        confidence_start = (confidence_start[0]).unsqueeze(-1)
        confidence_end = (confidence_end[0]).unsqueeze(-1)
        confidence_start = (1 - low) * confidence_start + low
        confidence_end = (1 - low) * confidence_end + low

        # rounding
        # plen = segments[:, 1] - segments[:, 0]
        # in_plen = plen / self.in_plen_r
        # out_plen = plen / self.out_plen_r
        # windows_size = out_plen + in_plen
        # windows_index = self.index_map.windows_to_index(windows_size)
        # start_index = segments[:, 0].long()
        # end_index = segments[:, 1].long()
        # conf_start = confidence_start[start_index, windows_index]
        # conf_end = confidence_end[end_index, windows_index]

        # bilinear interpolation
        grid_start, grid_end = self._get_interpolation_index(
            segments, self.in_plen_r, self.out_plen_r
        )

        confidence_start = confidence_start.permute(2, 0, 1)
        conf_start = (
            F.grid_sample(
                confidence_start.unsqueeze(0),
                grid_start,
                align_corners=True,
                padding_mode="border",
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
            .squeeze(0)
        )
        confidence_end = confidence_end.permute(2, 0, 1)
        conf_end = (
            F.grid_sample(
                confidence_end.unsqueeze(0),
                grid_end,
                align_corners=True,
                padding_mode="border",
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
            .squeeze(0)
        )

        # conf = torch.pow(conf_start * conf_end, 0.5) * fpn_conf
        conf = torch.pow(conf_start * conf_end, 0.5)
        return conf, segments

    def _get_interpolation_index(self, segments: Tensor, in_plen_r, out_plen_r):
        plen = segments[:, 1] - segments[:, 0]
        in_plen = plen / in_plen_r
        out_plen = plen / out_plen_r

        windows_size = out_plen + in_plen

        windows_size.clamp_(
            self.index_map.get_windows_size().min(),
            self.index_map.get_windows_size().max(),
        )
        windows_index = self.index_map.windows_to_index_interpolation(windows_size)
        windows_index_normed = 2 * windows_index / (self.windows_num - 1) - 1
        start_index = 2 * segments[:, 0] / (self.tscale - 1) - 1
        end_index = 2 * segments[:, 1] / (self.tscale - 1) - 1
        grid_start = (
            torch.stack([windows_index_normed, start_index], dim=-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        grid_end = (
            torch.stack([windows_index_normed, end_index], dim=-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return grid_start, grid_end


class BEM_Loss(nn.Module):
    def __init__(self, clip_length, index_map, num_classes=config.dataset.num_classes):
        super().__init__()
        self.tscale = clip_length
        self.clip_length = clip_length
        self.index_map = index_map
        self.windows_num = self.index_map.windows_num
        self.max_windows_size = self.index_map.index_to_windows(self.windows_num - 1)
        self.in_ratio = config.model.BEM.in_ratio
        self.out_ratio = config.model.BEM.out_ratio
        self._lambda_1 = config.model.BEM.lambda_1
        self.num_classes = num_classes

        self.priors = torch.arange(0, self.clip_length, dtype=torch.float).view(-1, 1)

    def _get_iou_heatmap(
        self,
        gt: Tensor,
        duration: Tensor,
        segment: Tensor,
        pos_mask: Tensor,
        ignore_mask: Tensor = None,
    ):
        """
        args:
            pos_mask: (N, D)
        """
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
        iou_max, max_index = iou.max(1)
        if pos_mask is not None:
            pos_mask = torch.gather(pos_mask, 0, max_index)
            iou_max[~pos_mask] = 0
        if ignore_mask is not None:
            iou_max[ignore_mask.bool()] = 0
        return iou_max

    def _get_mask(self, annos):
        if not config.model.BEM.use_mask:
            return None
        pos_mask = self._get_mask_v1(annos)
        return pos_mask

    def _get_mask_v1(self, annos: Tensor):
        k = config.model.BEM.match_topk
        truths = annos[:, :2] * self.clip_length
        length = truths[:, 1] - truths[:, 0]
        in_plen = length / self.in_ratio
        out_plen = length / self.out_ratio
        plen = (in_plen + out_plen).view(-1, 1)
        duration = self.index_map.get_windows_size()
        N, M = annos.size(0), duration.size(0)
        pos_mask = annos.new_zeros(N, M)
        distance = torch.abs(plen - duration.view(1, -1))
        _, best_match = distance.topk(k, dim=1, largest=False)
        pos_mask = pos_mask.scatter(1, best_match, 1)
        return pos_mask.bool()

    def forward(
        self,
        conf_dict: Dict[str, Tensor],
        annos: List[Tensor],
        ignore_mask: Tensor = None,
    ):
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
                duration = self.index_map.get_windows_size()
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
                pos_mask = self._get_mask(annos[idx])
                iou_target_s[idx] = self._get_iou_heatmap(
                    truths[:, 0],
                    duration,
                    segment,
                    pos_mask,
                    ignore_mask=ignore_mask[idx] if ignore_mask is not None else None,
                )
                iou_target_e[idx] = self._get_iou_heatmap(
                    truths[:, 1],
                    duration,
                    segment,
                    pos_mask,
                    ignore_mask=ignore_mask[idx] if ignore_mask is not None else None,
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


class ChannelAttnLayer(nn.Module):
    def __init__(self, in_channel, out_channel, down_ratio, use_conv=True, bias=True):
        super().__init__()
        self.use_conv = use_conv
        self.bias = bias
        self.attn_layer = nn.Sequential(
            nn.Conv1d(in_channel, in_channel // down_ratio, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel // down_ratio, in_channel, 1, bias=bias),
            nn.Sigmoid(),
        )
        if self.use_conv:
            self.reduction_layer = nn.Sequential(
                Unit1D(
                    in_channels=in_channel,
                    output_channels=out_channel,
                    kernel_shape=1,
                    stride=1,
                    use_bias=True,
                    activation_fn=None,
                ),
                nn.GroupNorm(32, out_channel),
                nn.ReLU(inplace=True),
            )

    def extra_repr(self) -> str:
        return "(bias={})".format(self.bias)

    def forward(self, feat: Tensor, feat_ave: Optional[Tensor] = None):
        if feat_ave is None:
            feat_ave = feat.mean(-1, keepdim=True)
        weight = self.attn_layer(feat_ave)
        if self.use_conv:
            feat = self.reduction_layer(weight * feat)
        else:
            feat = weight * feat
        return feat


class LayerAttnLayer(nn.Module):
    def __init__(self, in_channel, num_layer, down_ratio) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.feat_channel = in_channel * num_layer
        self.num_layer = num_layer
        self.attn_layer = nn.Sequential(
            nn.Conv1d(self.feat_channel, self.feat_channel // down_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feat_channel // down_ratio, num_layer, 1),
            nn.Sigmoid(),
        )
        self.reduction_layer = nn.Conv1d(self.feat_channel, in_channel, 1)
        self.norm = nn.GroupNorm(32, in_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor):
        B = feat.size(0)
        feat_ave = feat.mean(-1, keepdim=True)
        weight = self.attn_layer(feat_ave)
        conv_weight = self.reduction_layer.weight.reshape(
            1, self.in_channel, self.num_layer, self.in_channel
        )
        conv_weight = weight.reshape(B, 1, self.num_layer, 1) * conv_weight
        conv_weight = conv_weight.reshape(B, self.in_channel, self.feat_channel)
        feat = torch.bmm(conv_weight, feat)
        feat = self.relu(self.norm(feat))
        return feat


class DeformConv1D(nn.Module):
    """
    offset_mode: `(center|point)`. `center`: offset is the distance to the center of conv kernel. \
        `point`: offset is the distance to each point of conv kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        offset_mode: str = "center",
    ) -> None:
        super().__init__()
        kernel_size = (kernel_size, 1)
        stride = (stride, 1)
        padding = (padding, 0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.offset_mode = offset_mode
        self.conv = DeformConv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias
        )
        dcn_base_y = torch.arange(-padding[0], padding[0] + 1).float()
        dcn_base_x = torch.zeros_like(dcn_base_y)
        dcn_base_offset = torch.stack([dcn_base_y, dcn_base_x], dim=1).reshape(-1)
        dcn_base_offset = dcn_base_offset.view(1, -1, 1, 1)
        self.register_buffer("dcn_base_offset", dcn_base_offset)

    def forward(self, x: Tensor, offset: Tensor):
        """
        x: (B, C, T)
        """
        if self.offset_mode == "center":
            offset = offset - self.dcn_base_offset
        x = x.unsqueeze(-1)
        return self.conv(x, offset).squeeze(-1)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "{in_channels}"
        s += ", {out_channels}"
        s += ", kernel_size={kernel_size}"
        s += ", stride={stride}"
        s += ", padding={padding}" if self.padding != (0, 0) else ""
        s += ", bias=False" if self.bias is None else ""
        s += ", offset_mode={offset_mode}"
        s += ")"
        return s.format(**self.__dict__)
