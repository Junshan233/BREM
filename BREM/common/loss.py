import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Tuple, Optional
from torch import Tensor


def focal_loss(logit, target, weight=None, gamma=2, alpha=0.25, reduction="sum"):
    """
    logit: [batch_size, 126, num_class]
    target: [batch_size, 126]
    is_in_boundary: [batch_size, 126]
    """
    num_class = logit.size(-1)
    target = target.view(-1, 1).cpu()
    logit = logit.view(-1, num_class)
    one_hot_key = torch.FloatTensor(logit.size(0), num_class + 1).zero_()
    one_hot_key = one_hot_key.scatter_(1, target, 1)
    one_hot_key = one_hot_key[:, 1:]
    one_hot_key = one_hot_key.to(logit.device)
    if weight is None:
        weight = torch.ones_like(one_hot_key)
    else:
        weight = weight.view_as(one_hot_key)
    loss = sigmoid_focal_loss(
        logit,
        one_hot_key,
        weight=weight,
        gamma=gamma,
        alpha=alpha,
        reduction=reduction,
    )
    return loss


def sigmoid_focal_loss(pred, target, weight, gamma=2.0, alpha=0.25, reduction="mean"):
    assert weight.size() == target.size()
    pos_num = weight.sum()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * weight
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.sum() / max(pos_num, 1)
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("reduction should be `none|mean|sum`")


def iou_loss(pred, target, weight=None, loss_type="giou", reduction="none"):
    """
    jaccard: A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    """
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]
    target_left = target[:, 0]
    target_right = target[:, 1]

    pred_area = pred_left + pred_right
    target_area = target_left + target_right

    eps = torch.finfo(torch.float32).eps

    inter = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    area_union = target_area + pred_area - inter
    ious = inter / area_union.clamp(min=eps)

    if loss_type == "linear_iou":
        loss = 1.0 - ious
    elif loss_type == "giou":
        ac_uion = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right
        )
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1.0 - gious
    else:
        loss = ious

    if weight is not None:
        loss = loss * weight.view(loss.size())
    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    return loss


def iou_loss_v2(pred, target, weight=None, loss_type="giou", reduction="none"):
    """
    jaccard: A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    """
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]
    target_left = target[:, 0]
    target_right = target[:, 1]

    pred_area = pred_right - pred_left
    target_area = target_right - target_left

    eps = torch.finfo(torch.float32).eps

    inter = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
    area_union = target_area + pred_area - inter
    ious = inter / area_union.clamp(min=eps)

    if loss_type == "linear_iou":
        loss = 1.0 - ious
    elif loss_type == "giou":
        ac_uion = torch.max(pred_right, target_right) - torch.min(
            pred_left, target_left
        )
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1.0 - gious
    else:
        loss = ious

    if weight is not None:
        loss = loss * weight.view(loss.size())
    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    return loss


def quality_focal_loss(
    pred,  # (n, 80)
    label,  # (n) 0, 1-80: 0 is neg, 1-80 is positive
    score,  # (n) reg target 0-1, only positive is good
    weight=None,
    beta=2.0,
    reduction="mean",
    avg_factor=None,
):
    # all goes to 0
    pred_sigmoid = pred.sigmoid()
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * pt.pow(beta)

    label = label - 1
    pos = (label >= 0).nonzero().squeeze(1)
    a = pos
    b = label[pos].long()

    # positive goes to bbox quality
    pt = score[a] - pred_sigmoid[a, b]
    loss[a, b] = F.binary_cross_entropy_with_logits(
        pred[a, b], score[a], reduction="none"
    ) * pt.pow(beta)

    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    if reduction == "mean":
        if avg_factor is not None:
            return loss.sum() / avg_factor
        else:
            return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
