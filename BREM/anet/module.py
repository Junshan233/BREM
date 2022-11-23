import torch
import torch.nn.functional as F
from BREM.common.module import BEM as _BEM
from BREM.common.module import BEM_Loss as _BEM_Loss
from BREM.common.module import temporal_iop
from torch import Tensor, nn

__all__ = ["FocalLoss_Ori"]


class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
        self,
        num_class,
        alpha=[0.25, 0.75],
        gamma=2,
        balance_index=-1,
        size_average=True,
    ):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, "alpha should be in `(0,1)`)"
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError(
                "Not support alpha type, expect `int|float|list|tuple|torch.Tensor`"
            )

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps  # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            self.alpha = self.alpha.to(logpt.device)

        alpha_class = self.alpha.gather(0, target.view(-1))
        logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class BEM(_BEM):
    def __init__(
        self,
        dim,
        index_map,
        num_classes=1,
        tscale=768,
        num_sample_perbin=3,
        hid_dim_2d=128,
    ) -> None:
        super().__init__(
            dim,
            index_map,
            num_classes=num_classes,
            tscale=tscale,
            num_sample_perbin=num_sample_perbin,
            hid_dim_2d=hid_dim_2d,
        )

    def _reset_bias(self, bias_value):
        pass


class BEM_Loss(_BEM_Loss):
    def __init__(self, clip_length, index_map, num_classes=1):
        super().__init__(clip_length, index_map, num_classes=num_classes)
        self._lambda_2 = 1
