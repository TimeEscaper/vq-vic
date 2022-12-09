import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict
from nip import nip


@nip
class DistortionLoss(nn.Module):

    def __init__(self,
                 aux_weights: Optional[Dict[str, float]] = None):
        super(DistortionLoss, self).__init__()
        self._aux_weights = aux_weights or {}

    def forward(self, model_output: Dict[str, torch.Tensor], gt_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert "x_hat" in model_output
        x_hat = model_output["x_hat"]
        mse = F.mse_loss(x_hat, gt_image)

        total_loss = mse
        losses = {"loss_distortion": mse}
        for k, v in model_output.items():
            if k.startswith("loss_") and (k in self._aux_weights):
                losses[k] = v
                total_loss = total_loss + self._aux_weights[k] * v

        losses["loss"] = total_loss
        return losses


@nip
class RateDistortionLoss(nn.Module):

    def __init__(self, lmbda: float, commit_weight: float):
        super(RateDistortionLoss, self).__init__()
        self._lmbda = lmbda
        self._commit_weight = commit_weight

    def forward(self, model_output: Dict[str, torch.Tensor], gt_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert "x_hat" in model_output
        x_hat = model_output["x_hat"]
        N, _, H, W = x_hat.size()
        num_pixels = N * H * W
        mse = F.mse_loss(x_hat, gt_image)

        losses = {"loss_distortion": mse}
        total_loss = self._lmbda * mse

        if "loss_commit" in model_output:
            total_loss = total_loss + self._commit_weight * model_output["loss_commit"]
            losses["loss_commit"] = model_output["loss_commit"]

        bpp_loss = 0.
        for k, v in model_output.items():
            if k.endswith("_likelihoods"):
                bpp_loss = bpp_loss + (torch.log(v).sum() / (-math.log(2) * num_pixels))
        total_loss += bpp_loss
        losses["loss_bpp"] = bpp_loss

        losses["loss"] = total_loss

        return losses
