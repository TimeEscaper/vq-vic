import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Tuple, List
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
class MSEWrapLoss(nn.Module):

    def __init__(self):
        super(MSEWrapLoss, self).__init__()

    def forward(self, model_output: Dict[str, torch.Tensor], gt_image: torch.Tensor) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss = F.mse_loss(model_output["x_hat"], gt_image)
        return loss, {"loss_mse": loss}


@nip
class CommitLoss(nn.Module):

    def __init__(self):
        super(CommitLoss, self).__init__()

    def forward(self, model_output: Dict[str, torch.Tensor], gt_image: torch.Tensor) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss = model_output["loss_commit"]
        return loss, {"loss_commit": loss}


@nip
class CrossEntropyRateLoss(nn.Module):

    def __init__(self, names: Tuple[str, ...] = ("top", "bottom"), weights: Optional[Dict[str, float]] = None):
        super(CrossEntropyRateLoss, self).__init__()
        self._names = names
        if weights is None:
            weights = {name: 1. for name in names}
        self._weights = weights

    def forward(self, model_output: Dict[str, torch.Tensor], gt_image: torch.Tensor) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict = {}
        total_loss = 0.

        for name, weight in self._weights.items():
            probs_raw_key = f"y_{name}_probs_raw"
            if probs_raw_key in model_output:
                loss = F.cross_entropy(model_output[probs_raw_key], model_output[f"y_{name}_indices"])
                total_loss = total_loss + loss * weight
                loss_dict[f"loss_ce_{name}"] = loss
        loss_dict["loss_ce"] = total_loss

        return total_loss, loss_dict


@nip
class CompositeLoss(nn.Module):

    def __init__(self, components: List[Tuple[nn.Module, float]]):
        super(CompositeLoss, self).__init__()
        self._components = components

    def forward(self, model_output: Dict[str, torch.Tensor], gt_image: torch.Tensor):
        loss_dict = {}
        total_loss = 0.

        for (criterion, weight) in self._components:
            loss, loss_local_dict = criterion.forward(model_output, gt_image)
            total_loss = total_loss + weight * loss
            loss_dict.update(loss_local_dict)

        loss_dict["loss"] = total_loss

        return loss_dict


# @nip
# class RateDistortionLoss(nn.Module):
#
#     def __init__(self, lmbda: float, commit_weight: float):
#         super(RateDistortionLoss, self).__init__()
#         self._lmbda = lmbda
#         self._commit_weight = commit_weight
#
#     def forward(self, model_output: Dict[str, torch.Tensor], gt_image: torch.Tensor) -> Dict[str, torch.Tensor]:
#         assert "x_hat" in model_output
#         x_hat = model_output["x_hat"]
#         N, _, H, W = x_hat.size()
#         num_pixels = N * H * W
#         mse = F.mse_loss(x_hat, gt_image)
#
#         losses = {"loss_distortion": mse}
#         total_loss = self._lmbda * mse
#
#         if "loss_commit" in model_output:
#             total_loss = total_loss + self._commit_weight * model_output["loss_commit"]
#             losses["loss_commit"] = model_output["loss_commit"]
#
#         bpp_loss = 0.
#         for k, v in model_output.items():
#             if k.endswith("_likelihoods"):
#                 bpp_loss = bpp_loss + (torch.log(v).sum() / (-math.log(2) * num_pixels))
#         total_loss += bpp_loss
#         losses["loss_bpp"] = bpp_loss
#
#         losses["loss"] = total_loss
#
#         return losses
#
#
