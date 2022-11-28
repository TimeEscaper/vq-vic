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
