import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import cv2

from typing import Optional, Dict
from neptune.new.types import File as NeptuneFile
from pytorch_lightning.loggers import NeptuneLogger
from nip import nip


@nip
class LitAutoEncoderModule(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss: nn.Module,
                 sample_train: Optional[torch.Tensor] = None,
                 sample_val: Optional[torch.Tensor] = None):
        super(LitAutoEncoderModule, self).__init__()
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._samples = {"train": sample_train, "val": sample_val}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._model.forward(x)

    def configure_optimizers(self):
        return self._optimizer

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch_x, batch_idx):
        return self._common_step(batch_x, "val")

    def on_train_epoch_end(self):
        self._common_epoch_end("train")

    def on_validation_epoch_end(self):
        self._common_epoch_end("val")

    def _common_step(self, batch_x, phase: str):
        batch_size = batch_x.shape[0]
        model_output = self.forward(batch_x)
        loss_output = self._loss.forward(model_output, batch_x)
        loss = loss_output["loss"] if isinstance(loss_output, dict) else loss_output
        self._log_loss(loss_output, phase, batch_size)
        return loss

    def _log_loss(self, loss, phase: str, batch_size: int):
        if isinstance(loss, dict):
            for name, value in loss.items():
                self.log(f"{phase}_{name}", value, on_epoch=True, batch_size=batch_size)
        else:
            self.log(f"{phase}_loss", loss, on_epoch=True, batch_size=batch_size)

    def _common_epoch_end(self, phase: str):
        if phase not in self._samples or self._samples[phase] is None or type(self.logger) != NeptuneLogger:
            return

        gt_image = self._samples[phase]
        model_output = self.forward(gt_image.unsqueeze(0).to(self.device))

        bpp = LitAutoEncoderModule._estimate_bpp(model_output)

        rec_image = model_output["x_hat"][0]

        gt_image = LitAutoEncoderModule._tensor_to_neptune_image(gt_image)
        rec_image = LitAutoEncoderModule._tensor_to_neptune_image(rec_image)
        gap = np.ones((rec_image.shape[0], 70, 3), dtype=rec_image.dtype)
        paired_image = np.concatenate([gt_image, gap, rec_image], axis=1)

        gt_image_cv2 = LitAutoEncoderModule._neptune_image_to_cv2(gt_image)
        rec_image_cv2 = LitAutoEncoderModule._neptune_image_to_cv2(rec_image)
        psnr = cv2.PSNR(gt_image_cv2, rec_image_cv2)

        self.log(f"{phase}_sample_psnr", psnr)
        self.log(f"{phase}_sample_bpp", bpp)
        self.logger.experiment[f"{phase}_sample_image"].log(NeptuneFile.as_image(paired_image))

    @staticmethod
    def _estimate_bpp(model_output: Dict[str, torch.Tensor]) -> float:
        num_pixels = model_output["x_hat"].shape[2] * model_output["x_hat"].shape[3]
        bpp = 0.
        need_raw_probs = True

        for k, v in model_output.items():
            if k.endswith("_likelihoods"):
                bpp = bpp + (torch.log(v).sum() / (-math.log(2) * num_pixels))
                need_raw_probs = False
        if not need_raw_probs:
            return bpp.item()

        for name in ("top", "bottom"):
            if f"y_{name}_probs_raw" not in model_output:
                continue
            probs = model_output[f"y_{name}_probs_raw"][0]
            probs = torch.softmax(probs, dim=0)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                return bpp

            log_prob = 0
            for h in range(probs.shape[1]):
                for w in range(probs.shape[2]):
                    prob = probs[model_output[f"y_{name}_indices"][0, h, w].item(), h, w]
                    log_prob = log_prob + torch.log(prob)
            bpp = bpp + log_prob / (-math.log(2) * num_pixels)

        return bpp

    @staticmethod
    def _tensor_to_neptune_image(image: torch.Tensor) -> np.ndarray:
        return image.clone().detach().cpu().numpy().transpose(1, 2, 0).clip(0., 1.)

    @staticmethod
    def _neptune_image_to_cv2(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
