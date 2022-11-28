from typing import Dict

import torch
import torch.nn as nn

from vector_quantize_pytorch import VectorQuantize
from nip import nip


@nip
class VanillaAE(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module):
        super(VanillaAE, self).__init__()
        with torch.no_grad():
            test_image = torch.zeros((1, 3, 256, 256))
            test_latent = encoder.forward(test_image)
            rec_test_image = decoder.forward(test_latent)
            assert test_image.shape == rec_test_image.shape

        self._encoder = encoder
        self._decoder = decoder
        self._latent_channels = test_latent.shape[1]

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"y": self._encoder.forward(x)}

    def decode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"x_hat": self._decoder.forward(x)}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = {}
        result.update(self.encode(x))
        result.update(self.decode(result["y"]))
        return result


@nip
class VectorQuantizedAE(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 code_book_size: int,
                 vq_ema_decay: float = 0.8,
                 vq_commit_weight: float = 1.):
        super(VectorQuantizedAE, self).__init__()
        with torch.no_grad():
            test_image = torch.zeros((1, 3, 256, 256))
            test_latent = encoder.forward(test_image)
            rec_test_image = decoder.forward(test_latent)
            assert test_image.shape == rec_test_image.shape

        self._encoder = encoder
        self._decoder = decoder
        self._latent_channels = test_latent.shape[1]
        self._code_book_size = code_book_size
        self._vq = VectorQuantize(dim=self._latent_channels,
                                  codebook_size=code_book_size,
                                  decay=vq_ema_decay,
                                  commitment_weight=vq_commit_weight)

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    @property
    def code_book_size(self) -> int:
        return self._code_book_size

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self._encoder.forward(x)
        B, C, H, W = y.shape
        y = y.reshape((B, C, H * W))
        y = y.transpose(1, 2)
        y, indices, loss_commit = self._vq(y)
        y = y.transpose(1, 2)
        y = y.reshape((B, C, H, W))
        return {"y": y,
                "y_indices": indices,
                "loss_commit": loss_commit}

    def decode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"x_hat": self._decoder.forward(x)}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = {}
        result.update(self.encode(x))
        result.update(self.decode(result["y"]))
        return result
