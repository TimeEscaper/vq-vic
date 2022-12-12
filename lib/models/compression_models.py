import torch
import torch.nn as nn

from compressai.models import CompressionModel
from nip import nip
from lib.entropy_models import TopBottomTransformerAREntropyModel
from .auto_encoder import VectorQuantizedAE2


@nip
class VQVAE2CompressionModel(nn.Module):

    def __init__(self,
                 vq_ae: VectorQuantizedAE2,
                 entropy_model: TopBottomTransformerAREntropyModel):
        super(VQVAE2CompressionModel, self).__init__()
        self._vq_ae = vq_ae
        self._entropy_model = entropy_model

    def forward(self, x: torch.Tensor):
        output = self._vq_ae.encode(x)
        entropy_output = self._entropy_model.forward(output)
        output.update(entropy_output)
        dec = self._vq_ae.decode(output)
        output.update(dec)
        return output
