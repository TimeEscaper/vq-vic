import torch
import torch.nn as nn

from typing import Dict
from compressai.models import CompressionModel
from nip import nip
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
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

    def compress(self, x: torch.Tensor):
        output = self._vq_ae.encode(x)
        output.update(self._entropy_model.compress(output))
        return output
    
    def update(self):
        pass


@nip
class ConvScaleHyperpriorModel(CompressionModel):
    
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 hyper_encoder: nn.Module,
                 hyper_decoder: nn.Module):
        super(ConvScaleHyperpriorModel, self).__init__(entropy_bottleneck_channels=hyper_encoder.out_channels)
        self._encoder = encoder
        self._decoder = decoder
        self._hyper_encoder = hyper_encoder
        self._hyper_decoder = hyper_decoder
        
        self._gaussian_conditional = GaussianConditional(None)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self._encoder.forward(x)
        z = self._hyper_encoder.forward(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck.forward(z)
        scales_hat = self._hyper_decoder.forward(z_hat)
        y_hat, y_likelihoods = self._gaussian_conditional.forward(y, scales_hat)
        x_hat = self._decoder.forward(y_hat)
        
        return {
            "y": y, "z": z, "y_hat": y_hat, "z_hat": z_hat, "x_hat": x_hat,
            "y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods,
            "loss_aux": self.entropy_bottleneck.loss()
        }

    def main_parameters(self):
        return set(p for n, p in self.named_parameters() if not n.endswith(".quantiles"))
    
    def aux_parameters(self):
        return set(p for n, p in self.named_parameters() if n.endswith(".quantiles"))
