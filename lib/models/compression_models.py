import torch
import torch.nn as nn

from compressai.models import CompressionModel
from nip import nip
from .auto_encoder import VectorQuantizedAE2


@nip
class VQVAE2CompressionModel(CompressionModel):

    def __init__(self,
                 vq_ae: VectorQuantizedAE2):
        super(VQVAE2CompressionModel, self).__init__(entropy_bottleneck_channels=1)
        self._vq_ae = vq_ae
        self._bottom_extractor = nn.Conv2d(in_channels=vq_ae.code_book_dim,
                                           out_channels=1,
                                           kernel_size=3,
                                           padding=1)
        self._top_extractor = nn.Conv2d(in_channels=vq_ae.code_book_dim,
                                        out_channels=1,
                                        kernel_size=3,
                                        padding=1)

    def get_params(self):
        return set(p for n, p in self.named_parameters() if not n.endswith(".quantiles"))

    def get_aux_params(self):
        return set(p for n, p in self.named_parameters() if n.endswith(".quantiles"))

    def get_aux_loss(self):
        return self.aux_loss()

    def forward(self, x: torch.Tensor):
        output = self._vq_ae.encode(x)

        y_bottom_ext = self._bottom_extractor.forward(output["y_bottom"])
        y_top_ext = self._top_extractor.forward(output["y_top"])

        _, y_bottom_likelihoods = self.entropy_bottleneck(y_bottom_ext)
        _, y_top_likelihoods = self.entropy_bottleneck(y_top_ext)

        output["y_bottom_likelihoods"] = y_bottom_likelihoods
        output["y_top_likelihoods"] = y_top_likelihoods

        dec = self._vq_ae.decode(output)
        output.update(dec)

        return output
