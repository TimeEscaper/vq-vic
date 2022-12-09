from typing import Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

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


class _Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super(_Quantize, self).__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

@nip
class VectorQuantizedAE2(nn.Module):
    def __init__(
            self,
            encoder_bottom: nn.Module,
            decoder_bottom: nn.Module,
            encoder_top: nn.Module,
            decoder_top: nn.Module,
            code_book_dim: int = 64,
            code_book_size: int = 512,
            decay: float = 0.99,


        # in_channel=3,
        # channel=128,
        # n_res_block=2,
        # n_res_channel=32,
        # embed_dim=64,
        # n_embed=512,
    ):
        super(VectorQuantizedAE2, self).__init__()

        self.enc_b = encoder_bottom  # Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = encoder_top # Encoder(channel, channel, n_res_block, n_res_channel, stride=2)

        test_tensor = torch.zeros((1, 3, 256, 256))
        with torch.no_grad():
            test_tensor = self.enc_b.forward(test_tensor)
            channels_bottom = test_tensor.shape[1]
            test_tensor = self.enc_t.forward(test_tensor)
            channels_top = test_tensor.shape[1]

        self.quantize_conv_t = nn.Conv2d(channels_top, code_book_dim, 1)
        self.quantize_t = _Quantize(code_book_dim, code_book_size, decay=decay)

        # self.dec_t = Decoder(
        #     embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        # )
        self.dec_t = decoder_top

        self.quantize_conv_b = nn.Conv2d(code_book_dim + channels_bottom, code_book_dim, 1)
        self.quantize_b = _Quantize(code_book_dim, code_book_size, decay=decay)
        self.upsample_t = nn.ConvTranspose2d(code_book_dim, code_book_dim, 4, stride=2, padding=1)

        self.dec = decoder_bottom
        # self.dec = Decoder(
        #     embed_dim + embed_dim,
        #     in_channel,
        #     channel,
        #     n_res_block,
        #     n_res_channel,
        #     stride=4,
        # )

    def forward(self, input):
        output = self.encode(input)
        dec = self.decode(output["y_top"], output["y_bottom"])
        output.update(dec)
        return output

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return {"y_top": quant_t,
                "y_bottom": quant_b,
                "y_top_indices": id_t,
                "y_bottom_indices": id_b,
                "loss_commit": diff_t + diff_b}

        # return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return {"x_hat": dec}

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
