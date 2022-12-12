import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, Tuple
from nip import nip


@nip
class TransformerAREntropyModel(nn.Module):

    MODE_VECTORS = "vectors"
    MODE_INDICES = "indices"

    _REDUCTION_MEAN = "mean"
    _REDUCTION_SUM = "sum"

    def __init__(self,
                 input_mode: str,
                 code_book_size: int,
                 block_size: Union[int, Tuple[int, int]],
                 depth: int,
                 num_heads: int,
                 input_dim: Optional[int] = None,
                 embedding_dim: Optional[int] = None,
                 dropout: float = 0.,
                 activation: str = "relu",
                 norm_first: bool = False,
                 feedforward_dim: int = 2048,
                 reduction: str = "mean"):
        assert input_mode in (TransformerAREntropyModel.MODE_VECTORS, TransformerAREntropyModel.MODE_INDICES)
        assert reduction in (TransformerAREntropyModel._REDUCTION_MEAN, TransformerAREntropyModel._REDUCTION_SUM)
        if input_mode == TransformerAREntropyModel.MODE_VECTORS:
            assert input_dim is not None
        else:
            assert embedding_dim is not None
            input_dim = code_book_size
        super(TransformerAREntropyModel, self).__init__()

        self._input_mode = input_mode
        self._code_book_size = code_book_size

        self._block_size = (block_size, block_size) if isinstance(block_size, int) else block_size
        block_len = self._block_size[0] * self._block_size[1]
        self._mask = torch.Tensor([1 if i <= block_len // 2 else 0 for i in range(block_len)]).view(1, block_len, 1)
        self._block_len = block_len
        self._reduction = reduction

        if embedding_dim is not None:
            self._embedding_layer = nn.Linear(input_dim, embedding_dim)
        else:
            self._embedding_layer = nn.Identity()
            embedding_dim = input_dim
        self._unembedding_layer = nn.Linear(embedding_dim, code_book_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                   nhead=num_heads,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   norm_first=norm_first,
                                                   dim_feedforward=feedforward_dim,
                                                   batch_first=True)
        self._transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                          num_layers=depth)

    @property
    def input_mode(self) -> str:
        return self._input_mode

    def forward(self, x: torch.Tensor, output_softmax: bool = False) -> torch.Tensor:
        # x: (B, C, H, W) or (B, H, W)
        if self._input_mode == TransformerAREntropyModel.MODE_INDICES:
            x = F.one_hot(x, num_classes=self._code_book_size)
            x = x.permute(0, 3, 1, 2).float()  # B, C, H, W

        B, C, H, W = x.shape
        x_unfold = F.unfold(x, kernel_size=self._block_size, padding=(self._block_size[0] // 2,
                                                                      self._block_size[1] // 2))
        x_unfold = x_unfold.reshape(B, C, self._block_len, H * W).contiguous()\
            .view(-1, self._block_len, C)  # B*H*W, block_len, C
        x_unfold = self._embedding_layer.forward(x_unfold)
        x_unfold = x_unfold * self._mask.clone().to(x_unfold.device)

        x_out = self._transformer_encoder.forward(x_unfold)
        if self._reduction == TransformerAREntropyModel._REDUCTION_MEAN:
            x_out = torch.mean(x_out, dim=1)  # B*H*W, 1, C
        else:
            x_out = torch.sum(x_out, dim=1)  # B*H*W, 1, C
        x_out = x_out.reshape(B, H*W, -1)

        x_out = self._unembedding_layer.forward(x_out)
        x_out = x_out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if output_softmax:
            x_out = torch.softmax(x_out, dim=1)

        return x_out
