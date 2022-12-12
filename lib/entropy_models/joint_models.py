import torch
import torch.nn as nn

from typing import Dict, Optional
from nip import nip
from .transformer_models import TransformerAREntropyModel
from .pixel_snail import PixelSNAIL


@nip
class TopBottomTransformerAREntropyModel(nn.Module):

    def __init__(self,
                 top_model: TransformerAREntropyModel,
                 bottom_model: TransformerAREntropyModel):
        super(TopBottomTransformerAREntropyModel, self).__init__()
        self._top_model = top_model
        self._bottom_model = bottom_model

    def top_parameters(self):
        return self._top_model.parameters()

    def bottom_parameters(self):
        return self._bottom_model.parameters()

    def forward(self, encoder_output: Dict[str, torch.Tensor], output_softmax: bool = False) -> Dict[str, torch.Tensor]:
        output_dict = {}

        TopBottomTransformerAREntropyModel._forward_single(self._top_model, "top", encoder_output, output_dict,
                                                           output_softmax)
        TopBottomTransformerAREntropyModel._forward_single(self._bottom_model, "bottom", encoder_output, output_dict,
                                                           output_softmax)

        return output_dict

    @staticmethod
    def _forward_single(model: TransformerAREntropyModel, name: str, encoder_output: Dict[str, torch.Tensor],
                        output_dict: Dict[str, torch.Tensor], output_softmax: bool) -> None:
        in_key = f"y_{name}" if model.input_mode == TransformerAREntropyModel.MODE_VECTORS else f"y_{name}_indices"
        out_key = f"y_{name}_probs" if output_softmax else f"y_{name}_probs_raw"
        output_dict[out_key] = model.forward(encoder_output[in_key], output_softmax=output_softmax)


@nip
class TopBottomPixelSNAIL(nn.Module):

    def __init__(self,
                 top_model: PixelSNAIL,
                 bottom_model: PixelSNAIL):
        super(TopBottomPixelSNAIL, self).__init__()
        self._top_model = top_model
        self._bottom_model = bottom_model

    def forward(self, encoder_output: Dict[str, torch.Tensor], output_softmax: bool = False) -> Dict[str, torch.Tensor]:
        output_dict = {}
        top = TopBottomPixelSNAIL._forward_single(self._top_model, "top", encoder_output, output_dict,
                                                  output_softmax)
        _ = TopBottomPixelSNAIL._forward_single(self._bottom_model, "bottom", encoder_output, output_dict,
                                                output_softmax, condition=top)
        return output_dict

    @staticmethod
    def _forward_single(model: PixelSNAIL, name: str, encoder_output: Dict[str, torch.Tensor],
                        output_dict: Dict[str, torch.Tensor], output_softmax: bool,
                        condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        in_key = f"y_{name}" if model.input_mode == "vectors" else f"y_{name}_indices"
        out_key = f"y_{name}_probs" if output_softmax else f"y_{name}_probs_raw"
        output_dict[out_key] = model.forward(encoder_output[in_key], output_softmax=output_softmax,
                                             condition=condition)[0]
        return encoder_output[in_key]
