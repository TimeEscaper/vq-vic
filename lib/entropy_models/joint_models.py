import torch
import torch.nn as nn
import torchac

from typing import Dict, Optional, Any
from tqdm import tqdm
from nip import nip
from .transformer_models import TransformerAREntropyModel
from .pixel_snail import PixelSNAIL


@nip
class TopBottomTransformerAREntropyModel(nn.Module):

    def __init__(self,
                 top_model: TransformerAREntropyModel,
                 bottom_model: TransformerAREntropyModel,
                 ignore: Optional[str] = None):
        super(TopBottomTransformerAREntropyModel, self).__init__()
        self._top_model = top_model
        self._bottom_model = bottom_model
        self._ignore = ignore

    def top_parameters(self):
        return self._top_model.parameters()

    def bottom_parameters(self):
        return self._bottom_model.parameters()

    def forward(self, encoder_output: Dict[str, torch.Tensor], output_softmax: bool = False) -> Dict[str, torch.Tensor]:
        output_dict = {}

        if self._ignore != "top":
            TopBottomTransformerAREntropyModel._forward_single(self._top_model, "top", encoder_output, output_dict,
                                                               output_softmax)
        if self._ignore != "bottom":
            TopBottomTransformerAREntropyModel._forward_single(self._bottom_model, "bottom", encoder_output, output_dict,
                                                               output_softmax, condition=encoder_output["y_top"])

        return output_dict

    @staticmethod
    def _forward_single(model: TransformerAREntropyModel, name: str, encoder_output: Dict[str, torch.Tensor],
                        output_dict: Dict[str, torch.Tensor], output_softmax: bool, condition = None) -> None:
        in_key = f"y_{name}" if model.input_mode == TransformerAREntropyModel.MODE_VECTORS else f"y_{name}_indices"
        out_key = f"y_{name}_probs" if output_softmax else f"y_{name}_probs_raw"
        output_dict[out_key] = model.forward(encoder_output[in_key], output_softmax=output_softmax, condition=condition)


@nip
class TopBottomPixelSNAIL(nn.Module):

    def __init__(self,
                 top_model: PixelSNAIL,
                 bottom_model: PixelSNAIL,
                 ignore: Optional[str] = None):
        super(TopBottomPixelSNAIL, self).__init__()
        self._top_model = top_model
        self._bottom_model = bottom_model
        self._ignore = ignore

    def top_parameters(self):
        return self._top_model.parameters()

    def bottom_parameters(self):
        return self._bottom_model.parameters()

    def forward(self, encoder_output: Dict[str, torch.Tensor], output_softmax: bool = False) -> Dict[str, torch.Tensor]:
        output_dict = {}
        top = TopBottomPixelSNAIL._forward_single(self._top_model, "top", encoder_output, output_dict,
                                                  output_softmax)
        if self._ignore != "bottom":
            _ = TopBottomPixelSNAIL._forward_single(self._bottom_model, "bottom", encoder_output, output_dict,
                                                    output_softmax, condition=top)
        return output_dict

    def compress(self, encoder_output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        output_dict = {}

        top = TopBottomPixelSNAIL._compress_single(self._top_model, "top", encoder_output, output_dict, 
                                                   condition=None)
        _ = TopBottomPixelSNAIL._compress_single(self._bottom_model, "bottom", encoder_output, output_dict, 
                                                 condition=top)
        
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

    @staticmethod
    def _compress_single(model: PixelSNAIL, name: str, encoder_output: Dict[str, torch.Tensor],
                         output_dict: Dict[str, torch.Tensor], condition: Optional[torch.Tensor] = None):
        latent = encoder_output[f"y_{name}" if model.input_mode == "vectors" else f"y_{name}_indices"]
        if len(latent.shape) == 4:
            B, C, H, W = latent.shape
            row = torch.zeros(B, C, H, W).to(latent.device)
        else:
            B, H, W = latent.shape
            row = torch.zeros(B, H, W, dtype=torch.int64).to(latent.device)

        probs = torch.zeros(B, model.n_class, H, W).to(latent.device)
        cache = {}
        
        for i in tqdm(range(H), leave=False):
            for j in tqdm(range(W), leave=False):
                if len(latent.shape) == 4:
                    prob, cache = model.forward(row[:, :, : i + 1, :], condition=condition, cache=cache,
                                                output_softmax=True)
                    row[:, :, i, j] = latent[:, :, i, j]
                else:
                    prob, cache = model.forward(row[:, : i + 1, :], condition=condition, cache=cache,
                                                output_softmax=True)
                    row[:, i, j] = latent[:, i, j]
                prob = prob[:, :, i, j]
                probs[:, :, i, j] = prob
                                        
        latent_indices = encoder_output[f"y_{name}_indices"]
        cdf = TopBottomPixelSNAIL._probs_to_cdf(probs)
        latent_int = latent_indices.clone().detach().cpu().unsqueeze(1).to(torch.int16)
        byte_stream = torchac.encode_float_cdf(cdf, latent_int, check_input_bounds=True)

        output_dict[f"y_{name}_strings"] = byte_stream # encoder.get_encoded()
        output_dict[f"y_{name}_probs"] = probs
        
        return latent

    @staticmethod
    def _probs_to_cdf(probs: torch.Tensor) -> torch.Tensor:
        pmf = probs.clone().detach().cpu().permute(0, 2, 3, 1).unsqueeze(1)
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)
        return cdf_with_0
        
        