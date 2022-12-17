import math
import torch
import torch.nn.functional as F
import fire
import nip
import piq
import numpy as np
import cv2

from typing import Dict
from pathlib import Path
from tqdm import tqdm
from lib.lightning_modules import LitAutoEncoderModule
from lib.models import ConvScaleHyperpriorModel
from lib.datasets import VimeoImagesDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, CenterCrop
from tqdm import tqdm
from PIL import Image


def _open_image(image_path: Path):
    with Image.open(str(image_path)) as img:
        to_tensor = ToTensor()
        gt_image = to_tensor(img)
    assert gt_image.shape[1] % 256 == 0 and gt_image.shape[2] % 256 == 0, "Only images with sieds of multiple of 256 are supported"
    return gt_image


def _patchify_image(image: torch.Tensor, patch_size):
    image = image.unsqueeze(0)
    patches = F.unfold(image,
                       kernel_size=patch_size,
                       stride=patch_size)
    patches = patches.transpose(1, 2)
    patches = patches.reshape(
        patches.shape[0], patches.shape[1], 3, patch_size[0], patch_size[1]).squeeze(0)
    return patches


def _unpatchify_image(patches: torch.Tensor, output_size):
    patch_size = (patches.shape[2], patches.shape[3])
    patches = patches.unsqueeze(0)
    patches = patches.reshape(
        patches.shape[0], patches.shape[1], 3 * patch_size[0] * patch_size[1])
    patches = patches.transpose(1, 2)
    image = F.fold(patches,
                   output_size=output_size,
                   kernel_size=patch_size,
                   stride=patch_size)
    image = image.squeeze(0)
    return image


def _estimate_bpp(model_output: Dict[str, torch.Tensor], num_pixels: int) -> float:
    num_pixels = num_pixels or 1
    bpp = 0.
    need_raw_probs = True

    for k, v in model_output.items():
        if k.endswith("_likelihoods"):
            bpp = bpp + (torch.log(v).sum() / (-math.log(2) * num_pixels))
            need_raw_probs = False
    if not need_raw_probs:
        return bpp.item()

    for name in ("top", "bottom"):
        probs = model_output[f"y_{name}_probs_raw"]
        probs = torch.softmax(probs, dim=1)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            raise ValueError()

        indices = F.one_hot(
            model_output[f"y_{name}_indices"], probs.shape[1]).permute(0, 3, 1, 2)
        log_prob = torch.zeros(
            (probs.shape[0], 2, probs.shape[2], probs.shape[3])).to(probs.device)
        log_prob.scatter_(1, indices, probs)
        log_prob = log_prob[:, 1, :, :]
        log_prob = torch.log(log_prob)
        log_prob = torch.sum(torch.sum(log_prob, dim=1), dim=1)

        bpp = bpp + log_prob / (-math.log(2) * num_pixels)

    return bpp.sum().item()


def _process_kodak(pl_module: LitAutoEncoderModule,
                   checkpoint_path: Path,
                   image_path: Path,
                   device: str):
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    if isinstance(pl_module._model, ConvScaleHyperpriorModel):
        # Some CompressAI hack
        pl_module._model.update(force=True)
    pl_module.load_state_dict(checkpoint['state_dict'])
    pl_module = pl_module.to(device)
    pl_module.eval()

    gt_image = _open_image(image_path)
    h = gt_image.shape[1]
    w = gt_image.shape[2]
    gt_patches = _patchify_image(gt_image, (256, 256))

    with torch.no_grad():
        model_output = pl_module.forward(gt_patches.to(device))

    bpp = _estimate_bpp(model_output, (h * w))

    rec_image = _unpatchify_image(
        model_output["x_hat"], (gt_image.shape[1], gt_image.shape[2]))
    psnr = piq.psnr(rec_image.clamp(0., 1.).unsqueeze(0),
                    gt_image.unsqueeze(0).to(device)).item()

    print(f"BPP: {bpp}")
    print(f"PSNR: {psnr}")

    rec_image = rec_image.clone().detach().cpu().numpy()
    rec_image = (255 * rec_image).transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    rec_image = cv2.cvtColor(rec_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"{str(image_path.name.split('.')[0])}_rec.png", rec_image)


def main(model_path: str,
         image_path: str,
         device: str = "cuda:5"):
    model_path = Path(model_path)
    image_path = Path(image_path)

    for experiment in nip.load(str(model_path / "config.nip"), always_iter=True):
        _process_kodak(experiment["pl_module"],
                       model_path / "checkpoint.ckpt",
                       image_path,
                       device)


if __name__ == '__main__':
    fire.Fire(main)
