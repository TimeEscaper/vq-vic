import math
import torch
import torch.nn.functional as F
import fire
import nip
import piq
import zlib
import numpy as np

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


def _estimate_zlib_bits(y_hat: torch.Tensor) -> float:
    y_hat = y_hat.clone().detach().cpu().numpy().astype(np.uint16).copy(order='C')
    data_bytes = zlib.compress(y_hat, level=9)
    bits = len(data_bytes) * 8
    return bits


def _open_image(image_path: Path):
    with Image.open(str(image_path)) as img:
        to_tensor = Compose([CenterCrop((256, 256)), ToTensor()])
        gt_image = to_tensor(img)
    return gt_image


def _patchify_image(image: torch.Tensor, patch_size):
    image = image.unsqueeze(0)
    patches = F.unfold(image, 
                       kernel_size = patch_size,
                       stride=patch_size)
    patches = patches.transpose(1, 2)
    patches = patches.reshape(patches.shape[0], patches.shape[1], 3, patch_size[0], patch_size[1]).squeeze(0)
    return patches


def _unpatchify_image(patches: torch.Tensor, output_size):
    patch_size = (patches.shape[2], patches.shape[3])
    patches = patches.unsqueeze(0)
    patches = patches.reshape(patches.shape[0], patches.shape[1], 3 * patch_size[0] * patch_size[1])
    patches = patches.transpose(1, 2)
    image = F.fold(patches,
                   output_size=output_size,
                   kernel_size = patch_size,
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

        indices = F.one_hot(model_output[f"y_{name}_indices"], probs.shape[1]).permute(0, 3, 1, 2)
        log_prob = torch.zeros((probs.shape[0], 2, probs.shape[2], probs.shape[3])).to(probs.device)
        log_prob.scatter_(1, indices, probs)
        log_prob = log_prob[:, 1, :, :]
        log_prob = torch.log(log_prob)
        log_prob = torch.sum(torch.sum(log_prob, dim=1), dim=1)

        bpp = bpp + log_prob / (-math.log(2) * num_pixels)

    return bpp.sum().item()


def _process_vimeo(pl_module: LitAutoEncoderModule,
             checkpoint_path: Path,
             dataset_path: Path,
             batch_size: int,
             device: str):
    assert batch_size is not None
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    if hasattr(pl_module._model, "update"):
        # Some CompressAI hack
        pl_module._model.update(force=True)
    pl_module.load_state_dict(checkpoint['state_dict'])
    pl_module = pl_module.to(device)
    pl_module.eval()

    dataset = VimeoImagesDataset(sequences_root=dataset_path / "sequences",
                                 subset_list=dataset_path / "sep_testlist.txt",
                                 transform=Compose([CenterCrop((256, 256)), ToTensor()]))
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=12)

    cumulative_bpp = 0.
    cumulative_psnr = 0.
    n_images = 0

    for X_batch in tqdm(loader):
        with torch.no_grad():
            X_batch = X_batch.to(device)
            model_output = pl_module.forward(X_batch)

            num_pixels = X_batch.shape[2] * X_batch.shape[3]
            cumulative_bpp += _estimate_bpp(model_output, num_pixels)
            cumulative_psnr += piq.psnr(model_output["x_hat"].clamp(0., 1.),
                                        X_batch, reduction="sum").item()
            n_images += X_batch.shape[0]

    average_bpp = cumulative_bpp / n_images
    average_psnr = cumulative_psnr / n_images

    print(f"Average BPP: {average_bpp}")
    print(f"Average PSNR: {average_psnr}")
    print(f"Processed images: {n_images}")


def _process_kodak(pl_module: LitAutoEncoderModule,
             checkpoint_path: Path,
             dataset_path: Path,
             device: str):
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    if isinstance(pl_module._model, ConvScaleHyperpriorModel):
        # Some CompressAI hack
        pl_module._model.update(force=True)
    pl_module.load_state_dict(checkpoint['state_dict'])
    pl_module = pl_module.to(device)
    pl_module.eval()
    
    cumulative_bpp = 0.
    cumulative_psnr = 0.
    n_images = 0
    
    zlib_bpp_cumulative = 0.
    
    for image_path in tqdm(dataset_path.glob("kodim*.png")):
        if not image_path.is_file():
            return
        
        gt_image = _open_image(image_path)
        h = gt_image.shape[1]
        w = gt_image.shape[2]
        gt_patches = _patchify_image(gt_image, (256, 256))
        
        with torch.no_grad():
            model_output = pl_module.forward(gt_patches.to(device))
        
        cumulative_bpp += _estimate_bpp(model_output, None) / (h * w)
        
        rec_image = _unpatchify_image(model_output["x_hat"], (gt_image.shape[1], gt_image.shape[2]))
        cumulative_psnr += piq.psnr(rec_image.clamp(0., 1.).unsqueeze(0),
                                    gt_image.unsqueeze(0).to(device)).item()
        
        n_images += 1
        
        zlib_bpp = _estimate_zlib_bits(model_output["y_top_indices"]) + _estimate_zlib_bits(model_output["y_bottom_indices"])
        zlib_bpp = zlib_bpp / (h * w)
        zlib_bpp_cumulative += zlib_bpp
        
    average_bpp = cumulative_bpp / n_images
    average_psnr = cumulative_psnr / n_images
    average_zlib = zlib_bpp_cumulative / n_images

    print(f"Average BPP: {average_bpp}")
    print(f"Average BPP (zlib): {average_zlib}")
    print(f"Average PSNR: {average_psnr}")
    print(f"Processed images: {n_images}")


def main(model_path: str,
         dataset_type: str,
         dataset_path: str,
         batch_size: int = 12,
         device: str = "cuda"):
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)

    for experiment in nip.load(str(model_path / "config.nip"), always_iter=True):
        if dataset_type == "vimeo":
            _process_vimeo(experiment["pl_module"],
                    model_path / "checkpoint.ckpt",
                    dataset_path,
                    batch_size,
                    device)
        elif dataset_type == "kodak":
            _process_kodak(experiment["pl_module"],
                    model_path / "checkpoint.ckpt",
                    dataset_path,
                    device)


if __name__ == '__main__':
    fire.Fire(main)
