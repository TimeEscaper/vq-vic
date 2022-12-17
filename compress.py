import math
import torch
import torch.nn.functional as F
import fire
import nip
import cv2
import numpy as np
import zlib

from pathlib import Path
from tqdm import tqdm
from lib.lightning_modules import LitAutoEncoderModule
from PIL import Image
from torchvision.transforms import ToTensor, Compose, CenterCrop
from einops import rearrange


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def _estimate_bits(y_hat: torch.Tensor) -> float:
    y_hat = y_hat.clone().detach().cpu().numpy().copy(order='C')
    data_bytes = zlib.compress(y_hat, level=9)
    bits = len(data_bytes) * 8
    return bits


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


def _estimate_bpp(model_output, num_pixels: int) -> float:
    bpp = 0.
    for name in ("top", "bottom"):
        probs = model_output[f"y_{name}_probs"]
        indices = model_output[f"y_{name}_indices"]
        
        log_prob = 0.
        for b in range(probs.shape[0]):
            for h in range(probs.shape[2]):
                for w in range(probs.shape[3]):
                    prob = probs[b, indices[b, h, w].item(), h, w]
                    log_prob = log_prob + torch.log(prob)
        bpp = bpp + log_prob / (-math.log(2) * num_pixels)
    return bpp.item()


def _calculate_true_bpp(model_output, num_pixels: int) -> float:
    return (len(model_output["y_top_strings"])* 8 + len(model_output["y_bottom_strings"])* 8) / num_pixels
    # return (model_output["y_top_strings"] + model_output["y_bottom_strings"]) / num_pixels


def _process(pl_module: LitAutoEncoderModule,
             checkpoint_path: Path,
             input_path: Path,
             output_path: Path):
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    pl_module.load_state_dict(checkpoint['state_dict'])
    pl_module = pl_module.to(DEVICE)
    pl_module.eval()

    with Image.open(str(input_path)) as img:
        # to_tensor = Compose([CenterCrop((256, 256)), ToTensor()])
        to_tensor = ToTensor()
        gt_image = to_tensor(img)
    gt_patches = _patchify_image(gt_image, (256, 256))

    with torch.no_grad():
        forward_output = pl_module.forward(gt_patches.to(DEVICE))
        compress_out = pl_module.compress(gt_patches.to(DEVICE))
        
    x_hat = forward_output["x_hat"].clone().detach().cpu()
    x_hat = _unpatchify_image(x_hat, (gt_image.shape[1], gt_image.shape[2])).numpy()
    x_hat = (255 * x_hat).transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    x_hat = cv2.cvtColor(x_hat, cv2.COLOR_RGB2BGR)
    gt_image = gt_image.clone().detach().cpu().numpy()
    gt_image = (255 * gt_image).transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
        
    num_pixels = gt_image.shape[0] * gt_image.shape[1]
    true_bpp = _calculate_true_bpp(compress_out, num_pixels)
    est_bpp = _estimate_bpp(compress_out, num_pixels)
    psnr = cv2.PSNR(x_hat, gt_image)
    
    gt_paired_image = np.concatenate([gt_image, np.ones((20, gt_image.shape[1], 3)) * 255, x_hat], axis=0)
    
    print(f"True bpp: {true_bpp}, estimated bpp: {est_bpp}, PSNR: {psnr}")
    
    cv2.imwrite(f"{str(output_path / input_path.name.split('.')[0])}.png", gt_paired_image)


def main(model_path: str = "/data/shared/svc/pretrained/vq_vae/VQVIC-82/", 
        #  input_path: str = "/data/shared/svc/datasets/vimeo90k/vimeo_septuplet/sequences/00095/0730/im1.png", 
        # input_path: str = "/data/shared/svc/datasets/celeba_hq_256/29994.jpg",
         input_path="/data/shared/svc/datasets/kodak/kodim05.png",
         output_path: str = "/home/akhtyamov/projects/vq-vic"):
    model_path = Path(model_path)
    input_path = Path(input_path)
    output_path = Path(output_path)

    for experiment in nip.load(str(model_path / "config.nip"), always_iter=True):
        _process(experiment["pl_module"],
                 model_path / "checkpoint.ckpt",
                 input_path,
                 output_path)


if __name__ == '__main__':
    fire.Fire(main)
