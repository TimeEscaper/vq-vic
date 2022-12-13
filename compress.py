import torch
import fire
import nip
import cv2

from pathlib import Path
from tqdm import tqdm
from lib.lightning_modules import LitAutoEncoderModule
from PIL import Image
from torchvision.transforms import ToTensor


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

def _process(pl_module: LitAutoEncoderModule,
             checkpoint_path: Path,
             input_path: Path,
             output_path: Path):
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    pl_module.load_state_dict(checkpoint['state_dict'])
    pl_module = pl_module.to(DEVICE)
    pl_module.eval()

    with Image.open(str(input_path)) as img:
        to_tensor = ToTensor()
        gt_image = to_tensor(img)

    compress_out = pl_module.compress(gt_image.unsqueeze(0).to(DEVICE))



def main(model_path: str, input_path: str, output_path: str):
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
