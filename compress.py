import torch
import fire
import nip
import cv2

from pathlib import Path
from tqdm import tqdm
from lib.lightning_modules import LitAutoEncoderModule
from PIL import Image
from torchvision.transforms import ToTensor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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



def main(mode_path: str, input_path: str, output_path: str):
    mode_path = Path(mode_path)
    input_path = Path(input_path)
    output_path = Path(output_path)

    for experiment in nip.load(str(mode_path / "config.nip"), always_iter=True):
        for video_path in tqdm(videos, leave=False):
            video_result = _eval(experiment['pl_module'], str(model_dir / "checkpoint.ckpt"),
                                 video_path, rec_output, device, batch_size=batch_size)
            result.append(video_result)
            psnrs.append(video_result["average_frame_psnr"])
            bpps.append(video_result["bpp"])

            with open(str(output_dir / "test_videos_result.json"), "w") as f:
                json.dump(result, f, indent=4)

        print(f"Average PSNR: {round(np.mean(psnrs), 2)}")
        print(f"Average bpp: {round(np.mean(bpps), 2)}")


if __name__ == '__main__':
    fire.Fire(main)
