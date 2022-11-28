from pathlib import Path
from typing import Optional, Tuple, Callable, Union, Any
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from nip import nip


@nip
class ImagesDataset(Dataset):

    def __init__(self,
                 root_dir: Union[str, Path],
                 extension: Optional[str] = None,
                 segment: Tuple[int, int] = (0, -1),
                 transform: Optional[Callable] = None):
        """Simple dataset of the images. Can be used for image encoding tasks.

        Parameters
        ----------
        root_dir: Root directory with images
        extension: Target extension of the image files; if None, all files in directory will be used
        segment: Segment of the image list indices to use
        transform: Transformations and augmentations
        """
        super(ImagesDataset, self).__init__()
        root_dir = Path(root_dir)
        pattern = "*" if extension is None else f"*.{extension}"
        self._images = sorted([image_file for image_file in root_dir.glob(pattern)
                               if image_file.is_file()])[segment[0]:segment[1]]
        self._transform = transform or ToTensor()

    def __len__(self) -> int:
        """Calculates length of the dataset - number of images.

        Returns
        -------
        Length of the dataset
        """
        return len(self._images)

    def __getitem__(self, idx: int) -> Any:
        """Returns entry of the dataset.

        Parameters
        ----------
        idx: Index of the entry

        Returns
        -------
        Entry of the dataset - image with applied transforms
        """
        image = Image.open(str(self._images[idx]))
        image = self._transform(image)
        return image


@nip
class VimeoImagesDataset(Dataset):

    _FRAME_NAME = "im1.png"

    def __init__(self,
                 sequences_root: Union[Path, str],
                 subset_list: Union[Path, str],
                 segment: Tuple[int, int] = (0, -1),
                 transform: Optional[Callable] = None):
        with open(subset_list, "r") as f:
            self._videos = f.readlines()
        self._videos = [path[:-1] for path in self._videos]
        self._videos = self._videos[segment[0]:segment[1]]
        self._sequences_root = Path(sequences_root)
        self._transform = transform or ToTensor()

    def __len__(self) -> int:
        return len(self._videos)

    def __getitem__(self, item: int) -> Any:
        video = self._videos[item]
        video_dir, gop_dir = video.split("/")
        image_path = self._sequences_root / video_dir / gop_dir / VimeoImagesDataset._FRAME_NAME
        image = Image.open(str(image_path))
        image = self._transform(image)
        return image
