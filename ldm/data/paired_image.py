import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class PairedImageDataset(Dataset):
    """
    Dataset for paired image-to-image translation tasks.

    The dataset expects two directories with matching filenames for the source and
    target domains. Images are loaded, optionally flipped, resized, converted to
    RGB, and normalized to the [-1, 1] range.
    """

    def __init__(
        self,
        source_dir: Union[str, os.PathLike],
        target_dir: Union[str, os.PathLike],
        size: Union[int, Tuple[int, int], None] = None,
        interpolation: str = "bicubic",
        flip_p: float = 0.0,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)

        self.size = size
        self.interpolation = {
            "linear": Image.LINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.file_pairs = self._match_file_pairs()
        if len(self.file_pairs) == 0:
            raise ValueError(
                f"No paired images found between {self.source_dir} and {self.target_dir}."
            )

    def _match_file_pairs(self) -> List[Tuple[Path, Path]]:
        source_files = {
            path.stem: path
            for path in self.source_dir.iterdir()
            if path.suffix.lower() in ALLOWED_EXTENSIONS and path.is_file()
        }
        target_files = {
            path.stem: path
            for path in self.target_dir.iterdir()
            if path.suffix.lower() in ALLOWED_EXTENSIONS and path.is_file()
        }

        matching_keys = sorted(set(source_files.keys()) & set(target_files.keys()))
        return [(source_files[key], target_files[key]) for key in matching_keys]

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        source_path, target_path = self.file_pairs[index]
        source_image = self._load_image(source_path)
        target_image = self._load_image(target_path)

        return {
            "source": source_image,
            "target": target_image,
            "source_path": str(source_path),
            "target_path": str(target_path),
        }

    def _load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.size is not None:
            if isinstance(self.size, tuple):
                image = image.resize(self.size[::-1], resample=self.interpolation)
            else:
                image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        return (image / 127.5 - 1.0).astype(np.float32)
