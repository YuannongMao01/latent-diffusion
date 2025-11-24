import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:  # pragma: no cover - fallback for environments without Pillow
    Image = None


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".npy"}


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
        self.interpolation = interpolation
        self.flip_p = flip_p

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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_path, target_path = self.file_pairs[index]
        source_image = self._load_image(source_path)
        target_image = self._load_image(target_path)

        return {
            "cond": source_image,
            "jpg": target_image,
            "cond_path": str(source_path),
            "jpg_path": str(target_path),
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        image = self._read_image(path)
        image = self._maybe_resize(image)
        image = self._maybe_flip(image)
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor.contiguous()

    def _read_image(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            array = np.load(path)
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)
            return array
        if Image is None:
            raise ImportError(
                "Pillow is required to load image files. Install pillow or provide .npy arrays."
            )
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _maybe_resize(self, image: Union[np.ndarray, "Image.Image"]):
        if self.size is None:
            return image

        if isinstance(image, np.ndarray):
            if isinstance(self.size, tuple):
                target_h, target_w = self.size
            else:
                target_h = target_w = self.size

            if Image is not None:
                return np.array(
                    Image.fromarray(image.astype(np.uint8)).resize(
                        (target_w, target_h), resample=self._pil_interpolation()
                    )
                )

            # Minimal nearest-neighbor resize without Pillow (debug fallback).
            return np.resize(image, (target_h, target_w, image.shape[2]))

        if isinstance(self.size, tuple):
            return image.resize(self.size[::-1], resample=self._pil_interpolation())
        return image.resize((self.size, self.size), resample=self._pil_interpolation())

    def _maybe_flip(self, image: Union[np.ndarray, "Image.Image"]):
        if self.flip_p <= 0:
            return image
        if random.random() > self.flip_p:
            return image
        if isinstance(image, np.ndarray):
            return np.flip(image, axis=1).copy()
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def _pil_interpolation(self):
        if Image is None:
            raise ImportError("Pillow is required for resizing non-npy images.")
        return {
            "linear": Image.LINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[self.interpolation]
