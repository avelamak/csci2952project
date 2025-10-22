"""
Data utils for vecssl

Currently we have:
 - SVGX-Core-250k

"""

from typing import Any, Dict, Optional
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import io
import torch
import torchvision.transforms as transforms


class SVGXDataset(Dataset):
    """
    Dataset wrapper for SVGX-Core-250k.
    This dataset contains ~250k SVG vector graphics with:
    - SVG paths and code
    - BLIP captions
    - Rendered images (128x128)
    - Descriptions
    - Source information
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """
        Initialize the SVGX dataset.

        Args:
            split: Dataset split to use (e.g., "train", "test")
            cache_dir: Directory to cache the downloaded dataset
            transform: Optional transform to apply to images (if None, uses ToTensor())
        """
        self.dataset: datasets.Dataset = load_dataset(  # type: ignore
            "xingxm/SVGX-Core-250k",
            split=split,
            cache_dir=cache_dir,
        )

        # Default transform converts PIL Image to tensor
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]

        # Convert image bytes to PIL Image if needed, then to tensor
        if "image" in sample:
            if isinstance(sample["image"], (bytes, io.BytesIO)):
                sample["image"] = Image.open(io.BytesIO(sample["image"]))

            # Convert PIL Image to tensor
            if isinstance(sample["image"], Image.Image):
                sample["image"] = self.transform(sample["image"])

        return sample

    def get_sample_info(self, idx: int) -> dict[str, Any]:
        sample = self[idx]
        info = {
            "index": idx,
            "has_image": "image" in sample,
            "has_svg": "svg" in sample or "svg_code" in sample,
            "has_caption": "caption" in sample or "blip_caption" in sample,
            "keys": list(sample.keys()),
        }
        return info
