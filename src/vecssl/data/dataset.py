"""
Data utils for vecssl

Currently we have:
 - SVGX-Core-250k (preprocessed files)

"""

from typing import Any
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import pandas as pd

import torch
import random

from vecssl.data.svg import SVG
from vecssl.data.geom import Point
from vecssl.data.svg_tensor import SVGTensor


class SVGXDataset(Dataset):
    def __init__(
        self,
        svg_dir: str,
        img_dir: str,
        meta_filepath: str,
        max_num_groups: int = 8,
        max_seq_len: int = 40,
        pad_val: int = -1,
        split: str = "train",  # "train", "val", or "test"
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        already_preprocessed: bool = True,
        already_tensor: bool = True,
        cache: bool = False,
    ):
        self.svg_dir = svg_dir
        self.img_dir = img_dir
        self.already_preprocessed = already_preprocessed
        self.already_tensor = already_tensor  # ? maybe add an assert for this?
        self.cache = cache
        self._data_cache = {}

        self.MAX_NUM_GROUPS = max_num_groups
        self.MAX_SEQ_LEN = max_seq_len
        self.MAX_TOTAL_LEN = max_num_groups * max_seq_len
        self.PAD_VAL = pad_val

        # Load and filter metadata
        df = pd.read_csv(meta_filepath)

        # Filter by constraints (like DeepSVG)
        df = df[(df.nb_groups <= max_num_groups) & (df.max_len_group <= max_seq_len)]

        # Shuffle with seed for reproducible splits
        df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(df_shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Select split
        if split == "train":
            df = df_shuffled.iloc[:train_end]
        elif split == "val":
            df = df_shuffled.iloc[train_end:val_end]
        elif split == "test":
            df = df_shuffled.iloc[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        # Reset index to ensure contiguous 0-N indexing
        self.df = df.reset_index(drop=True)

        if self.cache:
            # ! DEBUG
            for i in range(len(self.df)):
                _ = self.__getitem__(i)

    def __len__(self) -> int:
        return len(self.df)

    def _load_svg_tensor(self, uuid: str):
        res = torch.load(Path(self.svg_dir) / f"{uuid}.pt")
        return res["t_sep"], res["fillings"]

    def _load_svg(self, uuid: str):
        svg_path = os.path.join(self.svg_dir, f"{uuid}.svg")
        # Read SVG file content
        with open(svg_path, "r") as f:
            svg_code = f.read()

        svg = SVG.load_svg(svg_code).to_path()

        # SVGs are already preprocessed, no need to simplify
        if not self.already_preprocessed:
            svg.fill_(False)
            svg.normalize().zoom(0.9)
            svg.canonicalize()
            svg = svg.simplify_heuristic()
        return svg

    @staticmethod
    def _augment(svg: SVG, mean=False):
        dx, dy = (0, 0) if mean else (5 * random.random() - 2.5, 5 * random.random() - 2.5)
        factor = 0.7 if mean else 0.2 * random.random() + 0.6
        return svg.zoom(factor).translate(Point(dx, dy))

    @staticmethod
    def preprocess(svg: SVG, augment=False, numericalize=True, mean=False):
        if augment:
            svg = SVGXDataset._augment(svg, mean=mean)
        if numericalize:
            return svg.numericalize(256)
        return svg

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.cache and idx in self._data_cache:
            return self._data_cache[idx]
        # Get entry from filtered dataframe
        entry = self.df.iloc[idx]
        uuid = entry["uuid"]

        if not self.already_tensor:
            # Load SVG from disk
            svg = self._load_svg(uuid)
            svg = SVGXDataset.preprocess(svg, augment=False)  # No augmentation

            # Convert to tensors
            t_sep, fillings = svg.to_tensor(concat_groups=False), svg.to_fillings()
        else:
            # Load SVG tensor from disk
            t_sep, fillings = self._load_svg_tensor(uuid)

        res = self.get_data(t_sep, fillings)
        # Load image from disk
        img_path = os.path.join(self.img_dir, f"{uuid}.png")
        image = Image.open(img_path)
        image_tensor = transforms.ToTensor()(image)

        _data = {
            "commands": res["commands"],
            "args": res["args"],
            "tensors": res["tensors"],  # SVGTensor objects with metadata
            "image": image_tensor,
            "uuid": uuid,
            "name": entry.get("name", ""),
            "source": entry.get("source", ""),
            "label": entry.get("label", -1),  # -1 for datasets without labels
            "family_label": entry.get("family_label", -1),  # -1 for datasets without labels
        }

        # Save to cache
        if self.cache:
            self._data_cache[idx] = _data

        # Return dict with all data
        return _data

    def get_data(self, t_sep, fillings):
        res = {}
        # Pad if there are too few groups
        pad_len = max(self.MAX_NUM_GROUPS - len(t_sep), 0)
        t_sep.extend([torch.empty(0, 14)] * pad_len)  # ! `14` hard-coded
        fillings.extend([0] * pad_len)

        t_sep = [
            SVGTensor.from_data(t, PAD_VAL=self.PAD_VAL, filling=f)
            .add_eos()
            .add_sos()
            .pad(seq_len=self.MAX_SEQ_LEN + 2)
            for t, f in zip(t_sep, fillings, strict=False)
        ]

        # Return both tensors AND SVGTensor objects (for reconstruction)
        res["commands"] = torch.stack([t.cmds() for t in t_sep])
        res["args"] = torch.stack([t.args() for t in t_sep])
        res["tensors"] = t_sep  # Keep SVGTensor objects with metadata
        return res

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
