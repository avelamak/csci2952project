from typing import Any, Optional
import io
import os
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from vecssl.data.svg import SVG
from vecssl.data.geom import Point
from vecssl.data.svg_tensor import SVGTensor
import logging
from vecssl.util import make_progress

logger = logging.getLogger(__file__)


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
        use_precomputed_dino: bool = False,
        dino_dir: Optional[str] = None,
        use_precomputed_dino_patches: bool = False,
        dino_patches_dir: Optional[str] = None,
        stratify_by: Optional[str] = None,
        min_class_count: int = 2,
    ):
        self.svg_dir = svg_dir
        self.img_dir = img_dir
        self.already_preprocessed = already_preprocessed
        self.already_tensor = already_tensor
        self.cache = cache
        self.use_precomputed_dino = use_precomputed_dino
        self.dino_dir = dino_dir
        self.use_precomputed_dino_patches = use_precomputed_dino_patches
        self.dino_patches_dir = dino_patches_dir

        if use_precomputed_dino and dino_dir is None:
            raise ValueError("dino_dir must be provided when use_precomputed_dino is True")

        if use_precomputed_dino_patches and dino_patches_dir is None:
            raise ValueError(
                "dino_patches_dir must be provided when use_precomputed_dino_patches is True"
            )

        self.MAX_NUM_GROUPS = max_num_groups
        self.MAX_SEQ_LEN = max_seq_len
        self.MAX_TOTAL_LEN = max_num_groups * max_seq_len
        self.PAD_VAL = pad_val

        # Cached samples: list indexed by dataset idx
        # Each entry will hold compressed image bytes + precomputed SVG tensors + meta
        self._data_cache: Optional[list[dict[str, Any]]] = None

        # Reuse ToTensor instance
        self._to_tensor = transforms.ToTensor()

        # ---------------------------------------------------------------------
        # Load and filter metadata
        # ---------------------------------------------------------------------
        df = pd.read_csv(meta_filepath)

        # Filter by constraints (like DeepSVG)
        df = df[(df.nb_groups <= max_num_groups) & (df.max_len_group <= max_seq_len)]

        # Shuffle with seed for reproducible splits
        df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        if stratify_by and stratify_by in df_shuffled.columns:
            # Filter out classes with too few samples for stratified split
            if min_class_count > 1:
                class_counts = df_shuffled[stratify_by].value_counts()
                valid_classes = class_counts[class_counts >= min_class_count].index
                original_len = len(df_shuffled)
                df_shuffled = df_shuffled[df_shuffled[stratify_by].isin(valid_classes)]
                if len(df_shuffled) < original_len:
                    logger.info(
                        f"Filtered {original_len - len(df_shuffled)} samples from classes "
                        f"with < {min_class_count} samples"
                    )

            # Stratified split: ensure each class appears proportionally in all splits
            stratify_col = df_shuffled[stratify_by]

            # First split: train vs (val+test)
            train_df, valtest_df = train_test_split(
                df_shuffled,
                train_size=train_ratio,
                stratify=stratify_col,
                random_state=seed,
            )

            # Second split: val vs test
            val_test_ratio = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                valtest_df,
                train_size=val_test_ratio,
                stratify=valtest_df[stratify_by],
                random_state=seed,
            )

            if split == "train":
                df_split = train_df
            elif split == "val":
                df_split = val_df
            elif split == "test":
                df_split = test_df
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        else:
            # Original random split behavior
            n = len(df_shuffled)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            if split == "train":
                df_split = df_shuffled.iloc[:train_end]
            elif split == "val":
                df_split = df_shuffled.iloc[train_end:val_end]
            elif split == "test":
                df_split = df_shuffled.iloc[val_end:]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        self.df = df_split.reset_index(drop=True)

        # ---------------------------------------------------------------------
        # Eager cache build (compressed images + precomputed SVG + DINO)
        # ---------------------------------------------------------------------
        if self.cache:
            self._build_cache()

    def __len__(self) -> int:
        return len(self.df)

    # -------------------------------------------------------------------------
    # Low-level I/O helpers
    # -------------------------------------------------------------------------

    def _load_svg_tensor(self, uuid: str):
        """Load pre-tensorized SVG from disk (t_sep + fillings)."""
        path = Path(self.svg_dir) / f"{uuid}.pt"
        res = torch.load(path, map_location="cpu")
        return res["t_sep"], res["fillings"]

    def _load_svg(self, uuid: str) -> SVG:
        """Load raw SVG and optionally simplify / normalize."""
        svg_path = os.path.join(self.svg_dir, f"{uuid}.svg")
        with open(svg_path, "r") as f:
            svg_code = f.read()

        svg = SVG.load_svg(svg_code).to_path()

        if not self.already_preprocessed:
            svg.fill_(False)
            svg.normalize().zoom(0.9)
            svg.canonicalize()
            svg = svg.simplify_heuristic()

        return svg

    @staticmethod
    def _augment(svg: SVG, mean: bool = False) -> SVG:
        dx, dy = (0, 0) if mean else (5 * random.random() - 2.5, 5 * random.random() - 2.5)
        factor = 0.7 if mean else 0.2 * random.random() + 0.6
        return svg.zoom(factor).translate(Point(dx, dy))

    @staticmethod
    def preprocess(svg: SVG, augment: bool = False, numericalize: bool = True, mean: bool = False):
        if augment:
            svg = SVGXDataset._augment(svg, mean=mean)
        if numericalize:
            return svg.numericalize(256)
        return svg

    # -------------------------------------------------------------------------
    # Cache building
    # -------------------------------------------------------------------------

    def _build_cache(self) -> None:
        """
        Build an in-RAM cache where:
          - images are stored as compressed PNG bytes
          - commands/args/tensors are precomputed and stored as tensors/objects
          - dino embeddings are stored as CPU tensors
        """
        logger.info("Building cache...")
        num_samples = len(self.df)
        cache: list[dict[str, Any]] = []

        progress = make_progress()
        with progress:
            _cache_task = progress.add_task("cache", total=num_samples)
            for idx in range(num_samples):
                entry = self.df.iloc[idx]
                uuid = entry["uuid"]

                # --- SVG / commands / args ---
                if not self.already_tensor:
                    svg = self._load_svg(uuid)
                    svg = SVGXDataset.preprocess(svg, augment=False)
                    t_sep, fillings = svg.to_tensor(concat_groups=False), svg.to_fillings()
                else:
                    t_sep, fillings = self._load_svg_tensor(uuid)

                svg_res = self.get_data(t_sep, fillings)  # commands / args / SVGTensor list

                # --- Image as compressed bytes ---
                img_path = os.path.join(self.img_dir, f"{uuid}.png")
                img_bytes = Path(img_path).read_bytes()

                # --- DINO CLS embedding ---
                dino_embedding = None
                if self.use_precomputed_dino:
                    dino_path = os.path.join(self.dino_dir, f"{uuid}.pt")
                    dino_data = torch.load(dino_path, map_location="cpu")
                    dino_embedding = dino_data["dino"]  # [768] or whatever

                # --- DINO patch embeddings (for MultiMAE) ---
                dino_patches = None
                if self.use_precomputed_dino_patches:
                    patches_path = os.path.join(self.dino_patches_dir, f"{uuid}.pt")
                    patches_data = torch.load(patches_path, map_location="cpu")
                    dino_patches = patches_data["patches"]  # (num_patches, 768)

                cached_sample: dict[str, Any] = {
                    # SVG stuff (ready-to-use tensors / objects)
                    "commands": svg_res["commands"],
                    "args": svg_res["args"],
                    "tensors": svg_res["tensors"],
                    # Compressed image bytes
                    "image_bytes": img_bytes,
                    # Metadata
                    "uuid": uuid,
                    "name": entry.get("name", ""),
                    "source": entry.get("source", ""),
                    "label": entry.get("label", -1),
                    "family_label": entry.get("family_label", -1),
                }

                if dino_embedding is not None:
                    cached_sample["dino_embedding"] = dino_embedding

                if dino_patches is not None:
                    cached_sample["dino_patches"] = dino_patches

                cache.append(cached_sample)
                progress.advance(_cache_task)
        self._data_cache = cache

    # -------------------------------------------------------------------------
    # Main accessor
    # -------------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Fast path: cached
        if self.cache and self._data_cache is not None:
            cached = self._data_cache[idx]

            # Decode image from compressed bytes
            img_bytes = cached["image_bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image_tensor = self._to_tensor(img)

            sample: dict[str, Any] = {
                "commands": cached["commands"],
                "args": cached["args"],
                "tensors": cached["tensors"],
                "image": image_tensor,
                "uuid": cached["uuid"],
                "name": cached["name"],
                "source": cached["source"],
                "label": cached["label"],
                "family_label": cached["family_label"],
            }

            if "dino_embedding" in cached:
                sample["dino_embedding"] = cached["dino_embedding"]

            if "dino_patches" in cached:
                sample["dino_patches"] = cached["dino_patches"]

            return sample

        # Slow path: load from disk every time (original behavior)
        entry = self.df.iloc[idx]
        uuid = entry["uuid"]

        if not self.already_tensor:
            svg = self._load_svg(uuid)
            svg = SVGXDataset.preprocess(svg, augment=False)
            t_sep, fillings = svg.to_tensor(concat_groups=False), svg.to_fillings()
        else:
            t_sep, fillings = self._load_svg_tensor(uuid)

        svg_res = self.get_data(t_sep, fillings)

        img_path = os.path.join(self.img_dir, f"{uuid}.png")
        img = Image.open(img_path).convert("RGB")
        image_tensor = self._to_tensor(img)

        sample: dict[str, Any] = {
            "commands": svg_res["commands"],
            "args": svg_res["args"],
            "tensors": svg_res["tensors"],
            "image": image_tensor,
            "uuid": uuid,
            "name": entry.get("name", ""),
            "source": entry.get("source", ""),
            "label": entry.get("label", -1),
            "family_label": entry.get("family_label", -1),
        }

        if self.use_precomputed_dino:
            dino_path = os.path.join(self.dino_dir, f"{uuid}.pt")
            dino_data = torch.load(dino_path, map_location="cpu")
            sample["dino_embedding"] = dino_data["dino"]

        if self.use_precomputed_dino_patches and self.dino_patches_dir is not None:
            patches_path = os.path.join(self.dino_patches_dir, f"{uuid}.pt")
            patches_data = torch.load(patches_path, map_location="cpu")
            sample["dino_patches"] = patches_data["patches"]  # (num_patches, 768)

        return sample

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def get_data(self, t_sep, fillings):
        res: dict[str, Any] = {}

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

        res["commands"] = torch.stack([t.cmds() for t in t_sep])
        res["args"] = torch.stack([t.args() for t in t_sep])
        res["tensors"] = t_sep
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
