# scripts/precompute_dino_patches.py

"""
Precompute DINOv2 patch embeddings for images using Accelerate.

Unlike precompute_dino_embeddings.py which saves CLS embeddings,
this script saves the patch token embeddings for use with MultiMAE.

Assumes:
  - `meta` CSV has a "uuid" column
  - images live in `img_dir / f"{uuid}.png"`

Output:
  - One .pt file per sample in `output_dir`:
      {"patches": <embedding: torch.FloatTensor[num_patches, 768]>}
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Dinov2Model, AutoImageProcessor

from accelerate import Accelerator
from vecssl.util import setup_logging, make_progress

logger = logging.getLogger(__name__)


class DINOPatchExtractor(nn.Module):
    """
    Extract DINO patch tokens (excluding CLS and register tokens).
    Similar to DINOImagePatchEncoder but as a standalone module.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", layer: int = -1):
        super().__init__()
        self.layer = layer

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()

    def forward(self, images):
        """
        Args:
            images: List of PIL Images

        Returns:
            patch_tokens: (batch, num_patches, hidden_dim) - patch tokens only (no CLS/register)
        """
        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Get model outputs
        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        h = hidden_states[self.layer]

        # Remove CLS token (and any register tokens if present)
        # DINOv2-base has 1 CLS token + 256 patch tokens (for 224x224 images with 14x14 patches)
        patch_tokens = h[:, 1:, :]  # (batch, num_patches, hidden_dim)

        return patch_tokens


class DinoPrecomputeDataset(Dataset):
    def __init__(self, img_dir: str, meta_filepath: str, max_samples: int | None = None):
        self.img_dir = Path(img_dir)
        df = pd.read_csv(meta_filepath)

        if max_samples is not None:
            df = df.iloc[:max_samples].reset_index(drop=True)

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uuid = row["uuid"]
        img_path = self.img_dir / f"{uuid}.png"

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        return {"uuid": uuid, "image": image}


def dino_collate(batch):
    # Keep as lists so DINOPatchExtractor can pass them directly to the HF processor
    images = [b["image"] for b in batch]
    uuids = [b["uuid"] for b in batch]
    return {"images": images, "uuids": uuids}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute DINOv2 patch embeddings with Accelerate"
    )

    parser.add_argument(
        "--DINO_layer",
        type=int,
        default=-1,
        help="-1 = last layer, 0 = patch embedding output, 1..n = transformer blocks",
    )
    parser.add_argument("--img-dir", type=str, required=True, help="Directory with PNG images")
    parser.add_argument("--meta", type=str, required=True, help="CSV with at least a 'uuid' column")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Where to save .pt patch embedding files"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (smaller than CLS due to more data)"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-base",
        help="DINO model name from HuggingFace",
    )

    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Accelerate mixed precision mode",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if accelerator.is_main_process:
        logger.info("Starting DINO patch precompute with Accelerate")
        logger.info(f"Using device: {accelerator.device}")
        logger.info(f"Using model: {args.model_name}")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    output_dir = Path(args.output_dir)  # ensure a Path on all ranks

    dataset = DinoPrecomputeDataset(
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_samples=args.max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=dino_collate,
    )

    # Create patch extractor model
    model = DINOPatchExtractor(model_name=args.model_name, layer=args.DINO_layer)
    model.eval()
    model.requires_grad_(False)

    # Let Accelerate handle device placement + DistributedSampler
    model, dataloader = accelerator.prepare(model, dataloader)

    torch.set_grad_enabled(False)

    progress = make_progress()
    with progress:
        task = progress.add_task("dino_patches", total=len(dataloader))

        for batch in dataloader:
            images = batch["images"]  # list[PIL.Image]
            uuids = batch["uuids"]  # list[str]

            # Use accelerator.autocast like in your Trainer
            with accelerator.autocast():
                patch_embeddings = model(images)  # (B, num_patches, 768) tensor on local device

            # Move to CPU for saving
            patch_embeddings = patch_embeddings.detach().cpu()

            # Each rank writes only its own shard of UUIDs.
            # DistributedSampler from accelerator.prepare ensures no overlap.
            for uuid, patches in zip(uuids, patch_embeddings, strict=False):
                out_path = output_dir / f"{uuid}.pt"
                torch.save({"patches": patches}, out_path)

            progress.advance(task)

    accelerator.print("Done precomputing DINO patch embeddings.")


if __name__ == "__main__":
    main()
