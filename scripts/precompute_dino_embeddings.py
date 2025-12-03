# scripts/precompute_dino_embeddings.py

"""
Precompute DINOv2 embeddings for images using Accelerate.

Assumes:
  - `meta` CSV has a "uuid" column
  - images live in `img_dir / f"{uuid}.png"`

Output:
  - One .pt file per sample in `output_dir`:
      {"dino": <embedding: torch.FloatTensor[dim]>}
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from vecssl.util import setup_logging, make_progress
from vecssl.models.model import DINOImageEncoder

logger = logging.getLogger(__name__)


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
    # Keep as lists so DINOImageEncoder can pass them directly to the HF processor
    images = [b["image"] for b in batch]
    uuids = [b["uuid"] for b in batch]
    return {"images": images, "uuids": uuids}


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute DINOv2 embeddings with Accelerate")

    parser.add_argument("--img-dir", type=str, required=True, help="Directory with PNG images")
    parser.add_argument("--meta", type=str, required=True, help="CSV with at least a 'uuid' column")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Where to save .pt embedding files"
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)

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
        logger.info("Starting DINO precompute with Accelerate")
        logger.info(f"Using device: {accelerator.device}")
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

    # Your DINOImageEncoder exactly as defined
    model = DINOImageEncoder()
    model.eval()
    model.requires_grad_(False)

    # Let Accelerate handle device placement + DistributedSampler
    model, dataloader = accelerator.prepare(model, dataloader)

    torch.set_grad_enabled(False)

    progress = make_progress()
    with progress:
        task = progress.add_task("dino", total=len(dataloader))

        for batch in dataloader:
            images = batch["images"]  # list[PIL.Image]
            uuids = batch["uuids"]  # list[str]

            # Use accelerator.autocast like in your Trainer
            with accelerator.autocast():
                embeddings = model(images)  # (B, dim) tensor on local device

            # Move to CPU for saving
            embeddings = embeddings.detach().cpu()

            # Each rank writes only its own shard of UUIDs.
            # DistributedSampler from accelerator.prepare ensures no overlap.
            for uuid, emb in zip(uuids, embeddings, strict=False):
                out_path = output_dir / f"{uuid}.pt"
                torch.save({"dino": emb}, out_path)

            progress.advance(task)

    accelerator.print("Done precomputing DINO embeddings.")


if __name__ == "__main__":
    main()
