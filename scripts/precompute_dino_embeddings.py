# scripts/precompute_dino_embeddings.py

"""
Precompute DINOv2 embeddings for SVGX images.

Assumes:
  - `meta` CSV has a "uuid" column
  - images live in `img_dir / f"{uuid}.png"`

Output:
  - One .pt file per sample in `output_dir`:
      {"dino": <embedding: torch.FloatTensor[hidden_dim]>}
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

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
    # Return lists so the HF processor can handle them nicely
    images = [b["image"] for b in batch]
    uuids = [b["uuid"] for b in batch]
    return {"image": images, "uuid": uuids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, required=True, help="Directory with PNGs")
    parser.add_argument("--meta", type=str, required=True, help="Metadata CSV with 'uuid'")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save DINO embeddings (*.pt)"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files")
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        help="Store L2-normalized DINO embeddings (recommended for distillation)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    console = setup_logging(level=args.log_level, reset=True)
    logger.info("[bold cyan]Precomputing DINO embeddings...[/bold cyan]", extra={"markup": True})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DinoPrecomputeDataset(args.img_dir, args.meta, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dino_collate,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DINOImageEncoder().to(device)
    model.eval()

    total = len(dataset)
    logger.info(f"[blue]Dataset size:[/blue] {total}", extra={"markup": True})

    from torch.nn.functional import normalize as l2_normalize

    progress = make_progress(console)
    with torch.no_grad(), progress:
        task = progress.add_task("DINO embeddings", total=total)

        processed = 0
        for batch in dataloader:
            images = batch["image"]
            uuids = batch["uuid"]

            # Forward through DINOImageEncoder
            embeddings = model(images)  # shape: [B, hidden_dim]
            if args.l2_normalize:
                embeddings = l2_normalize(embeddings, dim=-1)

            embeddings = embeddings.cpu()

            for emb, uuid in zip(embeddings, uuids, strict=False):
                out_path = output_dir / f"{uuid}.pt"
                if out_path.exists() and not args.overwrite:
                    # Optionally you could log a "skipping existing" message here
                    continue
                torch.save({"dino": emb}, out_path)

            processed += len(uuids)
            progress.advance(task, len(uuids))

    logger.info(
        f"[bold green]Done![/bold green] Processed approx. {processed}/{total} samples",
        extra={"markup": True},
    )


if __name__ == "__main__":
    main()

