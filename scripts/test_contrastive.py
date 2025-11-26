"""
Minimal Contrastive Test Script
"""

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset
from vecssl.models.contrastive import ContrastiveModel
from vecssl.models.config import ContrastiveConfig
from vecssl.trainer import Trainer
from vecssl.util import setup_logging


logger = logging.getLogger(__name__)


def custom_collate(batch):
    """Custom collate function that handles SVGTensor objects"""
    collated = {}

    # Stack tensors normally
    collated["commands"] = torch.stack([item["commands"] for item in batch])
    collated["args"] = torch.stack([item["args"] for item in batch])
    collated["image"] = torch.stack([item["image"] for item in batch])

    # Keep SVGTensor objects and strings as lists
    collated["tensors"] = [item["tensors"] for item in batch]
    collated["uuid"] = [item["uuid"] for item in batch]
    collated["name"] = [item["name"] for item in batch]
    collated["source"] = [item["source"] for item in batch]

    return collated


def create_dataloaders(args):
    """Create train and val dataloaders"""
    logger.info("Creating datasets...")

    # Training dataset (80% of data)
    train_dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        train_ratio=0.8,
        already_preprocessed=True,
    )

    # Validation dataset (20% of data)
    val_dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        train_ratio=0.2,
        already_preprocessed=True,
    )

    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        drop_last=True,  # Drop incomplete batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        drop_last=False,
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Test Contrastive")

    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="svgx_svgs", help="SVG directory")
    parser.add_argument("--img-dir", type=str, default="svgx_imgs", help="Image directory")
    parser.add_argument("--meta", type=str, default="svgx_meta.csv", help="Metadata CSV")

    # Model args
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max number of paths")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument("--encode-stages", type=int, default=2, help="Encoder stages (1 or 2)")
    parser.add_argument("--decode-stages", type=int, default=2, help="Decoder stages (1 or 2)")
    parser.add_argument("--use-vae", action="store_true", default=True, help="Use VAE")
    parser.add_argument("--temp", type=float, default=0.07, help="CLIP temperature parameter")

    # Training args
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--tb-dir", type=str, default="runs/test_contrastive", help="TensorBoard dir"
    )

    # Logging args
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file (optional)")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (print shapes & gradients)"
    )

    # Wandb args
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vecssl",
        help="Wandb project name (enables wandb if set)",
    )
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name (optional)")
    parser.add_argument(
        "--wandb-entity", type=str, default="vecssl", help="Wandb entity/team (optional)"
    )

    args = parser.parse_args()

    # Setup logging with Rich formatting
    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        rich_tracebacks=True,
        show_level=True,
    )

    logger.info("=" * 60)
    logger.info("[bold cyan]SVG Contrastive Test[/bold cyan]", extra={"markup": True})
    logger.info("=" * 60)

    # Create config
    cfg = ContrastiveConfig()
    cfg.max_num_groups = args.max_num_groups
    cfg.max_seq_len = args.max_seq_len
    cfg.encode_stages = args.encode_stages
    cfg.decode_stages = args.decode_stages
    cfg.use_vae = args.use_vae
    cfg.contrastive_logit_scale = args.temp

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Create model
    logger.info("Creating model...")
    model = ContrastiveModel(cfg)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: [bold]{n_params:,}[/bold]", extra={"markup": True})

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create trainer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: [bold]{device}[/bold]", extra={"markup": True})

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        grad_clip=args.grad_clip,
        tb_dir=args.tb_dir,
        amp=False,  # Disable AMP for debugging
    )

    # Run training
    logger.info("Starting training...")
    try:
        trainer.run(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.epochs,
            log_every=args.log_every,
        )
        logger.info(
            "[bold green]✓ Training completed successfully![/bold green]", extra={"markup": True}
        )
    except Exception as e:
        logger.error(f"[bold red]✗ Training failed: {e}[/bold red]", extra={"markup": True})
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
