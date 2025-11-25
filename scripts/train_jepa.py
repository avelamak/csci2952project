import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset
from vecssl.models.config import JepaConfig
from vecssl.models.jepa import Jepa
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
    parser = argparse.ArgumentParser(description="Train JEPA model")

    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="svgx_svgs", help="SVG directory")
    parser.add_argument("--img-dir", type=str, default="svgx_imgs", help="Image directory")
    parser.add_argument("--meta", type=str, default="svgx_meta.csv", help="Metadata CSV")

    # Model args
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max number of paths")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument("--use-resnet", action="store_true", default=False, help="Use ResNet")
    parser.add_argument(
        "--predictor-type", type=str, default="mlp", help="Type of predictor (transformer/mlp)"
    )

    # Training args
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--tb-dir", type=str, default="runs/test_jepa", help="TensorBoard dir")

    # Logging args
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Wandb args
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="Wandb project name (enables wandb if set)"
    )
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name (optional)")
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="Wandb entity/team (optional)"
    )

    # Checkpoint args
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    setup_logging(
        level=args.log_level, log_file=args.log_file, rich_tracebacks=True, show_level=True
    )

    logger.info("=" * 60)
    logger.info("[bold cyan]JEPA Training[/bold cyan]", extra={"markup": True})
    logger.info("=" * 60)

    cfg = JepaConfig()
    cfg.max_num_groups = args.max_num_groups
    cfg.max_seq_len = args.max_seq_len
    cfg.use_resnet = args.use_resnet
    cfg.predictor_type = args.predictor_type

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Count parameters
    logger.info("Creating model...")
    model = Jepa(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: [bold]{n_params:,}[/bold]", extra={"markup": True})
    n_params = sum(p.numel() for p in model.svg_encoder.parameters() if p.requires_grad)
    logger.info(f"Total parameters svg encoder: [bold]{n_params:,}[/bold]", extra={"markup": True})
    n_params = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
    logger.info(f"Total parameters predictor: [bold]{n_params:,}[/bold]", extra={"markup": True})
    n_params = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
    logger.info(
        f"Total parameters image encoder: [bold]{n_params:,}[/bold]", extra={"markup": True}
    )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create trainer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: [bold]{device}[/bold]", extra={"markup": True})

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from:
        from vecssl.trainer import load_chkpt

        logger.info(
            f"Resuming from checkpoint: [bold]{args.resume_from}[/bold]", extra={"markup": True}
        )
        metadata = load_chkpt(
            checkpoint_path=args.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=None,  # No scheduler in this script
            device=device,
        )
        start_epoch = metadata["epoch"] + 1
        logger.info(f"Resuming training from epoch {start_epoch}", extra={"markup": True})

    # Prepare wandb config (combine model config + training args)
    wandb_config = None
    if args.wandb_project:
        wandb_config = {
            # Model config
            "max_num_groups": cfg.max_num_groups,
            "max_seq_len": cfg.max_seq_len,
            "use_resnet": cfg.use_resnet,
            "predictor_type": cfg.predictor_type,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_layers_decode": cfg.n_layers_decode,
            "n_heads": cfg.n_heads,
            "dim_feedforward": cfg.dim_feedforward,
            "dim_z": cfg.dim_z,
            "dropout": cfg.dropout,
            "d_joint": cfg.d_joint,
            "predictor_transformer_num_heads": cfg.predictor_transformer_num_heads,
            "predictor_transformer_num_layers": cfg.predictor_transformer_num_layers,
            "predictor_transformer_hidden_dim": cfg.predictor_transformer_hidden_dim,
            "predictor_transformer_dropout": cfg.predictor_transformer_dropout,
            "predictor_mlp_num_layers": cfg.predictor_mlp_num_layers,
            "predictor_mlp_hidden_dim": cfg.predictor_mlp_hidden_dim,
            "predictor_mlp_dropout": cfg.predictor_mlp_dropout,
            # Training args
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "grad_clip": args.grad_clip,
            "num_workers": args.num_workers,
            "log_every": args.log_every,
            "device": str(device),
            "amp": True,
            "n_params": n_params,
        }
        logger.info(
            f"Wandb enabled - project: [bold]{args.wandb_project}[/bold]", extra={"markup": True}
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        device=device,
        grad_clip=args.grad_clip,
        tb_dir=args.tb_dir,
        amp=True,
        wandb_project=args.wandb_project,
        cfg=wandb_config,
    )

    # Run training
    logger.info("Starting training...")
    try:
        trainer.run(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.epochs,
            log_every=args.log_every,
            save_every=args.save_every,
            start_epoch=start_epoch,
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
