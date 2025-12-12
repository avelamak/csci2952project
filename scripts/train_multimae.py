import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset
from vecssl.models.config import MultiMAEConfig
from vecssl.models.multimae import MultiMAE
from vecssl.trainer import Trainer
from vecssl.util import setup_logging, set_seed

from eval_utils import custom_collate

logger = logging.getLogger(__name__)


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
        split="train",
        seed=args.seed,
        already_preprocessed=True,
        use_precomputed_dino_patches=args.use_precomputed_dino_patches,
        dino_patches_dir=args.dino_patches_dir,
    )

    # Validation dataset (10% of data)
    val_dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        split="val",
        seed=args.seed,
        already_preprocessed=True,
        use_precomputed_dino_patches=args.use_precomputed_dino_patches,
        dino_patches_dir=args.dino_patches_dir,
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
    parser = argparse.ArgumentParser(description="Train Multi-modal MAE model (SVG + Image)")

    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="svgx_svgs", help="SVG directory")
    parser.add_argument("--img-dir", type=str, default="svgx_imgs", help="Image directory")
    parser.add_argument("--meta", type=str, default="svgx_meta.csv", help="Metadata CSV")
    parser.add_argument(
        "--use-precomputed-dino-patches",
        action="store_true",
        help="Use precomputed DINO patch embeddings instead of computing on-the-fly",
    )
    parser.add_argument(
        "--dino-patches-dir",
        type=str,
        default=None,
        help="Directory containing precomputed DINO patch embeddings (required if --use-precomputed-dino-patches)",
    )

    # Model args
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max number of paths")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument("--mask-ratio-svg", type=float, default=0.5, help="SVG group masking ratio")
    parser.add_argument(
        "--mask-ratio-img", type=float, default=0.75, help="Image patch masking ratio"
    )
    parser.add_argument("--mae-depth", type=int, default=8, help="MAE encoder depth")
    parser.add_argument("--mae-num-heads", type=int, default=8, help="MAE encoder num heads")
    parser.add_argument("--mae-mlp-ratio", type=float, default=4.0, help="MAE MLP ratio")
    parser.add_argument("--mae-dropout", type=float, default=0.1, help="MAE dropout")
    parser.add_argument("--loss-cmd-weight", type=float, default=1.0, help="Command loss weight")
    parser.add_argument("--loss-args-weight", type=float, default=2.0, help="Args loss weight")
    parser.add_argument(
        "--train-dino",
        action="store_true",
        help="Train DINO encoder (default: frozen)",
    )

    # Training args
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument("--tb-dir", type=str, default="runs/multimae", help="TensorBoard dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Logging args
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

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

    # Checkpoint args
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    setup_logging(
        level=args.log_level, log_file=args.log_file, rich_tracebacks=True, show_level=True
    )

    logger.info("=" * 60)
    logger.info(f"Random seed: {args.seed}")
    logger.info(
        "[bold cyan]Multi-modal MAE Training (SVG + Image)[/bold cyan]", extra={"markup": True}
    )
    logger.info("=" * 60)

    # Create config
    cfg = MultiMAEConfig()
    cfg.max_num_groups = args.max_num_groups
    cfg.max_seq_len = args.max_seq_len
    cfg.mask_ratio_svg = args.mask_ratio_svg
    cfg.mask_ratio_img = args.mask_ratio_img
    cfg.mae_depth = args.mae_depth
    cfg.mae_num_heads = args.mae_num_heads
    cfg.mae_mlp_ratio = args.mae_mlp_ratio
    cfg.mae_dropout = args.mae_dropout
    cfg.loss_cmd_weight = args.loss_cmd_weight
    cfg.loss_args_weight = args.loss_args_weight
    cfg.train_dino = args.train_dino
    cfg.use_precomputed_dino_patches = args.use_precomputed_dino_patches

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Create model
    logger.info("Creating model...")
    model = MultiMAE(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: [bold]{n_params:,}[/bold]", extra={"markup": True})
    n_params_svg_enc = sum(
        p.numel() for p in model.svg_group_encoder.parameters() if p.requires_grad
    )
    logger.info(
        f"SVG Group Encoder parameters: [bold]{n_params_svg_enc:,}[/bold]", extra={"markup": True}
    )
    n_params_mae = sum(p.numel() for p in model.mae_encoder.parameters() if p.requires_grad)
    logger.info(f"MAE Encoder parameters: [bold]{n_params_mae:,}[/bold]", extra={"markup": True})
    n_params_decoder = sum(p.numel() for p in model.svg_decoder.parameters() if p.requires_grad)
    logger.info(
        f"SVG Decoder parameters: [bold]{n_params_decoder:,}[/bold]", extra={"markup": True}
    )

    if model.image_encoder is not None:
        n_params_dino = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
        if n_params_dino > 0:
            logger.info(
                f"DINO Encoder parameters (trainable): [bold]{n_params_dino:,}[/bold]",
                extra={"markup": True},
            )
        else:
            logger.info("DINO Encoder: frozen (0 trainable parameters)")
    else:
        logger.info("Using precomputed DINO patch embeddings (no image encoder loaded)")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
            scheduler=None,
        )
        start_epoch = metadata["epoch"] + 1
        logger.info(f"Resuming training from epoch {start_epoch}", extra={"markup": True})

    # Prepare wandb config
    wandb_config = {
        # Model config
        "model_name": "multimae",
        "max_num_groups": cfg.max_num_groups,
        "max_seq_len": cfg.max_seq_len,
        "mask_ratio_svg": cfg.mask_ratio_svg,
        "mask_ratio_img": cfg.mask_ratio_img,
        "mae_depth": cfg.mae_depth,
        "mae_num_heads": cfg.mae_num_heads,
        "mae_mlp_ratio": cfg.mae_mlp_ratio,
        "mae_dropout": cfg.mae_dropout,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "dim_feedforward": cfg.dim_feedforward,
        "dropout": cfg.dropout,
        "loss_cmd_weight": cfg.loss_cmd_weight,
        "loss_args_weight": cfg.loss_args_weight,
        "train_dino": cfg.train_dino,
        "use_precomputed_dino_patches": cfg.use_precomputed_dino_patches,
        # Training args
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "num_workers": args.num_workers,
        "log_every": args.log_every,
        "mixed_precision": args.mixed_precision,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_params": n_params,
        "random_seed": args.seed,
    }

    if args.wandb_project:
        logger.info(
            f"Wandb enabled - project: [bold]{args.wandb_project}[/bold]", extra={"markup": True}
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        tb_dir=args.tb_dir,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
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
