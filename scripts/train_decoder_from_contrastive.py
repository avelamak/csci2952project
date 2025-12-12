"""
Contrastive Decoder Training Script

Loads a pretrained Contrastive checkpoint, transfers the encoder weights to an SVG
autoencoder, freezes the encoder, and trains the 2-stage decoder for SVG
reconstruction.

Usage:
    python scripts/train_decoder_from_contrastive.py \
        --contrastive-checkpoint checkpoints/contrastive/best_model.pt \
        --svg-dir svgx_svgs \
        --img-dir svgx_imgs \
        --meta svgx_meta.csv \
        --checkpoint-dir checkpoints/decoder \
        --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset
from vecssl.models.config import _DefaultConfig, ContrastiveConfig
from vecssl.models.contrastive import ContrastiveModel
from vecssl.trainer import Trainer

from vecssl.util import setup_logging, set_seed

# Reuse components from test_svg_autoencoder
from test_svg_autoencoder import (
    SimpleSVGAutoencoder,
    custom_collate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def load_contrastive_checkpoint(ckpt_path: Path) -> tuple[ContrastiveModel, ContrastiveConfig]:
    """
    Load Contrastive checkpoint and return (model, config).
    """
    logger.info(f"Loading Contrastive checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg_obj = ckpt.get("cfg")
    if cfg_obj is None:
        logger.warning("No config found in checkpoint, using default ContrastiveConfig")
        cfg = ContrastiveConfig()
    elif isinstance(cfg_obj, ContrastiveConfig):
        cfg = cfg_obj
    elif isinstance(cfg_obj, dict):
        logger.info("Reconstructing ContrastiveConfig from checkpoint dict")
        cfg = ContrastiveConfig()
        for key, value in cfg_obj.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    else:
        raise TypeError(f"Unexpected cfg type in checkpoint: {type(cfg_obj)}")

    if "model" not in ckpt:
        raise KeyError("Checkpoint missing 'model' key")

    contrastive = ContrastiveModel(cfg)
    contrastive.load_state_dict(ckpt["model"])
    contrastive.eval()

    epoch = ckpt.get("epoch", "unknown")
    logger.info(
        "Loaded Contrastive (epoch=%s, d_model=%d, d_joint=%d, n_layers=%d, max_num_groups=%d, max_seq_len=%d)",
        epoch,
        cfg.d_model,
        cfg.d_joint,
        cfg.n_layers,
        cfg.max_num_groups,
        cfg.max_seq_len,
    )

    return contrastive, cfg


def build_ae_config_from_contrastive(contrastive_cfg: ContrastiveConfig) -> _DefaultConfig:
    """
    Create SVGTransformer config matching Contrastive encoder settings,
    configured as a 2-stage VAE autoencoder.
    """
    cfg = _DefaultConfig()

    # Encoder settings
    cfg.max_num_groups = contrastive_cfg.max_num_groups
    cfg.max_seq_len = contrastive_cfg.max_seq_len
    cfg.max_total_len = cfg.max_num_groups * cfg.max_seq_len
    cfg.d_model = contrastive_cfg.d_model
    cfg.n_layers = contrastive_cfg.n_layers
    cfg.n_heads = contrastive_cfg.n_heads
    cfg.dim_feedforward = contrastive_cfg.dim_feedforward
    cfg.dropout = contrastive_cfg.dropout
    cfg.dim_z = contrastive_cfg.dim_z
    cfg.use_resnet = contrastive_cfg.use_resnet

    cfg.args_dim = contrastive_cfg.args_dim
    cfg.n_args = contrastive_cfg.n_args
    cfg.n_commands = contrastive_cfg.n_commands

    # Decoder / autoencoder settings
    cfg.encode_stages = 2
    cfg.decode_stages = 2
    cfg.use_vae = True
    cfg.n_layers_decode = getattr(contrastive_cfg, "n_layers_decode", 4)
    cfg.pred_mode = "one_shot"
    cfg.self_match = False
    cfg.num_groups_proposal = cfg.max_num_groups

    logger.info(
        "Built AE config: encode_stages=%d decode_stages=%d use_vae=%s dim_z=%d pred_mode=%s",
        cfg.encode_stages,
        cfg.decode_stages,
        cfg.use_vae,
        cfg.dim_z,
        cfg.pred_mode,
    )
    return cfg


def transfer_encoder_weights(contrastive: ContrastiveModel, ae_model: SimpleSVGAutoencoder) -> None:
    """
    Copy svg_encoder weights from Contrastive to SVGTransformer.encoder.
    """
    logger.info("Transferring encoder weights from Contrastive")
    contrastive_encoder_state = contrastive.encoder.state_dict()
    ae_model.model.encoder.load_state_dict(contrastive_encoder_state, strict=False)
    logger.info("Transferred %d encoder parameter tensors", len(contrastive_encoder_state))

    if hasattr(contrastive, "resnet") and hasattr(ae_model.model, "resnet"):
        ae_model.model.resnet.load_state_dict(contrastive.resnet.state_dict())
        logger.info("Transferred ResNet weights")


def freeze_encoder(ae_model: SimpleSVGAutoencoder) -> None:
    """
    Freeze encoder (and optional ResNet) parameters.
    """
    frozen = 0
    for p in ae_model.model.encoder.parameters():
        p.requires_grad = False
        frozen += 1

    if hasattr(ae_model.model, "resnet"):
        for p in ae_model.model.resnet.parameters():
            p.requires_grad = False
            frozen += 1

    logger.info("Froze %d encoder parameter tensors", frozen)


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """
    Return (total, trainable, frozen) parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


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
        drop_last=True,  # Drop incomplete batches for consistent batch sizes
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


def build_wandb_config(cfg: _DefaultConfig, args, total: int, trainable: int, frozen: int):
    return {
        # Model config
        "max_num_groups": cfg.max_num_groups,
        "max_seq_len": cfg.max_seq_len,
        "encode_stages": cfg.encode_stages,
        "decode_stages": cfg.decode_stages,
        "use_vae": cfg.use_vae,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_layers_decode": cfg.n_layers_decode,
        "n_heads": cfg.n_heads,
        "dim_feedforward": cfg.dim_feedforward,
        "dim_z": cfg.dim_z,
        "dropout": cfg.dropout,
        # Training args
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "mixed_precision": args.mixed_precision,
        "n_params_total": total,
        "n_params_trainable": trainable,
        "n_params_frozen": frozen,
        "contrastive_checkpoint": str(args.contrastive_checkpoint),
        "encoder_frozen": True,
        "random_seed": args.seed,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVG decoder from pretrained Contrastive encoder")

    # Contrastive checkpoint
    parser.add_argument(
        "--contrastive-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained Contrastive checkpoint",
    )

    # Dataset args
    parser.add_argument("--svg-dir", type=str, required=True, help="SVG directory")
    parser.add_argument("--img-dir", type=str, default=None, help="Image directory (optional)")
    parser.add_argument("--meta", type=str, required=True, help="Metadata CSV")

    # Model args (will be overridden by contrastive config where applicable)
    parser.add_argument("--max-num-groups", type=int, default=None, help="Override max_num_groups")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override max_seq_len")

    # Training args
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Checkpoint args
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/decoder",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Resume training from checkpoint"
    )

    # Logging args
    parser.add_argument("--tb-dir", type=str, default=None, help="TensorBoard directory")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb-name", type=str, default='contrastive_decoder', help="Wandb run name")
    parser.add_argument("--wandb-entity", type=str, default='vecssl', help="Wandb entity/team")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (print shapes & gradients)"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        rich_tracebacks=True,
        show_level=True,
    )

    logger.info("=" * 60)
    logger.info(f"Random seed: {args.seed}")
    logger.info("Contrastive Decoder Training")
    logger.info("=" * 60)

    # 1. Load Contrastive checkpoint
    contrastive, contrastive_cfg = load_contrastive_checkpoint(Path(args.contrastive_checkpoint))

    # 2. Override config values if specified
    if args.max_num_groups is not None:
        contrastive_cfg.max_num_groups = args.max_num_groups
    if args.max_seq_len is not None:
        contrastive_cfg.max_seq_len = args.max_seq_len

    # Store for dataloader
    args.max_num_groups = contrastive_cfg.max_num_groups
    args.max_seq_len = contrastive_cfg.max_seq_len
    
    logger.info(f"Training with {args.max_num_groups=}, {args.max_seq_len=}")

    # 3. Build autoencoder config
    cfg = build_ae_config_from_contrastive(contrastive_cfg)

    # 4. Create autoencoder wrapper
    logger.info("Creating autoencoder model...")
    model = SimpleSVGAutoencoder(cfg, debug_mode=args.debug)

    # 5. Transfer encoder weights from Contrastive
    transfer_encoder_weights(contrastive, model)

    # 6. Freeze encoder
    freeze_encoder(model)

    # Free Contrastive memory
    del contrastive
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Parameter counts
    total, trainable, frozen = count_parameters(model)
    logger.info(
        "Parameters: total=%s trainable=%s frozen=%s",
        f"{total:,}",
        f"{trainable:,}",
        f"{frozen:,}",
    )

    # 7. Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # 8. Optimizer - only trainable params (VAE + decoder)
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # Handle checkpoint resuming
    start_epoch = 0
    if args.resume_from:
        from vecssl.trainer import load_chkpt

        logger.info("Resuming from checkpoint: %s", args.resume_from)
        metadata = load_chkpt(
            checkpoint_path=args.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )
        start_epoch = metadata["epoch"] + 1
        logger.info("Resuming training from epoch %d", start_epoch)

    # Prepare wandb config
    wandb_config = None
    if args.wandb_project:
        wandb_config = build_wandb_config(cfg, args, total, trainable, frozen)
        logger.info("Wandb enabled - project: %s", args.wandb_project)

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        tb_dir=args.tb_dir,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        cfg=wandb_config,
    )

    # 10. Run training
    logger.info("Starting decoder training...")
    try:
        trainer.run(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.epochs,
            log_every=args.log_every,
            save_every=args.save_every,
            start_epoch=start_epoch,
        )
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error("Training failed: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
