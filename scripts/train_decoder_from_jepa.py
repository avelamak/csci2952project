"""
JEPA Decoder Training Script

Loads a pretrained JEPA checkpoint, transfers the encoder weights to an SVG
autoencoder, freezes the encoder, and trains the 2-stage decoder for SVG
reconstruction.

Usage:
    python scripts/train_decoder_from_jepa.py \
        --jepa-checkpoint checkpoints/jepa/best_model.pt \
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
from vecssl.models.config import _DefaultConfig, JepaConfig
from vecssl.models.jepa import Jepa

# from vecssl.trainer import Trainer  # only used indirectly via DebugTrainer
from vecssl.util import setup_logging

# Reuse components from test_svg_autoencoder
from test_svg_autoencoder import (
    SimpleSVGAutoencoder,
    DebugTrainer,
    custom_collate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def load_jepa_checkpoint(jepa_ckpt_path: Path, device: torch.device) -> tuple[Jepa, JepaConfig]:
    """
    Load JEPA checkpoint and return (model, config).
    """
    logger.info(f"Loading JEPA checkpoint from: {jepa_ckpt_path}")
    ckpt = torch.load(jepa_ckpt_path, map_location=device)

    cfg_obj = ckpt.get("cfg")
    if cfg_obj is None:
        logger.warning("No config found in checkpoint, using default JepaConfig")
        cfg = JepaConfig()
    elif isinstance(cfg_obj, JepaConfig):
        cfg = cfg_obj
    elif isinstance(cfg_obj, dict):
        logger.info("Reconstructing JepaConfig from checkpoint dict")
        cfg = JepaConfig()
        for key, value in cfg_obj.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    else:
        raise TypeError(f"Unexpected cfg type in checkpoint: {type(cfg_obj)}")

    if "model" not in ckpt:
        raise KeyError("Checkpoint missing 'model' key")

    jepa = Jepa(cfg)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()

    epoch = ckpt.get("epoch", "unknown")
    logger.info(
        "Loaded JEPA (epoch=%s, d_model=%d, d_joint=%d, n_layers=%d, max_num_groups=%d, max_seq_len=%d)",
        epoch,
        cfg.d_model,
        cfg.d_joint,
        cfg.n_layers,
        cfg.max_num_groups,
        cfg.max_seq_len,
    )

    return jepa, cfg


def build_ae_config_from_jepa(jepa_cfg: JepaConfig) -> _DefaultConfig:
    """
    Create SVGTransformer config matching JEPA encoder settings,
    configured as a 2-stage VAE autoencoder.
    """
    cfg = _DefaultConfig()

    # Encoder settings
    cfg.max_num_groups = jepa_cfg.max_num_groups
    cfg.max_seq_len = jepa_cfg.max_seq_len
    cfg.max_total_len = cfg.max_num_groups * cfg.max_seq_len
    cfg.d_model = jepa_cfg.d_model
    cfg.n_layers = jepa_cfg.n_layers
    cfg.n_heads = jepa_cfg.n_heads
    cfg.dim_feedforward = jepa_cfg.dim_feedforward
    cfg.dropout = jepa_cfg.dropout
    cfg.dim_z = jepa_cfg.dim_z
    cfg.use_resnet = jepa_cfg.use_resnet

    cfg.args_dim = jepa_cfg.args_dim
    cfg.n_args = jepa_cfg.n_args
    cfg.n_commands = jepa_cfg.n_commands

    # Decoder / autoencoder settings
    cfg.encode_stages = 2
    cfg.decode_stages = 2
    cfg.use_vae = True
    cfg.n_layers_decode = getattr(jepa_cfg, "n_layers_decode", 4)
    cfg.pred_mode = "autoregressive"
    cfg.self_match = False
    cfg.num_groups_proposal = cfg.max_num_groups

    logger.info(
        "Built AE config: encode_stages=%d decode_stages=%d use_vae=%s dim_z=%d",
        cfg.encode_stages,
        cfg.decode_stages,
        cfg.use_vae,
        cfg.dim_z,
    )
    return cfg


def transfer_encoder_weights(jepa: Jepa, ae_model: SimpleSVGAutoencoder) -> None:
    """
    Copy svg_encoder weights from JEPA to SVGTransformer.encoder.
    """
    logger.info("Transferring encoder weights from JEPA")
    jepa_encoder_state = jepa.svg_encoder.state_dict()
    ae_model.model.encoder.load_state_dict(jepa_encoder_state, strict=False)
    logger.info("Transferred %d encoder parameter tensors", len(jepa_encoder_state))

    if hasattr(jepa, "resnet") and hasattr(ae_model.model, "resnet"):
        ae_model.model.resnet.load_state_dict(jepa.resnet.state_dict())
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


def create_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    """
    Create train and val dataloaders (both over the same dataset).
    """
    logger.info("Creating dataset...")
    dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        train_ratio=1.0,  # use all data
        already_preprocessed=True,
    )
    logger.info("Dataset size: %d samples", len(dataset))

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        drop_last=False,
    )

    return train_loader, val_loader


def build_wandb_config(
    cfg: _DefaultConfig, args, device: torch.device, total: int, trainable: int, frozen: int
):
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
        "device": str(device),
        "n_params_total": total,
        "n_params_trainable": trainable,
        "n_params_frozen": frozen,
        "jepa_checkpoint": str(args.jepa_checkpoint),
        "encoder_frozen": True,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVG decoder from pretrained JEPA encoder")

    # JEPA checkpoint
    parser.add_argument(
        "--jepa-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained JEPA checkpoint",
    )

    # Dataset args
    parser.add_argument("--svg-dir", type=str, required=True, help="SVG directory")
    parser.add_argument("--img-dir", type=str, default=None, help="Image directory (optional)")
    parser.add_argument("--meta", type=str, required=True, help="Metadata CSV")

    # Model args (will be overridden by JEPA config where applicable)
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
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

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
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (print shapes & gradients)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        rich_tracebacks=True,
        show_level=True,
    )

    logger.info("=" * 60)
    logger.info("JEPA Decoder Training")
    logger.info("=" * 60)

    # Device
    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # 1. Load JEPA checkpoint
    jepa, jepa_cfg = load_jepa_checkpoint(Path(args.jepa_checkpoint), device)

    # 2. Override config values if specified
    if args.max_num_groups is not None:
        jepa_cfg.max_num_groups = args.max_num_groups
    if args.max_seq_len is not None:
        jepa_cfg.max_seq_len = args.max_seq_len

    # Store for dataloader
    args.max_num_groups = jepa_cfg.max_num_groups
    args.max_seq_len = jepa_cfg.max_seq_len

    # 3. Build autoencoder config
    cfg = build_ae_config_from_jepa(jepa_cfg)

    # 4. Create autoencoder wrapper
    logger.info("Creating autoencoder model...")
    model = SimpleSVGAutoencoder(cfg, debug_mode=args.debug)

    # 5. Transfer encoder weights from JEPA
    transfer_encoder_weights(jepa, model)

    # 6. Freeze encoder
    freeze_encoder(model)
    model.to(device)

    # Free JEPA memory
    del jepa
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
            device=device,
        )
        start_epoch = metadata["epoch"] + 1
        logger.info("Resuming training from epoch %d", start_epoch)

    # Prepare wandb config
    wandb_config = None
    if args.wandb_project:
        wandb_config = build_wandb_config(cfg, args, device, total, trainable, frozen)
        logger.info("Wandb enabled - project: %s", args.wandb_project)

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 9. Create trainer
    trainer = DebugTrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        device=device,
        grad_clip=args.grad_clip,
        tb_dir=args.tb_dir,
        amp=True,  # AMP for faster training
        check_gradients=args.debug,
        wandb_project=args.wandb_project,
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
