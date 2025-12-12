"""
MultiMAE Decoder Training Script

Loads a pretrained MultiMAE checkpoint, freezes the encoder, and trains a decoder
to reconstruct SVG from the CLS embedding.

Usage:
    python scripts/train_decoder_from_multimae.py \
        --multimae-checkpoint checkpoints/multimae/best_model.pt \
        --svg-dir data/fonts_svg \
        --img-dir data/fonts_img \
        --meta data/fonts_meta.csv \
        --checkpoint-dir checkpoints/decoder_multimae \
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
from vecssl.models.base import TrainStep
from vecssl.models.config import MultiMAEConfig, _DefaultConfig
from vecssl.models.multimae import MultiMAE
from vecssl.models.model import SVGTransformer
from vecssl.models.loss import SVGLoss
from vecssl.trainer import Trainer
from vecssl.util import setup_logging, set_seed

logger = logging.getLogger(__name__)


# =============================================================================
# Custom collate function
# =============================================================================


def custom_collate(batch):
    """Standard collate for SVGXDataset."""
    collated = {
        "commands": torch.stack([item["commands"] for item in batch]),
        "args": torch.stack([item["args"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "tensors": [item["tensors"] for item in batch],
        "uuid": [item["uuid"] for item in batch],
        "name": [item["name"] for item in batch],
        "source": [item["source"] for item in batch],
        "glyph_label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "family_label": torch.tensor([item["family_label"] for item in batch], dtype=torch.long),
    }

    if "dino_embedding" in batch[0]:
        collated["dino_embedding"] = torch.stack([item["dino_embedding"] for item in batch])
    if "dino_patches" in batch[0]:
        collated["dino_patches"] = torch.stack([item["dino_patches"] for item in batch])

    return collated


# =============================================================================
# Model wrapper
# =============================================================================


class DecoderFromMultiMAE(nn.Module):
    """
    Frozen MultiMAE encoder + trainable SVG decoder.

    Flow:
        1. Get CLS embedding from frozen MultiMAE via encode_joint()
        2. Reshape CLS to match decoder expected input
        3. Decode to SVG commands/args
        4. Compute reconstruction loss
    """

    def __init__(
        self,
        frozen_encoder: MultiMAE,
        decoder_cfg: _DefaultConfig,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.encoder = frozen_encoder
        self.cfg = decoder_cfg
        self.debug_mode = debug_mode
        self.step_count = 0

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Create SVGTransformer for decoder (encoder will be bypassed)
        self.svg_transformer = SVGTransformer(decoder_cfg)

        # Freeze the encoder part of SVGTransformer (we'll use MultiMAE instead)
        if hasattr(self.svg_transformer, "encoder"):
            for p in self.svg_transformer.encoder.parameters():
                p.requires_grad = False

        # Create loss function
        self.loss_fn = SVGLoss(decoder_cfg)

        # Loss weights
        self.loss_weights = {
            "kl_tolerance": 0.1,
            "loss_kl_weight": 1.0,
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0,
            "loss_visibility_weight": 1.0,
        }

        logger.info("Created DecoderFromMultiMAE:")
        logger.info("  - Frozen MultiMAE encoder")
        logger.info(f"  - decode_stages: {decoder_cfg.decode_stages}")
        logger.info(f"  - use_vae: {decoder_cfg.use_vae}")

    def forward(self, batch):
        device = next(self.svg_transformer.parameters()).device

        # Move data to device
        commands = batch["commands"].to(device)
        args = batch["args"].to(device)

        # Move image/dino_patches for MultiMAE
        batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Get CLS embedding from frozen MultiMAE
        with torch.no_grad():
            joint = self.encoder.encode_joint(batch_device)
            cls_embed = joint["svg"]  # [N, d_model]

        # Reshape CLS to match decoder expected input: (N, d_model) -> (N, 1, 1, d_model)
        # After _make_seq_first: (1, 1, N, d_model)
        z = cls_embed.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, d_model)

        # Debug: Print shapes on first step
        if self.debug_mode and self.step_count == 0:
            logger.info("[bold yellow]Debug: Input shapes[/bold yellow]", extra={"markup": True})
            logger.info(f"  commands: {commands.shape}")
            logger.info(f"  args: {args.shape}")
            logger.info(f"  cls_embed: {cls_embed.shape}")
            logger.info(f"  z (reshaped): {z.shape}")

        # Forward through SVGTransformer with pre-computed z
        # Pass None for encoder inputs since we're providing z directly
        output = self.svg_transformer(
            commands_enc=None,
            args_enc=None,
            commands_dec=commands,
            args_dec=args,
            z=z,
            params={},
        )

        # Compute loss
        loss_dict = self.loss_fn(output, labels=None, weights=self.loss_weights)

        logs = {
            "loss_total": loss_dict["loss"].item(),
        }
        if "loss_cmd" in loss_dict:
            logs["loss_cmd"] = loss_dict["loss_cmd"].item()
        if "loss_args" in loss_dict:
            logs["loss_args"] = loss_dict["loss_args"].item()
        if "loss_visibility" in loss_dict:
            logs["loss_visibility"] = loss_dict["loss_visibility"].item()

        self.step_count += 1

        return TrainStep(loss=loss_dict["loss"], logs=logs, extras={"output": output})


# =============================================================================
# Helper Functions
# =============================================================================


def load_multimae_checkpoint(ckpt_path: Path) -> tuple[MultiMAE, MultiMAEConfig]:
    """Load MultiMAE checkpoint and return (model, config)."""
    logger.info(f"Loading MultiMAE checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg_obj = ckpt.get("cfg")
    if cfg_obj is None:
        logger.warning("No config found in checkpoint, using default MultiMAEConfig")
        cfg = MultiMAEConfig()
    elif isinstance(cfg_obj, MultiMAEConfig):
        cfg = cfg_obj
    elif isinstance(cfg_obj, dict):
        logger.info("Reconstructing MultiMAEConfig from checkpoint dict")
        cfg = MultiMAEConfig()
        for key, value in cfg_obj.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    else:
        raise TypeError(f"Unexpected cfg type in checkpoint: {type(cfg_obj)}")

    if "model" not in ckpt:
        raise KeyError("Checkpoint missing 'model' key")

    multimae = MultiMAE(cfg)

    state_dict = ckpt["model"]
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
        }

    multimae.load_state_dict(state_dict)
    multimae.eval()

    epoch = ckpt.get("epoch", "unknown")
    logger.info(
        "Loaded MultiMAE (epoch=%s, d_model=%d, max_num_groups=%d, max_seq_len=%d)",
        epoch,
        cfg.d_model,
        cfg.max_num_groups,
        cfg.max_seq_len,
    )

    return multimae, cfg


def build_decoder_config(multimae_cfg: MultiMAEConfig) -> _DefaultConfig:
    """
    Create SVGTransformer config for decoder, matching MultiMAE settings.
    """
    cfg = _DefaultConfig()

    # Match MultiMAE encoder settings
    cfg.max_num_groups = multimae_cfg.max_num_groups
    cfg.max_seq_len = multimae_cfg.max_seq_len
    cfg.max_total_len = cfg.max_num_groups * cfg.max_seq_len
    cfg.d_model = multimae_cfg.d_model
    cfg.n_layers = getattr(multimae_cfg, "n_layers", 4)
    cfg.n_heads = getattr(multimae_cfg, "n_heads", 8)
    cfg.dim_feedforward = getattr(multimae_cfg, "dim_feedforward", 512)
    cfg.dropout = getattr(multimae_cfg, "dropout", 0.1)
    cfg.dim_z = multimae_cfg.d_model  # CLS embedding dimension

    cfg.args_dim = multimae_cfg.args_dim
    cfg.n_args = multimae_cfg.n_args
    cfg.n_commands = multimae_cfg.n_commands

    # Decoder settings
    cfg.encode_stages = 0  # No encoding (using MultiMAE CLS)
    cfg.decode_stages = 2  # 2-stage hierarchical decoding
    cfg.use_vae = False  # No VAE (CLS is already deterministic)
    cfg.n_layers_decode = getattr(multimae_cfg, "n_layers_decode", 4)
    cfg.pred_mode = "one_shot"
    cfg.self_match = False
    cfg.num_groups_proposal = cfg.max_num_groups

    logger.info(
        "Built decoder config: encode_stages=%d decode_stages=%d dim_z=%d pred_mode=%s",
        cfg.encode_stages,
        cfg.decode_stages,
        cfg.dim_z,
        cfg.pred_mode,
    )
    return cfg


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Return (total, trainable, frozen) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def create_dataloaders(args):
    """Create train and val dataloaders."""
    logger.info("Creating datasets...")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        drop_last=True,
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


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SVG decoder from pretrained MultiMAE encoder"
    )

    # MultiMAE checkpoint
    parser.add_argument(
        "--multimae-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained MultiMAE checkpoint",
    )

    # Dataset args
    parser.add_argument("--svg-dir", type=str, required=True, help="SVG directory")
    parser.add_argument("--img-dir", type=str, default=None, help="Image directory")
    parser.add_argument("--meta", type=str, required=True, help="Metadata CSV")

    # Model args
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Checkpoint args
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/decoder_multimae",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")

    # Logging args
    parser.add_argument("--tb-dir", type=str, default=None, help="TensorBoard directory")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity/team")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    set_seed(args.seed)

    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        rich_tracebacks=True,
        show_level=True,
    )

    logger.info("=" * 60)
    logger.info(f"Random seed: {args.seed}")
    logger.info("MultiMAE Decoder Training")
    logger.info("=" * 60)

    # 1. Load MultiMAE checkpoint
    multimae, multimae_cfg = load_multimae_checkpoint(Path(args.multimae_checkpoint))

    # 2. Override config values if specified
    if args.max_num_groups is not None:
        multimae_cfg.max_num_groups = args.max_num_groups
    if args.max_seq_len is not None:
        multimae_cfg.max_seq_len = args.max_seq_len

    args.max_num_groups = multimae_cfg.max_num_groups
    args.max_seq_len = multimae_cfg.max_seq_len
    
    logger.info(f"Training with {args.max_num_groups}")

    # 3. Build decoder config
    decoder_cfg = build_decoder_config(multimae_cfg)

    # 4. Create model wrapper
    logger.info("Creating decoder model...")
    model = DecoderFromMultiMAE(multimae, decoder_cfg, debug_mode=args.debug)

    # Free memory
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

    # 5. Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # 6. Optimizer - only trainable params
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
        wandb_config = {
            "max_num_groups": decoder_cfg.max_num_groups,
            "max_seq_len": decoder_cfg.max_seq_len,
            "decode_stages": decoder_cfg.decode_stages,
            "d_model": decoder_cfg.d_model,
            "dim_z": decoder_cfg.dim_z,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "grad_clip": args.grad_clip,
            "mixed_precision": args.mixed_precision,
            "n_params_total": total,
            "n_params_trainable": trainable,
            "n_params_frozen": frozen,
            "multimae_checkpoint": str(args.multimae_checkpoint),
            "encoder_frozen": True,
            "random_seed": args.seed,
        }

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

    # Run training
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
