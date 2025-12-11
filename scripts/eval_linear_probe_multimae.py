"""
Linear Probe Evaluation for MultiMAE

Train a linear classifier on frozen MultiMAE CLS latents to evaluate representation quality.

Usage:
    python scripts/eval_linear_probe_multimae.py \
        --checkpoint checkpoints/multimae/best_model.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv \
        --epochs 10 --lr 1e-3 --task family_label
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vecssl.models.base import TrainStep
from vecssl.models.config import MultiMAEConfig
from vecssl.models.multimae import MultiMAE
from vecssl.trainer import Trainer
from vecssl.util import setup_logging

from eval_utils import (
    add_common_args,
    create_eval_dataloader,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Model wrapper
# --------------------------------------------------------


class FrozenEncoderLinearProbe(nn.Module):
    """
    Frozen MultiMAE encoder + trainable linear classifier.

    Uses encoder.encode_joint()['svg'] (CLS token) as latent.
    """

    def __init__(self, encoder: MultiMAE, num_classes: int, embed_dim: int, label: str):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.label = label

        # Freeze encoder parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, batch):
        device = next(self.classifier.parameters()).device

        # Move batch tensors to device
        batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Get latent from frozen encoder (CLS embedding)
        with torch.no_grad():
            joint = self.encoder.encode_joint(batch_device)
            z = joint["svg"]  # [B, d_model]

        # Classify
        logits = self.classifier(z)
        labels = batch_device[self.label]

        # Filter out samples with invalid labels (-1)
        valid_mask = labels >= 0
        if not valid_mask.any():
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            acc = 0.0
        else:
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask]

            loss = F.cross_entropy(valid_logits, valid_labels)
            preds = valid_logits.argmax(dim=-1)
            acc = (preds == valid_labels).float().mean().item()

        logs = {
            "probe_loss": loss.item(),
            "probe_acc": acc,
        }
        return TrainStep(loss=loss, logs=logs)


@torch.no_grad()
def evaluate(model, loader, label, accelerator):
    """Evaluate probe accuracy on validation set."""
    model.eval()
    correct = 0
    total = 0

    unwrapped = accelerator.unwrap_model(model)
    device = accelerator.device

    for batch in loader:
        batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        joint = unwrapped.encoder.encode_joint(batch_device)
        z = joint["svg"]
        logits = unwrapped.classifier(z)
        labels = batch_device[label]

        valid_mask = labels >= 0
        if valid_mask.any():
            preds = logits[valid_mask].argmax(dim=-1)
            correct += (preds == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()

    acc = correct / max(1, total)
    return acc


# --------------------------------------------------------
# MultiMAE loader
# --------------------------------------------------------


def load_multimae_encoder(ckpt_path: Path) -> tuple[MultiMAE, MultiMAEConfig]:
    logger.info(f"Loading MultiMAE from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg_obj = ckpt.get("cfg")
    if isinstance(cfg_obj, MultiMAEConfig):
        cfg = cfg_obj
    elif isinstance(cfg_obj, dict):
        cfg = MultiMAEConfig()
        for k, v in cfg_obj.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        logger.warning("No config in checkpoint, using default MultiMAEConfig")
        cfg = MultiMAEConfig()

    model = MultiMAE(cfg)

    state_dict = ckpt["model"]
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
        }

    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


# --------------------------------------------------------
# Main
# --------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Linear probe on MultiMAE CLS latent")
    parser = add_common_args(parser)

    # Encoder args
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to MultiMAE checkpoint",
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument(
        "--task",
        type=str,
        default="glyph_label",
        help="Training task: 'glyph_label' or 'family_label'",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=100,
        help="Minimum samples per class for stratified split",
    )

    # Output / logging
    parser.add_argument("--tb-dir", type=str, default=None, help="TensorBoard directory")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project")
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity/team")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Save checkpoints")

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("MultiMAE Linear Probe Evaluation")
    logger.info("=" * 60)

    # Load frozen encoder
    encoder, cfg = load_multimae_encoder(Path(args.checkpoint))

    # Embedding dimension is d_model (CLS output size)
    embed_dim = cfg.d_model

    # Stratify by family_label if task is family_label
    stratify_by = "family_label" if args.task == "family_label" else None

    # Dataloaders (same as your existing eval scripts)
    train_loader = create_eval_dataloader(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="train",
        shuffle=True,
        stratify_by=stratify_by,
        min_class_count=args.min_class_count,
    )

    val_loader = create_eval_dataloader(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="val",
        shuffle=False,
        stratify_by=stratify_by,
        min_class_count=args.min_class_count,
    )

    # Infer number of classes
    labels_in_dataset = set()
    label_key = args.task
    for batch in train_loader:
        for lbl in batch[label_key].tolist():
            if lbl >= 0:
                labels_in_dataset.add(lbl)
    num_classes = max(labels_in_dataset) + 1 if labels_in_dataset else 62

    logger.info(f"Training task: {label_key}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Embedding dimension (CLS): {embed_dim}")

    # Probe model
    model = FrozenEncoderLinearProbe(encoder, num_classes, embed_dim, label_key)

    # Only classifier is trainable
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    wandb_config = (
        {
            "encoder_type": "multimae",
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "task": label_key,
        }
        if args.wandb_project
        else None
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        mixed_precision="no",
        tb_dir=args.tb_dir,
        grad_clip=None,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        cfg=wandb_config,
    )

    logger.info("Starting linear probe training...")
    trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.epochs,
        log_every=args.log_every,
        save_every=args.epochs,  # save at the end
        start_epoch=0,
    )

    final_acc = evaluate(trainer.model, val_loader, label_key, trainer.accelerator)
    logger.info(f"Final validation accuracy: {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
