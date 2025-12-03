"""
Linear Probe Evaluation Script

Train a linear classifier on frozen encoder latents to evaluate representation quality.
Supports JEPA, Contrastive, and Autoencoder encoders.

Usage:
    python scripts/eval_linear_probe.py \
        --encoder-type jepa \
        --checkpoint checkpoints/jepa/best_model.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv \
        --epochs 10 --lr 1e-3 --task family_label

    # Or with autoencoder:
    python scripts/eval_linear_probe.py \
        --encoder-type autoencoder \
        --checkpoint checkpoints/ae/checkpoint.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vecssl.models.base import TrainStep
from vecssl.trainer import Trainer
from vecssl.util import setup_logging

from eval_utils import (
    add_common_args,
    create_eval_dataloader,
    load_encoder,
)

logger = logging.getLogger(__name__)


class FrozenEncoderLinearProbe(nn.Module):
    """
    Frozen encoder + trainable linear classifier.

    Uses encode_joint()['svg'] as the latent representation.
    """

    def __init__(self, encoder, num_classes: int, embed_dim: int, label: str):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.label = label

    def forward(self, batch):
        device = next(self.classifier.parameters()).device

        # Move batch tensors to device
        batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Get latent from frozen encoder
        with torch.no_grad():
            joint = self.encoder.encode_joint(batch_device)
            z = joint["svg"]  # [B, d_joint]

        # Classify
        logits = self.classifier(z)
        labels = batch_device[self.label]

        # Filter out samples with invalid labels (-1)
        valid_mask = labels >= 0
        if not valid_mask.any():
            # No valid samples in batch
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
    """Evaluate model accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0

    # Get unwrapped model for accessing encoder
    unwrapped = accelerator.unwrap_model(model)
    device = accelerator.device

    for batch in loader:
        # Move batch to device (loader may not be prepared)
        batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        joint = unwrapped.encoder.encode_joint(batch_device)
        z = joint["svg"]
        logits = unwrapped.classifier(z)
        labels = batch_device[label]
        # Filter valid labels
        valid_mask = labels >= 0
        if valid_mask.any():
            preds = logits[valid_mask].argmax(dim=-1)
            correct += (preds == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()

    acc = correct / max(1, total)
    return acc


def main():
    parser = argparse.ArgumentParser(description="Linear probe on frozen encoder latent")
    parser = add_common_args(parser)

    # Encoder args
    parser.add_argument(
        "--encoder-type",
        type=str,
        required=True,
        choices=["jepa", "contrastive", "autoencoder"],
        help="Type of encoder",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to encoder checkpoint",
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument(
        "--task",
        type=str,
        default="glyph_label",
        help="Training task to predict: glyph_label for glyph label and family_label for family label",
    )

    # Output args
    parser.add_argument("--tb-dir", type=str, default=None, help="TensorBoard directory")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project")
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity/team")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Save checkpoints")

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("Linear Probe Evaluation")
    logger.info("=" * 60)

    # Load frozen encoder
    encoder, cfg = load_encoder(Path(args.checkpoint), args.encoder_type)
    cfg.training_task = args.task

    # Determine embedding dimension
    if args.encoder_type == "jepa":
        embed_dim = cfg.d_joint
    elif args.encoder_type == "contrastive":
        embed_dim = cfg.joint_dim
    else:  # autoencoder
        embed_dim = cfg.dim_z

    # Create train and val dataloaders (separate splits to avoid data leakage)
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
    )

    # Determine number of classes from dataset
    labels_in_dataset = set()
    label = cfg.training_task
    for batch in train_loader:
        for lbl in batch[label].tolist():
            if lbl >= 0:
                labels_in_dataset.add(lbl)
    num_classes = max(labels_in_dataset) + 1 if labels_in_dataset else 62

    logger.info(f"Training task: {label}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Embedding dimension: {embed_dim}")

    # Create probe model
    model = FrozenEncoderLinearProbe(encoder, num_classes, embed_dim, label)

    # Only train the classifier
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    # Setup trainer
    wandb_config = (
        {
            "encoder_type": args.encoder_type,
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "task": label,
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

    # Train
    logger.info("Starting linear probe training...")
    trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.epochs,
        log_every=args.log_every,
        save_every=args.epochs,  # Save at end
        start_epoch=0,
    )

    # Final evaluation on validation set
    final_acc = evaluate(trainer.model, val_loader, label, trainer.accelerator)
    logger.info(f"Final accuracy: {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
