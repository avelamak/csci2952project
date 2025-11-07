"""
Minimal JEPA Test Script

Tests that the JEPA model forward + loss pipeline runs (smoke test).
This mirrors scripts/test_svg_autoencoder.py but builds a JEPA model.
"""

import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset
from vecssl.models.base import TrainStep
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


class SimpleJEPA(nn.Module):
    """
    Minimal wrapper for testing JEPA.

    The real `Jepa.forward` returns a TrainStep; this wrapper just delegates and
    exposes a `check_gradients` helper similar to the SVG test script.
    """

    def __init__(self, cfg: JepaConfig, debug_mode: bool = False):
        super().__init__()
        self.cfg = cfg
        self.debug_mode = debug_mode
        self.step_count = 0

        self.model = Jepa(cfg)

        logger.info("Created SimpleJEPA:")
        logger.info(f"  - masking_ratio: {getattr(cfg, 'masking_ratio', None)}")
        logger.info(f"  - max_num_groups: {cfg.max_num_groups}")
        logger.info(f"  - max_seq_len: {cfg.max_seq_len}")

    def forward(self, batch):
        device = next(self.parameters()).device

        # Move data to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        try:
            step = self.model(batch)
        except Exception as e:
            logger.error(f"Model forward failed: {e}")
            raise

        # Optionally print debug info on first step
        if self.debug_mode and self.step_count == 0:
            logger.info(f"JEPA forward returned TrainStep: loss={step.loss if hasattr(step, 'loss') else None}")

        self.step_count += 1
        return step

    def check_gradients(self):
        gradient_info = {}
        total_params = 0
        params_with_grad = 0
        max_grad = 0.0
        min_grad = float("inf")

        for name, param in self.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
                    grad_norm = param.grad.norm().item()
                    gradient_info[name] = {
                        "shape": tuple(param.shape),
                        "grad_norm": grad_norm,
                        "has_grad": True,
                    }
                    max_grad = max(max_grad, grad_norm)
                    min_grad = min(min_grad, grad_norm)
                else:
                    gradient_info[name] = {"shape": tuple(param.shape), "has_grad": False}

        summary = {
            "total_params": total_params,
            "params_with_grad": params_with_grad,
            "max_grad_norm": max_grad,
            "min_grad_norm": min_grad if min_grad != float("inf") else 0.0,
            "gradient_flow_percentage": (params_with_grad / total_params * 100)
            if total_params > 0
            else 0,
        }

        return gradient_info, summary


class DebugTrainer(Trainer):
    """A small Trainer extension to exercise one training loop and optionally check gradients."""

    def __init__(self, *args, check_gradients=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_gradients = check_gradients
        self.first_step_done = False

    def run(self, train_loader, val_loader=None, max_epochs=1, log_every=50):
        device = self.device
        self.model.to(device)
        scaler = torch.amp.grad_scaler.GradScaler(enabled=self.amp)

        for ep in range(max_epochs):
            self.model.train()
            for i, batch in enumerate(train_loader):
                with torch.amp.autocast_mode.autocast(device.type):
                    step = self.model(batch)
                    loss = step.loss
                scaler.scale(loss).backward()

                if self.check_gradients and not self.first_step_done:
                    grad_info, grad_summary = self.model.check_gradients()
                    logger.info(f"Grad summary: {grad_summary}")
                    self.first_step_done = True

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()


def create_dataloaders(args):
    """Create train and val dataloaders"""
    logger.info("Creating datasets...")

    train_dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        train_ratio=0.8,
        already_preprocessed=True,
    )

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


def main():
    parser = argparse.ArgumentParser(description="Test JEPA model")

    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="svgx_svgs", help="SVG directory")
    parser.add_argument("--img-dir", type=str, default="svgx_imgs", help="Image directory")
    parser.add_argument("--meta", type=str, default="svgx_meta.csv", help="Metadata CSV")

    # Model args
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max number of paths")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")

    # Training args
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--tb-dir", type=str, default="runs/test_jepa", help="TensorBoard dir")

    # Logging args
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file=args.log_file, rich_tracebacks=True, show_level=True)

    logger.info("=" * 60)
    logger.info("[bold cyan]JEPA Test[/bold cyan]", extra={"markup": True})
    logger.info("=" * 60)

    cfg = JepaConfig()
    cfg.max_num_groups = args.max_num_groups
    cfg.max_seq_len = args.max_seq_len

    train_loader, val_loader = create_dataloaders(args)

    logger.info("Creating model...")
    model = SimpleJEPA(cfg, debug_mode=args.debug)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: [bold]{n_params:,}[/bold]", extra={"markup": True})

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: [bold]{device}[/bold]", extra={"markup": True})

    trainer = DebugTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        grad_clip=args.grad_clip,
        tb_dir=args.tb_dir,
        amp=False,
        check_gradients=args.debug,
    )

    logger.info("Starting training...")
    try:
        trainer.run(train_loader=train_loader, val_loader=val_loader, max_epochs=args.epochs, log_every=args.log_every)
        logger.info("[bold green]✓ JEPA training completed (smoke test)![/bold green]", extra={"markup": True})
    except Exception as e:
        logger.error(f"[bold red]✗ JEPA training failed: {e}[/bold red]", extra={"markup": True})
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
