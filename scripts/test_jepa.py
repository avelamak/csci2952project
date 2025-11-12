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
        logger.info(f"  - max_num_groups: {cfg.max_num_groups}")
        logger.info(f"  - max_seq_len: {cfg.max_seq_len}")
        logger.info(f"  - d_joint: {cfg.d_joint}")
        logger.info(f"  - debug_mode: {debug_mode}")
        logger.info(f"  - use_resnet: {cfg.use_resnet}")

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
            logger.info(
                f"JEPA forward returned TrainStep: loss={step.loss if hasattr(step, 'loss') else None}"
            )

        self.step_count += 1
        return step

    def check_gradients(self):
        """Check gradient flow through the model"""
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
    """Extended trainer with gradient checking"""

    def __init__(self, *args, check_gradients=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_gradients = check_gradients
        self.first_step_done = False

    def run(self, train_loader, val_loader=None, max_epochs=10, log_every=50):
        device = self.device
        self.model.to(device)
        scaler = torch.amp.grad_scaler.GradScaler(enabled=self.amp)

        from vecssl.util import make_progress

        progress = make_progress()
        with progress:
            epoch_task = progress.add_task("epoch", total=max_epochs)
            for ep in range(max_epochs):
                self.model.train()
                batch_task = progress.add_task("train", total=len(train_loader))
                run = 0.0
                for i, batch in enumerate(train_loader):
                    with torch.amp.autocast_mode.autocast(device.type):
                        step = self.model(batch)
                        loss = step.loss
                    scaler.scale(loss).backward()

                    # Check gradients on first step
                    if self.check_gradients and not self.first_step_done:
                        logger.info(
                            "[bold yellow]Debug: Gradient flow check[/bold yellow]",
                            extra={"markup": True},
                        )
                        grad_info, grad_summary = self.model.check_gradients()

                        logger.info(
                            f"  Total params: [bold]{grad_summary['total_params']}[/bold]",
                            extra={"markup": True},
                        )
                        logger.info(
                            f"  Params with grad: [bold]{grad_summary['params_with_grad']}[/bold] "
                            f"({grad_summary['gradient_flow_percentage']:.1f}%)",
                            extra={"markup": True},
                        )
                        logger.info(f"  Max grad norm: {grad_summary['max_grad_norm']:.6f}")
                        logger.info(f"  Min grad norm: {grad_summary['min_grad_norm']:.6f}")

                        # Show first few and last few layers
                        logger.info("  Sample gradient norms:")
                        shown = 0
                        for name, info in list(grad_info.items())[:5]:
                            if info["has_grad"]:
                                logger.info(f"    {name}: {info['grad_norm']:.6f}")
                                shown += 1
                        logger.info("    ...")
                        for name, info in list(grad_info.items())[-5:]:
                            if info["has_grad"]:
                                logger.info(f"    {name}: {info['grad_norm']:.6f}")

                        self.first_step_done = True

                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                    run += float(loss.detach())
                    if self.tb and i % log_every == 0:
                        self.tb.add_scalar("train/loss", run / (i + 1), ep * len(train_loader) + i)
                        if step.logs:
                            for k, v in step.logs.items():
                                self.tb.add_scalar(f"train/{k}", v, ep * len(train_loader) + i)
                    progress.advance(batch_task)
                progress.advance(epoch_task)
                logger.info(f"epoch {ep + 1}/{max_epochs} done")

                if val_loader:
                    self.validate(val_loader, ep)


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
    parser = argparse.ArgumentParser(description="Test JEPA model")

    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="svgx_svgs", help="SVG directory")
    parser.add_argument("--img-dir", type=str, default="svgx_imgs", help="Image directory")
    parser.add_argument("--meta", type=str, default="svgx_meta.csv", help="Metadata CSV")

    # Model args
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max number of paths")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument("--use-resnet", action="store_true", default=True, help="Use ResNet")

    # Training args
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
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

    setup_logging(
        level=args.log_level, log_file=args.log_file, rich_tracebacks=True, show_level=True
    )

    logger.info("=" * 60)
    logger.info("[bold cyan]JEPA Test[/bold cyan]", extra={"markup": True})
    logger.info("=" * 60)

    cfg = JepaConfig()
    cfg.max_num_groups = args.max_num_groups
    cfg.max_seq_len = args.max_seq_len
    cfg.use_resnet = args.use_resnet

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Count parameters
    logger.info("Creating model...")
    model = SimpleJEPA(cfg, debug_mode=args.debug)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: [bold]{n_params:,}[/bold]", extra={"markup": True})

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create trainer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: [bold]{device}[/bold]", extra={"markup": True})

    trainer = DebugTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        grad_clip=args.grad_clip,
        tb_dir=args.tb_dir,
        amp=False,  # Disable AMP for debugging
        check_gradients=args.debug,
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
