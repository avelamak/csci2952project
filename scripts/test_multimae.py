"""
Minimal MultiMAE Test Script

Tests that the Multi-modal MAE (SVG + Image) model forward + loss pipeline runs (smoke test).
"""

import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset
from vecssl.models.config import MultiMAEConfig
from vecssl.models.multimae import MultiMAE
from vecssl.trainer import Trainer
from vecssl.util import setup_logging, set_seed

from eval_utils import custom_collate

logger = logging.getLogger(__name__)


class SimpleMultiMAE(nn.Module):
    """
    Minimal wrapper for testing MultiMAE.
    """

    def __init__(self, cfg: MultiMAEConfig, debug_mode: bool = False):
        super().__init__()
        self.cfg = cfg
        self.debug_mode = debug_mode
        self.step_count = 0

        self.model = MultiMAE(cfg)

        logger.info("Created SimpleMultiMAE:")
        logger.info(f"  - max_num_groups: {cfg.max_num_groups}")
        logger.info(f"  - max_seq_len: {cfg.max_seq_len}")
        logger.info(f"  - mask_ratio_svg: {cfg.mask_ratio_svg}")
        logger.info(f"  - mask_ratio_img: {cfg.mask_ratio_img}")
        logger.info(f"  - mae_depth: {cfg.mae_depth}")
        logger.info(f"  - train_dino: {cfg.train_dino}")
        logger.info(f"  - use_precomputed_dino_patches: {cfg.use_precomputed_dino_patches}")
        logger.info(f"  - debug_mode: {debug_mode}")

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
                f"MultiMAE forward returned TrainStep: loss={step.loss if hasattr(step, 'loss') else None}"
            )
            if hasattr(step, "logs") and step.logs:
                for k, v in step.logs.items():
                    logger.info(f"  {k}: {v}")

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

    def _check_gradients_once(self):
        """Check gradient flow through the model (only on first step)"""
        if not self.check_gradients or self.first_step_done:
            return

        if not self.accelerator.is_main_process:
            self.first_step_done = True
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        logger.info(
            "[bold yellow]Debug: Gradient flow check[/bold yellow]",
            extra={"markup": True},
        )
        grad_info, grad_summary = unwrapped_model.check_gradients()

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
        for name, info in list(grad_info.items())[:5]:
            if info["has_grad"]:
                logger.info(f"    {name}: {info['grad_norm']:.6f}")
        logger.info("    ...")
        for name, info in list(grad_info.items())[-5:]:
            if info["has_grad"]:
                logger.info(f"    {name}: {info['grad_norm']:.6f}")

        self.first_step_done = True

    def run(self, train_loader, val_loader=None, max_epochs=10, log_every=50):
        from vecssl.util import make_progress

        # Prepare model, optimizer, and dataloaders
        if val_loader is not None:
            self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
                self._model, self._optimizer, train_loader, val_loader
            )
        else:
            self.model, self.optimizer, train_loader = self.accelerator.prepare(
                self._model, self._optimizer, train_loader
            )

        if self._scheduler is not None:
            self.scheduler = self.accelerator.prepare(self._scheduler)

        progress = make_progress()
        with progress:
            epoch_task = progress.add_task("epoch", total=max_epochs)
            for ep in range(max_epochs):
                self.model.train()
                batch_task = progress.add_task("train", total=len(train_loader))
                run_loss = 0.0

                for i, batch in enumerate(train_loader):
                    with self.accelerator.autocast():
                        step = self.model(batch)
                        loss = step.loss

                    self.accelerator.backward(loss)

                    # Check gradients on first step
                    self._check_gradients_once()

                    if self.grad_clip:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler:
                        self.scheduler.step()

                    run_loss += float(loss.detach())
                    global_step = ep * len(train_loader) + i

                    if self.accelerator.is_main_process and i % log_every == 0:
                        log_dict = {"train/loss": run_loss / (i + 1), "epoch": ep}
                        if step.logs:
                            for k, v in step.logs.items():
                                log_dict[f"train/{k}"] = v
                        self.accelerator.log(log_dict, step=global_step)

                    progress.advance(batch_task)

                progress.advance(epoch_task)
                if self.accelerator.is_main_process:
                    logger.info(f"epoch {ep + 1}/{max_epochs} done")

                if val_loader:
                    self.validate(val_loader, ep)

        self.accelerator.end_training()


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
    parser = argparse.ArgumentParser(description="Test Multi-modal MAE model (SVG + Image)")

    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="svgx_svgs", help="SVG directory")
    parser.add_argument("--img-dir", type=str, default="svgx_imgs", help="Image directory")
    parser.add_argument("--meta", type=str, default="svgx_meta.csv", help="Metadata CSV")
    parser.add_argument(
        "--use-precomputed-dino-patches",
        action="store_true",
        help="Use precomputed DINO patch embeddings",
    )
    parser.add_argument(
        "--dino-patches-dir",
        type=str,
        default=None,
        help="Directory containing precomputed DINO patch embeddings",
    )

    # Model args
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max number of paths")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument("--mask-ratio-svg", type=float, default=0.5, help="SVG masking ratio")
    parser.add_argument("--mask-ratio-img", type=float, default=0.75, help="Image masking ratio")
    parser.add_argument(
        "--mae-depth", type=int, default=4, help="MAE encoder depth (smaller for test)"
    )
    parser.add_argument(
        "--train-dino", action="store_true", help="Train DINO encoder (default: frozen)"
    )

    # Training args
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument("--tb-dir", type=str, default="runs/test_multimae", help="TensorBoard dir")

    # Logging args
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    setup_logging(
        level=args.log_level, log_file=args.log_file, rich_tracebacks=True, show_level=True
    )

    logger.info("=" * 60)
    logger.info("[bold cyan]MultiMAE Test (SVG + Image)[/bold cyan]", extra={"markup": True})
    logger.info("=" * 60)

    cfg = MultiMAEConfig()
    cfg.max_num_groups = args.max_num_groups
    cfg.max_seq_len = args.max_seq_len
    cfg.mask_ratio_svg = args.mask_ratio_svg
    cfg.mask_ratio_img = args.mask_ratio_img
    cfg.mae_depth = args.mae_depth
    cfg.train_dino = args.train_dino
    cfg.use_precomputed_dino_patches = args.use_precomputed_dino_patches

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Count parameters
    logger.info("Creating model...")
    model = SimpleMultiMAE(cfg, debug_mode=args.debug)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: [bold]{n_params:,}[/bold]", extra={"markup": True})

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Create trainer
    trainer = DebugTrainer(
        model=model,
        optimizer=optimizer,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        tb_dir=args.tb_dir,
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
