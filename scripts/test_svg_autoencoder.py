"""
Minimal SVG Autoencoder Test Script

Tests that the SVGTransformer encoder→decoder→loss pipeline works correctly.
Ignores images for now, focuses only on SVG reconstruction.
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
from vecssl.models.config import _DefaultConfig
from vecssl.models.loss import SVGLoss
from vecssl.trainer import Trainer
from vecssl.util import setup_logging

from vecssl.models.model import SVGTransformer

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


class SimpleSVGAutoencoder(nn.Module):
    """
    Minimal autoencoder wrapper for testing SVGTransformer.

    Flow:
        1. Encode: commands/args → latent z (+ mu, logsigma if VAE)
        2. Decode: z → reconstructed commands/args logits
        3. Loss: reconstruction loss (commands + args + KL if VAE)
    """

    def __init__(self, cfg: _DefaultConfig, debug_mode: bool = False):
        super().__init__()
        self.cfg = cfg
        self.debug_mode = debug_mode
        self.step_count = 0

        self.model = SVGTransformer(cfg)

        # Create loss function
        self.loss_fn = SVGLoss(cfg)

        # Loss weights (same as DeepSVG defaults)
        self.loss_weights = {
            "kl_tolerance": 0.1,
            "loss_kl_weight": 1.0,  # Will ramp up during training
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0,
            "loss_visibility_weight": 1.0,
        }

        logger.info("Created SimpleSVGAutoencoder:")
        logger.info(f"  - encode_stages: {cfg.encode_stages}")
        logger.info(f"  - decode_stages: {cfg.decode_stages}")
        logger.info(f"  - use_vae: {cfg.use_vae}")
        logger.info(f"  - max_num_groups: {cfg.max_num_groups}")
        logger.info(f"  - max_seq_len: {cfg.max_seq_len}")
        logger.info(f"  - debug_mode: {debug_mode}")

    def forward(self, batch):
        """
        Forward pass for training.

        Args:
            batch: dict with keys:
                - commands: [B, G, S] long tensor
                - args: [B, G, S, n_args] long tensor
                - (other keys ignored)

        Returns:
            TrainStep(loss, logs, extras)
        """
        device = next(self.parameters()).device

        # Move data to device
        commands = batch["commands"].to(device)
        args = batch["args"].to(device)

        # Debug: Print input shapes on first step
        if self.debug_mode and self.step_count == 0:
            logger.info("[bold yellow]Debug: Input shapes[/bold yellow]", extra={"markup": True})
            logger.info(f"  commands: {commands.shape} dtype={commands.dtype}")
            logger.info(f"  args: {args.shape} dtype={args.dtype}")

        # Prepare model args based on config
        if self.cfg.encode_stages == 1:
            # Flatten for 1-stage (not implemented yet, will fail gracefully)
            raise NotImplementedError("1-stage encoding not yet implemented in test script")
        # 2-stage: use separated format
        model_args = [commands, args, commands, args]  # encoder + decoder inputs

        # Forward pass through model
        try:
            output = self.model(*model_args, params={})
        except Exception as e:
            logger.error(f"Model forward failed: {e}")
            raise

        # Debug: Print output shapes on first step
        if self.debug_mode and self.step_count == 0:
            logger.info("[bold yellow]Debug: Output shapes[/bold yellow]", extra={"markup": True})
            for key, val in output.items():
                if isinstance(val, torch.Tensor):
                    logger.info(f"  {key}: {val.shape} dtype={val.dtype}")
                else:
                    logger.info(f"  {key}: {type(val)}")

        # Compute loss
        try:
            loss_dict = self.loss_fn(output, labels=None, weights=self.loss_weights)
        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            logger.error(f"Output keys: {output.keys()}")
            raise

        # Debug: Print loss values on first step
        if self.debug_mode and self.step_count == 0:
            logger.info("[bold yellow]Debug: Loss components[/bold yellow]", extra={"markup": True})
            for key, val in loss_dict.items():
                if isinstance(val, torch.Tensor):
                    logger.info(f"  {key}: {val.item():.6f}")

        # Extract individual losses for logging
        logs = {
            "loss_total": loss_dict["loss"].item(),
        }

        if "loss_kl" in loss_dict:
            logs["loss_kl"] = loss_dict["loss_kl"].item()
        if "loss_cmd" in loss_dict:
            logs["loss_cmd"] = loss_dict["loss_cmd"].item()
        if "loss_args" in loss_dict:
            logs["loss_args"] = loss_dict["loss_args"].item()
        if "loss_visibility" in loss_dict:
            logs["loss_visibility"] = loss_dict["loss_visibility"].item()

        self.step_count += 1

        return TrainStep(loss=loss_dict["loss"], logs=logs, extras={"output": output})

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

    def __init__(self, *args, check_gradients=False, wandb_project=None, cfg=None, **kwargs):
        super().__init__(*args, wandb_project=wandb_project, cfg=cfg, **kwargs)
        self.check_gradients = check_gradients
        self.first_step_done = False

    def run(
        self,
        train_loader,
        val_loader=None,
        max_epochs=10,
        log_every=50,
        save_every=1,
        start_epoch=0,
    ):
        device = self.device
        self.model.to(device)
        scaler = torch.amp.grad_scaler.GradScaler(enabled=self.amp)

        from vecssl.util import make_progress

        progress = make_progress()
        with progress:
            epoch_task = progress.add_task("epoch", total=max_epochs - start_epoch)
            for ep in range(start_epoch, max_epochs):
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
                    global_step = ep * len(train_loader) + i
                    if self.tb and i % log_every == 0:
                        self.tb.add_scalar("train/loss", run / (i + 1), global_step)
                        if step.logs:
                            for k, v in step.logs.items():
                                self.tb.add_scalar(f"train/{k}", v, global_step)
                    if self.wandb_run and i % log_every == 0:
                        wandb_logs = {
                            "train/loss": run / (i + 1),
                            "epoch": ep,
                            "step": global_step,
                        }
                        if step.logs:
                            for k, v in step.logs.items():
                                wandb_logs[f"train/{k}"] = v
                        self.wandb_run.log(wandb_logs)
                    progress.advance(batch_task)
                progress.advance(epoch_task)
                logger.info(f"epoch {ep + 1}/{max_epochs} done")

                # Save checkpoint periodically
                if self.checkpoint_dir and ep % save_every == 0 and ep > 1:
                    from vecssl.trainer import save_chkpt

                    save_chkpt(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        cfg=self.cfg,
                        checkpoint_dir=self.checkpoint_dir,
                        ep=ep,
                    )

                if val_loader:
                    self.validate(val_loader, ep)

        # Finish wandb run after training completes
        if self.wandb_run:
            import wandb

            wandb.finish()


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
        train_ratio=0.1,
        already_preprocessed=True,
    )

    # Validation dataset (20% of data)
    val_dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        train_ratio=0.1,
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
    parser = argparse.ArgumentParser(description="Test SVG Autoencoder")

    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="svgx_svgs", help="SVG directory")
    parser.add_argument("--img-dir", type=str, default="svgx_imgs", help="Image directory")
    parser.add_argument("--meta", type=str, default="svgx_meta.csv", help="Metadata CSV")

    # Model args
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max number of paths")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument("--encode-stages", type=int, default=2, help="Encoder stages (1 or 2)")
    parser.add_argument("--decode-stages", type=int, default=2, help="Decoder stages (1 or 2)")
    parser.add_argument("--use-vae", action="store_true", default=True, help="Use VAE")

    # Training args
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log-every", type=int, default=1, help="Log every N steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--tb-dir", type=str, default="runs/test_autoencoder", help="TensorBoard dir"
    )

    # Logging args
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file (optional)")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (print shapes & gradients)"
    )

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

    # Setup logging with Rich formatting
    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        rich_tracebacks=True,
        show_level=True,
    )

    logger.info("=" * 60)
    logger.info("[bold cyan]SVG Autoencoder Test[/bold cyan]", extra={"markup": True})
    logger.info("=" * 60)

    # Create config
    cfg = _DefaultConfig()
    cfg.max_num_groups = args.max_num_groups
    cfg.max_seq_len = args.max_seq_len
    cfg.encode_stages = args.encode_stages
    cfg.decode_stages = args.decode_stages
    cfg.use_vae = args.use_vae

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Create model
    logger.info("Creating model...")
    model = SimpleSVGAutoencoder(cfg, debug_mode=args.debug)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: [bold]{n_params:,}[/bold]", extra={"markup": True})

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup device
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
            "num_workers": args.num_workers,
            "log_every": args.log_every,
            "device": str(device),
            "amp": False,
            "n_params": n_params,
        }
        logger.info(
            f"Wandb enabled - project: [bold]{args.wandb_project}[/bold]", extra={"markup": True}
        )

    # Create trainer

    trainer = DebugTrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        device=device,
        grad_clip=args.grad_clip,
        tb_dir=args.tb_dir,
        amp=False,  # Disable AMP for debugging
        check_gradients=args.debug,
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
