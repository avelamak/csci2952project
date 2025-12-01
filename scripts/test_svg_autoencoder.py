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
from vecssl.data.geom import Bbox
from vecssl.data.svg import SVG
from vecssl.data.svg_tensor import SVGTensor
from vecssl.models.base import TrainStep
from vecssl.models.config import _DefaultConfig
from vecssl.models.loss import SVGLoss
from vecssl.trainer import Trainer
from vecssl.util import setup_logging

from vecssl.models.model import SVGTransformer

logger = logging.getLogger(__name__)


def decode_svg_from_cmd_args(
    commands: torch.Tensor,
    args: torch.Tensor,
    viewbox_size: int = 256,  # Match ARGS_DIM quantization range
    pad_val: int = -1,
) -> SVG:
    """
    Decode commands and args tensors back to an SVG object.

    Args:
        commands: [S] or [G, S] long tensor of command indices
        args: [S, 11] or [G, S, 11] float tensor of arguments
        viewbox_size: Size of the SVG viewbox (default 24)
        pad_val: Padding value used (default -1)

    Returns:
        SVG object that can be saved via .save_svg()
    """
    # If we have multiple groups, just take the first one for debugging
    if commands.ndim == 2:
        commands = commands[0]
        args = args[0]

    commands = commands.detach().cpu()
    args = args.detach().cpu().float()

    # Find actual sequence length (before EOS/padding)
    eos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
    valid_mask = (commands != pad_val) & (commands != eos_idx)

    if valid_mask.any():
        # Find last valid command
        seq_len = valid_mask.long().sum().item()
    else:
        seq_len = 0

    if seq_len == 0:
        # Return empty SVG
        return SVG([], viewbox=Bbox(viewbox_size))

    # Truncate to valid length
    commands = commands[:seq_len]
    args = args[:seq_len]

    # Create SVGTensor from commands and args
    svg_tensor = SVGTensor.from_cmd_args(
        commands,
        args,
        PAD_VAL=pad_val,
    )

    # Get full data tensor (14 columns) and convert to SVG
    tensor_data = svg_tensor.data
    svg = SVG.from_tensor(tensor_data, viewbox=Bbox(viewbox_size), allow_empty=True)

    return svg


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

    @torch.no_grad()
    def encode_joint(self, batch):
        """
        Encode batch to joint embedding space (compatible with JEPA/Contrastive interface).

        For autoencoder, we use the VAE mu (mean) as the embedding for deterministic eval.
        Image embedding uses the same z (no separate image encoder).

        Returns:
            dict: {"svg": z, "img": z} where z is [B, dim_z]
        """
        from vecssl.util import _make_seq_first

        device = next(self.parameters()).device
        commands = batch["commands"].to(device)
        args = batch["args"].to(device)

        # Replicate encoding path but use mu for deterministic eval (no VAE sampling noise)
        commands_enc, args_enc = _make_seq_first(commands, args)
        z = self.model.encoder(commands_enc, args_enc, label=None)

        if self.cfg.use_resnet:
            z = self.model.resnet(z)

        # Use mu directly (not sampled z) for deterministic embedding
        mu = self.model.vae.enc_mu_fcn(z)

        # Shape: [1, 1, B, dim_z] → [B, dim_z]
        mu = mu.squeeze(0).squeeze(0)

        # Return same embedding for both modalities (AE has no separate image encoder)
        return {"svg": mu, "img": mu}

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
    """Extended trainer with gradient checking and visualization"""

    def __init__(self, *args, check_gradients=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_gradients = check_gradients
        self.first_step_done = False

    def _maybe_visualize_batch(self, batch, step, ep, i, out_dir="debug_svgs"):
        """Save GT and predicted SVGs for visualization."""
        if i != 0 or not self.accelerator.is_main_process:
            return

        logger.info(f"[viz] Attempting visualization for epoch {ep}")

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # ---------------- GT SVG ----------------
        output = step.extras.get("output", None)
        if output is None:
            logger.warning("[viz] No output in step.extras")
            return

        tgt_commands = output.get("tgt_commands")  # [B, G, S]
        tgt_args = output.get("tgt_args")  # [B, G, S, n_args]

        if tgt_commands is None or tgt_args is None:
            logger.warning("[viz] Missing tgt_commands or tgt_args")
            return

        b = 0  # first sample in batch

        # Skip SOS at position 0
        gt_cmd = tgt_commands[b, :, 1:]  # [G, S-1]
        gt_args = tgt_args[b, :, 1:]  # [G, S-1, n_args]

        try:
            gt_svg = decode_svg_from_cmd_args(gt_cmd, gt_args)
            gt_path = Path(out_dir) / f"ep{ep:03d}_gt.svg"
            gt_svg.save_svg(str(gt_path))
        except Exception as e:
            logger.warning(f"[viz] Failed to save GT SVG: {e}")

        # ---------------- Predicted SVG (via greedy_sample) ----------------
        # Get the unwrapped model for greedy_sample
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        device = next(unwrapped_model.parameters()).device

        # Explicitly move tensors to model device for multi-GPU safety
        enc_commands = batch["commands"][b : b + 1].to(device, non_blocking=True)  # [1, G, S]
        enc_args = batch["args"][b : b + 1].to(device, non_blocking=True)  # [1, G, S, n_args]

        with torch.no_grad():
            # Use the SVGTransformer's own sampling logic
            commands_y, args_y = unwrapped_model.model.greedy_sample(
                commands_enc=enc_commands,
                args_enc=enc_args,
                commands_dec=None,
                args_dec=None,
                label=None,
                z=None,
                hierarch_logits=None,
                concat_groups=True,  # flatten groups into one sequence
                temperature=0.0001,
            )

        # commands_y: [N, S'], args_y: [N, S', n_args], with N=1 here
        cmd_names = SVGTensor.COMMANDS_SIMPLIFIED
        first_seq = commands_y[0].cpu().tolist()
        decoded_cmds = [
            cmd_names[int(c)] if 0 <= int(c) < len(cmd_names) else f"?{c}" for c in first_seq[:10]
        ]
        logger.info(f"[viz] Pred cmds (first 10 via greedy_sample): {decoded_cmds}")

        try:
            pred_svg = decode_svg_from_cmd_args(
                commands_y[0].cpu(),
                args_y[0].cpu().float(),
            )
            pred_path = Path(out_dir) / f"ep{ep:03d}_pred.svg"
            pred_svg.save_svg(str(pred_path))
            logger.info(f"[viz] Saved SVGs to {out_dir}/ep{ep:03d}_*.svg")
        except Exception as e:
            logger.warning(f"[viz] Failed to save pred SVG: {e}")

    def _check_gradients_once(self):
        """Check gradient flow through the model (only on first step)"""
        if not self.check_gradients or self.first_step_done:
            return

        if not self.accelerator.is_main_process:
            self.first_step_done = True
            return

        # Get unwrapped model for gradient checking
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

    def run(
        self,
        train_loader,
        val_loader=None,
        max_epochs=10,
        log_every=50,
        save_every=1,
        start_epoch=0,
    ):
        from vecssl.util import make_progress

        # Prepare model, optimizer, scheduler, and dataloaders
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

        # Initialize trackers (wandb/tensorboard)
        if self.accelerator.is_main_process:
            tracker_config = self.cfg if isinstance(self.cfg, dict) else {}
            init_kwargs = {}
            if self.wandb_project:
                init_kwargs["wandb"] = {"name": self.wandb_project}

            self.accelerator.init_trackers(
                project_name=self.wandb_project or "training",
                config=tracker_config,
                init_kwargs=init_kwargs if init_kwargs else None,
            )

        # Ensure all processes ready before training loop
        self.accelerator.wait_for_everyone()

        progress = make_progress()
        with progress:
            epoch_task = progress.add_task("epoch", total=max_epochs - start_epoch)
            for ep in range(start_epoch, max_epochs):
                self.model.train()
                batch_task = progress.add_task("train", total=len(train_loader))
                run_loss = 0.0

                for i, batch in enumerate(train_loader):
                    # Forward pass (autocast handled by accelerator)
                    with self.accelerator.autocast():
                        step = self.model(batch)
                        loss = step.loss

                    # Visualize once per epoch (first batch)
                    self._maybe_visualize_batch(batch, step, ep, i)

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Check gradients on first step (after backward)
                    self._check_gradients_once()

                    # Gradient clipping
                    if self.grad_clip:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Scheduler step
                    if self.scheduler:
                        self.scheduler.step()

                    run_loss += float(loss.detach())
                    global_step = ep * len(train_loader) + i

                    # Logging (main process only)
                    if self.accelerator.is_main_process and i % log_every == 0:
                        log_dict = {
                            "train/loss": run_loss / (i + 1),
                            "epoch": ep,
                        }
                        if step.logs:
                            for k, v in step.logs.items():
                                log_dict[f"train/{k}"] = v
                        self.accelerator.log(log_dict, step=global_step)

                    progress.advance(batch_task)

                progress.advance(epoch_task)

                if self.accelerator.is_main_process:
                    logger.info(f"epoch {ep + 1}/{max_epochs} done")

                # Checkpointing
                if self.checkpoint_dir and ep % save_every == 0 and ep > 1:
                    self._save_checkpoint(ep)

                # Validation
                if val_loader and ep % 5 == 0 and ep > 1:
                    self.validate(val_loader, ep)

        # End training (cleanup trackers)
        self.accelerator.wait_for_everyone()
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
        train_ratio=1,
        already_preprocessed=True,
    )

    # Validation dataset (20% of data)
    val_dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        train_ratio=1,
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
        drop_last=False,  # Drop incomplete batches
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
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
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
        )
        start_epoch = metadata["epoch"] + 1
        logger.info(f"Resuming training from epoch {start_epoch}", extra={"markup": True})

    # Prepare wandb config (combine model config + training args)
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
        "mixed_precision": args.mixed_precision,
        "n_params": n_params,
    }
    if args.wandb_project:
        logger.info(
            f"Wandb enabled - project: [bold]{args.wandb_project}[/bold]", extra={"markup": True}
        )

    # Create trainer
    trainer = DebugTrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        tb_dir=args.tb_dir,
        wandb_project=args.wandb_project,
        cfg=wandb_config,
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
