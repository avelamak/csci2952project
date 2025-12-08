"""
Minimal SVGMAE Test Script

Tests that the SVG-only MAE model forward + loss pipeline runs (smoke test).
"""

import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from vecssl.data.svg import SVG
from vecssl.data.svg_tensor import SVGTensor
from vecssl.data.geom import Bbox


from vecssl.data.dataset import SVGXDataset
from vecssl.models.config import SVGMAEConfig
from vecssl.models.svgmae import SVGMAE
from vecssl.trainer import Trainer
from vecssl.util import setup_logging, set_seed


from eval_utils import custom_collate

import cairosvg


def svg_to_png(svg_path: str, png_path: str):
    cairosvg.svg2png(url=svg_path, write_to=png_path, background_color="#ffffff")


logger = logging.getLogger(__name__)


def decode_svg_from_cmd_args(
    commands: torch.Tensor,
    args: torch.Tensor,
    viewbox_size: int = 256,  # Match ARGS_DIM quantization range
    pad_val: int = -1,
    logger=None,
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

    # Ensure commands are flat ints
    commands = commands.long()

    # Find EOS position safely
    eos_positions = (commands == eos_idx).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        seq_len = int(eos_positions[0].item())
    else:
        # No EOS found trim padding only
        valid_positions = (commands != pad_val).nonzero(as_tuple=True)[0]
        seq_len = int(valid_positions[-1].item()) + 1 if len(valid_positions) > 0 else 0

    if seq_len == 0:
        # Return empty SVG
        return SVG([], viewbox=Bbox(viewbox_size))

    # Truncate to valid length
    max_len = min(len(commands), len(args))
    commands = commands[:max_len]
    args = args[:max_len]

    cmd_vocab = SVGTensor.COMMANDS_SIMPLIFIED
    valid_svg_cmds = {"m", "l", "c", "z", "h", "v", "a"}  # adapt if needed

    filtered_cmds = []
    filtered_args = []

    for i, c_idx in enumerate(commands.tolist()):
        cmd = cmd_vocab[c_idx]

        # Stop on EOS
        if cmd == "EOS":
            break

        # Skip SOS / PAD / MASK / etc
        if cmd not in valid_svg_cmds:
            continue

        filtered_cmds.append(c_idx)
        filtered_args.append(args[i])

    if len(filtered_cmds) == 0:
        return SVG([], viewbox=Bbox(viewbox_size))

    if logger:
        logger.info(
            f"[viz] First 30 SVG cmds before filtering: {[cmd_vocab[c] for c in filtered_cmds[:30]]}"
        )

    if len(filtered_cmds) > 0:
        first_cmd = cmd_vocab[filtered_cmds[0]]
        if first_cmd not in ("m", "M"):
            # Insert moveto(0,0)
            m_idx = cmd_vocab.index("m")
            filtered_cmds.insert(0, m_idx)

            # Create dummy args
            dummy = torch.zeros_like(filtered_args[0])
            filtered_args.insert(0, dummy)

    if logger:
        logger.info(
            f"[viz] First 30 SVG cmds after filtering: {[cmd_vocab[c] for c in filtered_cmds[:30]]}"
        )

    commands = torch.tensor(filtered_cmds, dtype=torch.long)
    args = torch.stack(filtered_args)
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


def visualize_batch(model, batch, epoch, out_dir="debug_svgs"):
    """
    Save GT and predicted SVGs for visualization.

    Args:
        model: The SVG autoencoder model (unwrapped)
        batch: A batch dict with 'commands' and 'args' tensors
        epoch: Current epoch number (for filename)
        out_dir: Output directory for SVG files
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # device = next(model.parameters()).device

    b = 0  # first sample in batch

    # Get GT from batch (skip SOS at position 0)
    gt_commands = batch["commands"][b, :, 1:]  # [G, S-1]
    gt_args = batch["args"][b, :, 1:]  # [G, S-1, n_args]

    try:
        gt_svg = decode_svg_from_cmd_args(gt_commands, gt_args)
        gt_path = Path(out_dir) / f"ep{epoch:03d}_gt.svg"
        gt_png_path = Path(out_dir) / f"ep{epoch:03d}_gt.png"
        gt_svg.save_svg(str(gt_path))
        svg_to_png(str(gt_path), str(gt_png_path))
        logger.info(f"[viz] Saved GT to {gt_path}")
    except Exception as e:
        logger.warning(f"[viz] Failed to save GT SVG: {e}")

    # Generate prediction via greedy sampling
    # enc_commands = batch["commands"][b : b + 1].to(device)
    # enc_args = batch["args"][b : b + 1].to(device)

    with torch.no_grad():
        commands_y, args_y = model.model.greedy_reconstruct(batch)

    # Log first few predicted commands
    cmd_names = SVGTensor.COMMANDS_SIMPLIFIED
    # first_seq = commands_y[0].view(-1).cpu().tolist()

    # Flatten outputs
    cmds = commands_y[0].reshape(-1).cpu().tolist()
    args = args_y[0].reshape(-1, args_y.shape[-1]).cpu().float()

    # Filter out non-draw commands
    valid_cmds = []
    valid_args = []
    for i, c in enumerate(cmds):
        name = cmd_names[int(c)]
        if name in ["SOS", "EOS", "PAD"]:
            continue
        valid_cmds.append(c)
        valid_args.append(args[i])

    valid_cmds = torch.tensor(valid_cmds, dtype=torch.long)
    valid_args = (
        torch.stack(valid_args) if len(valid_args) > 0 else torch.empty((0, args.shape[-1]))
    )

    logger.info(f"[viz] filtered cmds count: {len(valid_cmds)}")
    logger.info(f"[viz] filtered args count: {len(valid_args)}")
    decoded_cmds = [
        cmd_names[int(c)] if 0 <= int(c) < len(cmd_names) else f"?{c}" for c in valid_cmds[:10]
    ]
    logger.info(f"[viz] Pred cmds (first 10): {decoded_cmds}")

    try:
        pred_svg = decode_svg_from_cmd_args(valid_cmds, valid_args, logger=logger)
        pred_path = Path(out_dir) / f"ep{epoch:03d}_pred.svg"
        pred_path_png = Path(out_dir) / f"ep{epoch:03d}_pred.png"
        pred_svg.save_svg(str(pred_path))
        svg_to_png(str(pred_path), str(pred_path_png))
        logger.info(f"[viz] Saved pred to {pred_path}")
    except Exception as e:
        logger.warning(f"[viz] Failed to save pred SVG: {e}")


class SimpleSVGMAE(nn.Module):
    """
    Minimal wrapper for testing SVGMAE.
    """

    def __init__(self, cfg: SVGMAEConfig, debug_mode: bool = False):
        super().__init__()
        self.cfg = cfg
        self.debug_mode = debug_mode
        self.step_count = 0

        self.model = SVGMAE(cfg)

        logger.info("Created SimpleSVGMAE:")
        logger.info(f"  - max_num_groups: {cfg.max_num_groups}")
        logger.info(f"  - max_seq_len: {cfg.max_seq_len}")
        logger.info(f"  - mask_ratio_svg: {cfg.mask_ratio_svg}")
        logger.info(f"  - mae_depth: {cfg.mae_depth}")
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
                f"SVGMAE forward returned TrainStep: loss={step.loss if hasattr(step, 'loss') else None}"
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

    @torch.no_grad()
    def reconstruct_svgs(self, train_loader, epochs, batch=None, out_dir="svg_recon"):
        if not self.accelerator.is_main_process:
            return None

        # Use a batch provided or grab first batch from train_loader
        if batch is None:
            batch = next(iter(train_loader))

        # Ensure model is in eval mode
        self.model.eval()
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        epoch_dir = Path(out_dir) / f"epoch_{epochs:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Call visualize_batch
        visualize_batch(unwrapped_model, batch, epoch=epochs, out_dir=str(epoch_dir))

        logger.info(f"[bold green]Saved reconstruction SVGs to {out_dir}[/bold green]")

        return 0

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

        if self.accelerator.is_main_process:
            tracker_config = self.cfg if isinstance(self.cfg, dict) else {}
            init_kwargs = {}
            if self.wandb_project:
                wandb_init = {}
                if self.wandb_name:
                    wandb_init["name"] = self.wandb_name
                if self.wandb_entity:
                    wandb_init["entity"] = self.wandb_entity
                if wandb_init:
                    init_kwargs["wandb"] = wandb_init

            self.accelerator.init_trackers(
                project_name=self.wandb_project or "debug-training",
                config=tracker_config,
                init_kwargs=init_kwargs if init_kwargs else None,
            )

        self.accelerator.wait_for_everyone()

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

    train_shuffle = True
    train_drop_last = True

    if args.overfit_samples is not None:
        n = args.overfit_samples
        logger.info(f" Limiting datasets to first {n} glyphs")

        train_dataset = Subset(train_dataset, list(range(min(n, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(n, len(val_dataset)))))
        train_shuffle = False
        train_drop_last = False

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        drop_last=train_drop_last,  # Drop incomplete batches
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
    parser = argparse.ArgumentParser(description="Test SVG-only MAE model")

    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="svgx_svgs", help="SVG directory")
    parser.add_argument("--img-dir", type=str, default="svgx_imgs", help="Image directory")
    parser.add_argument("--meta", type=str, default="svgx_meta.csv", help="Metadata CSV")

    # Model args
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max number of paths")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument("--mask-ratio-svg", type=float, default=0.75, help="SVG masking ratio")
    parser.add_argument("--mae-depth", type=int, default=4, help="MAE encoder depth")

    parser.add_argument(
        "--overfit_samples",
        type=int,
        default=None,  # None = no limit
        help="Limit number of glyph samples (use for overfitting/debug)",
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
    parser.add_argument("--tb-dir", type=str, default="runs/test_svgmae", help="TensorBoard dir")

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
    logger.info("[bold cyan]SVGMAE Test[/bold cyan]", extra={"markup": True})
    logger.info("=" * 60)

    cfg = SVGMAEConfig()
    cfg.max_num_groups = args.max_num_groups
    cfg.max_seq_len = args.max_seq_len
    cfg.mask_ratio_svg = args.mask_ratio_svg
    cfg.mae_depth = args.mae_depth

    wandb_config = {
        # Model config
        "model_name": "svgmae",
        "max_num_groups": cfg.max_num_groups,
        "max_seq_len": cfg.max_seq_len,
        "mask_ratio_svg": cfg.mask_ratio_svg,
        "mae_depth": cfg.mae_depth,
        "mae_num_heads": cfg.mae_num_heads,
        "mae_mlp_ratio": cfg.mae_mlp_ratio,
        "mae_dropout": cfg.mae_dropout,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "dim_feedforward": cfg.dim_feedforward,
        "dropout": cfg.dropout,
        "loss_cmd_weight": cfg.loss_cmd_weight,
        "loss_args_weight": cfg.loss_args_weight,
        # Training args
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "num_workers": args.num_workers,
        "log_every": args.log_every,
        "mixed_precision": args.mixed_precision,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": args.seed,
    }

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Count parameters
    logger.info("Creating model...")
    model = SimpleSVGMAE(cfg, debug_mode=args.debug)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: [bold]{n_params:,}[/bold]", extra={"markup": True})

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
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
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
        )
        logger.info(
            "[bold green]✓ Training completed successfully![/bold green]", extra={"markup": True}
        )
    except Exception as e:
        logger.error(f"[bold red]✗ Training failed: {e}[/bold red]", extra={"markup": True})
        import traceback

        traceback.print_exc()
        sys.exit(1)

    logger.info("Checking recon quality")
    try:
        trainer.reconstruct_svgs(train_loader=train_loader, epochs=args.epochs)
        logger.info(
            "[bold green]✓ Recon completed successfully SVGs saved to: ![/bold green]",
            extra={"markup": True},
        )
    except Exception as e:
        logger.error(f"[bold red]✗ Recon failed: {e}[/bold red]", extra={"markup": True})
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
