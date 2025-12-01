"""Base trainer for all SSL experiments using Accelerate"""

import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from vecssl.util import make_progress

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dir=None,
        scheduler=None,
        grad_clip=None,
        mixed_precision="no",  # "no", "fp16", or "bf16"
        tb_dir=None,
        wandb_project=None,
        cfg=None,
    ) -> None:
        # Store unprepared objects - will be prepared in run()
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        self.cfg = cfg
        self.tb_dir = tb_dir
        self.wandb_project = wandb_project
        self.best_val_loss = float("inf")

        # Setup logging backends for Accelerator
        log_with = []
        if tb_dir:
            log_with.append("tensorboard")
        if wandb_project:
            log_with.append("wandb")

        # Project configuration for checkpointing
        project_config = None
        if checkpoint_dir:
            project_config = ProjectConfiguration(
                project_dir=str(checkpoint_dir),
                logging_dir=tb_dir,
            )

        # Create Accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            log_with=log_with if log_with else None,
            project_config=project_config,
        )

        # These will be set in run() after prepare()
        self.model = None
        self.optimizer = None
        self.scheduler = None

    @property
    def device(self):
        """For backward compatibility - returns the accelerator's device"""
        return self.accelerator.device

    def run(
        self,
        train_loader,
        val_loader=None,
        max_epochs=10,
        log_every=50,
        save_every=1,
        start_epoch=0,
    ):
        # Prepare model, optimizer, train_loader (val_loader stays unprepared for rank-0 only eval)
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

                    # Backward pass
                    self.accelerator.backward(loss)

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

                # Validation (only rank 0 runs, others wait)
                if val_loader:
                    if self.accelerator.is_main_process:
                        self.validate(val_loader, ep)
                    self.accelerator.wait_for_everyone()

        # End training (cleanup trackers)
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    @torch.no_grad()
    def validate(self, val_loader, ep):
        """Run validation only on the current process (call from rank 0 only).

        No collectives are used - this avoids NCCL desync issues in multi-GPU training.
        Evaluates on rank 0's shard of the validation set only.
        """
        self.model.eval()
        losses = []

        for batch in val_loader:
            with self.accelerator.autocast():
                step = self.model(batch)
            loss = step.loss.detach()
            if loss.ndim > 0:
                loss = loss.mean()
            losses.append(loss.item())

        val_loss = sum(losses) / len(losses) if losses else float("nan")

        # Logging (this should only be called from main process anyway)
        if self.accelerator.is_main_process:
            self.accelerator.log({"val/loss": val_loss, "epoch": ep})

            # Save best model checkpoint
            if self.checkpoint_dir and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                logger.info(f"New best validation loss: {val_loss:.6f} - saving best model")
                self._save_checkpoint(ep, is_best=True, val_loss=val_loss)

    def _save_checkpoint(self, ep, is_best=False, val_loss=None):
        """Save checkpoint on the main process only (no collectives)."""
        if not self.accelerator.is_main_process:
            return  # Only rank 0 saves - no barriers here to avoid NCCL desync

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        _state = {
            "model": unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "cfg": self.cfg,
            "epoch": ep,
        }
        if val_loss is not None:
            _state["val_loss"] = val_loss

        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)

        if is_best:
            torch.save(_state, self.checkpoint_dir / "best_model.pt")
        else:
            torch.save(_state, self.checkpoint_dir / f"checkpoint_{ep:04d}.pt")


def save_chkpt(model, optimizer, scheduler, cfg, checkpoint_dir: Path, ep: int, accelerator=None):
    """Standalone checkpoint save function for backward compatibility"""
    if accelerator is not None:
        # Use accelerator's unwrap if available
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        model = accelerator.unwrap_model(model)

    _state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "cfg": cfg,
        "epoch": ep,
    }
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    torch.save(_state, checkpoint_dir / f"checkpoint_{ep:04d}.pt")


def load_chkpt(checkpoint_path, model, optimizer=None, scheduler=None, device=None):
    """Load checkpoint and restore model/optimizer/scheduler state

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map tensors to (optional, for backward compat)

    Returns:
        Dictionary with checkpoint metadata (epoch, cfg, etc.)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "cfg": checkpoint.get("cfg"),
    }

    logger.info(f"Checkpoint loaded from epoch {metadata['epoch']}")
    return metadata
