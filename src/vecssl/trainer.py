"""Base trainer for all SSL experiments"""

import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from vecssl.util import make_progress
import wandb
from pathlib import Path

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dir=None,
        device=None,
        scheduler=None,
        amp=True,
        grad_clip=None,
        tb_dir=None,
        wandb_project=None,
        cfg=None,
    ) -> None:
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = amp
        self.grad_clip = grad_clip
        self.tb = SummaryWriter(tb_dir) if tb_dir else None
        self.wandb_run = wandb.init(project=wandb_project, config=cfg) if wandb_project else None
        self.cfg = cfg
        self.best_val_loss = float("inf")

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
                        for k, v in step.logs.items():
                            self.tb.add_scalar(f"train/{k}", v, global_step)
                    if self.wandb_run and i % log_every == 0:
                        wandb_logs = {
                            "train/loss": run / (i + 1),
                            "epoch": ep,
                            "step": global_step,
                        }
                        for k, v in step.logs.items():
                            wandb_logs[f"train/{k}"] = v
                        self.wandb_run.log(wandb_logs)
                    progress.advance(batch_task)
                progress.advance(epoch_task)
                logger.info(f"epoch {ep + 1}/{max_epochs} done")
                if self.checkpoint_dir and ep % save_every == 0 and ep > 1:
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
            wandb.finish()

    @torch.no_grad()
    def validate(self, val_loader, ep):
        self.model.eval()
        losses = []
        for batch in val_loader:
            step = self.model(batch)
            losses.append(float(step.loss))
        val_loss = sum(losses) / max(1, len(losses))
        if self.tb:
            self.tb.add_scalar("val/loss", val_loss, ep)
        if self.wandb_run:
            self.wandb_run.log({"val/loss": val_loss, "epoch": ep})

        # Save best model checkpoint
        if self.checkpoint_dir and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            logger.info(f"New best validation loss: {val_loss:.6f} - saving best model")
            _state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "cfg": self.cfg,
                "epoch": ep,
                "val_loss": val_loss,
            }
            if not self.checkpoint_dir.exists():
                self.checkpoint_dir.mkdir(parents=True)
            torch.save(_state, self.checkpoint_dir / "best_model.pt")


def save_chkpt(model, optimizer, scheduler, cfg, checkpoint_dir: Path, ep: int):
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
        device: Device to map tensors to

    Returns:
        Dictionary with checkpoint metadata (epoch, cfg, etc.)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

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
