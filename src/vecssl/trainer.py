"""Base trainer for all SSL experiments"""

import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from vecssl.util import make_progress
import wandb

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device=None,
        scheduler=None,
        amp=True,
        grad_clip=None,
        tb_dir=None,
        wandb_project=None,
        cfg=None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = amp
        self.grad_clip = grad_clip
        self.tb = SummaryWriter(tb_dir) if tb_dir else None
        self.wandb_run = wandb.init(project=wandb_project, config=cfg) if wandb_project else None

    def run(self, train_loader, val_loader=None, max_epochs=10, log_every=50):
        device = self.device
        self.model.to(device)
        scaler = torch.amp.grad_scaler.GradScaler(enabled=self.amp)

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
