import torch
import torch.nn as nn
import numpy as np
from base import JointModel, TrainStep


class ContrastiveModel(JointModel):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, batch: dict[str, any]) -> TrainStep:
        """
        Perform a forward pass and return contrastive loss.
        """
        img_embedding = batch["img"]
        svg_embedding = batch["svg"]

        # normalized features
        img_embedding = img_embedding / img_embedding.norm(dim=1, keepdim=True)
        svg_embedding = svg_embedding / svg_embedding.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_embedding @ svg_embedding.t()
        logits_per_text = logits_per_image.t()

        # define labels
        batch_size = img_embedding.shape[0]
        labels = torch.arange(batch_size, device=img_embedding.device)

        # compute cross-entropy loss for both directions
        loss_image = nn.CrossEntropyLoss()(logits_per_image, labels)  # image -> svg
        loss_svg = nn.CrossEntropyLoss()(logits_per_text, labels)  # svg -> image

        # average the two losses
        loss = (loss_image + loss_svg) / 2

        return TrainStep(
            loss=loss,
            logs={
                "loss": loss.item(),
                "loss_image": loss_image.item(),
                "loss_svg": loss_svg.item(),
            },
        )
