import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vit_b_16
from .base import JointModel, TrainStep
from model import SVGEmbedding
from .config import ContrastiveConfig


class ContrastiveModel(JointModel):
    def __init__(self, cfg: ContrastiveConfig):
        super().__init__()
        self.cfg = cfg

        # Initialize ViT model for image encoding
        self.image_encoder = vit_b_16(weights=None)  # No pre-trained weights
        self.image_encoder.heads = nn.Identity()  # Remove classification head
        img_dim = self.image_encoder.hidden_dim  # Output dimension of ViT

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.contrastive_logit_scale))

        # Initialize SVG embedding model
        self.svg_embedding_model = SVGEmbedding(
            seq_len=cfg.max_seq_len,
            cfg=cfg,
            rel_args=False,
            use_group=cfg.use_group,
            group_len=None,
        )

        # Projection heads
        joint_dim = 512  # Joint embedding dimension
        self.svg_projection = nn.Sequential(
            nn.Linear(cfg.dim_z, joint_dim),
            nn.BatchNorm1d(joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, joint_dim),
        )
        self.img_projection = nn.Sequential(
            nn.Linear(img_dim, joint_dim),
            nn.BatchNorm1d(joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, joint_dim),
        )

    def encode_joint(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Encode SVGs and images into a joint embedding space.
        """
        # Encode SVG
        svg_commands = batch["commands"]
        svg_args = batch["args"]
        z_svg = self.svg_embedding_model(svg_commands, svg_args)  # [batch, dim_z]
        z_svg = self.svg_projection(z_svg)  # Project to joint space

        # Encode image
        img = batch["image"]
        z_img = self.image_encoder(img)  # [batch, img_dim]
        z_img = self.img_projection(z_img)  # Project to joint space

        return {"svg": F.normalize(z_svg, dim=-1), "img": F.normalize(z_img, dim=-1)}

    def forward(self, batch: dict[str, any]) -> TrainStep:
        """
        Perform a forward pass and return contrastive loss.
        """
        # Encode SVGs and images into joint space
        joint_embeddings = self.encode_joint(batch)
        z_svg = joint_embeddings["svg"]  # [batch, joint_dim]
        z_img = joint_embeddings["img"]  # [batch, joint_dim]

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * z_img @ z_svg.T
        logits_per_text = logits_per_image.T

        # define labels
        batch_size = z_img.size(0)
        labels = torch.arange(batch_size, device=z_img.device)

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
