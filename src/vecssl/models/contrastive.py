import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from vecssl.util import _make_seq_first
from vecssl.models.base import JointModel, TrainStep
from vecssl.models.model import Encoder, DINOImageEncoder
from vecssl.models.config import ContrastiveConfig
from vecssl.models.basic_blocks import ResNet


class ContrastiveModel(JointModel):
    def __init__(self, cfg: ContrastiveConfig):
        super().__init__()
        self.cfg = cfg

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.contrastive_logit_scale))

        # Initialize SVG embedding model
        self.encoder = Encoder(cfg)
        if cfg.use_resnet:
            self.resnet = ResNet(cfg.d_model)

        # Initialize DINO embedding model
        self.image_encoder = DINOImageEncoder(layer=self.cfg.DINO_layer)
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.svg_proj = nn.Linear(self.cfg.d_joint, self.cfg.d_model)
        self.img_proj = nn.Linear(self.image_encoder.backbone.config.hidden_size, self.cfg.d_model)

    def encode_joint(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Encode SVGs and images into a joint embedding space.
        """
        # Encode SVG
        device = next(self.parameters()).device

        svg_commands = batch["commands"].to(device)
        svg_args = batch["args"].to(device)

        commands_enc, args_enc = _make_seq_first(svg_commands, svg_args)
        z_svg = self.encoder(commands_enc, args_enc)  # [batch, dim_z]

        # remove extra 1 dim
        z_svg = z_svg.squeeze()

        if self.cfg.use_resnet:
            z_svg = self.resnet(z_svg)

        # Encode image
        img = batch["image"].to(device)
        with torch.no_grad():
            z_img = self.image_encoder(img)  # [batch, img_dim]

        # Project to joint space
        z_svg = self.svg_proj(z_svg)
        z_img = self.img_proj(z_img)

        # Normalize
        z_svg = F.normalize(z_svg, dim=-1)
        z_img = F.normalize(z_img, dim=-1)

        return {"svg": z_svg, "img": z_img}

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
