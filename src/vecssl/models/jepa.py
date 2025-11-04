import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from typing import Any
from vecssl.models.base import JointModel, TrainStep

d_joint = 1024 
N = 256 
D_row = 28 

class SVGEncoder(nn.Module):
    def __init__(
        self, input_dim, embed_dim, output_dim,
        num_layers, num_heads, mlp_ratio,
        dropout, max_seq_len
    ):
        super().__init__()

        # Patch embedding
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final normalization and projection
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x):
        B, N, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :N, :]
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("facebook/dino-v2-base")
        self.backbone = AutoModel.from_pretrained("facebook/dino-v2-base")
        self.proj = nn.Linear(self.backbone.config.hidden_size, output_dim)

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(next(self.backbone.parameters()).device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        z_img = self.proj(cls_token_output)
        return z_img
    
class Predictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.norm(x)
        z_pred = self.output_proj(x)
        return z_pred

class SVGDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(SVGDecoder, self).__init__()

    def forward(self, z):
        raise NotImplementedError

class SVGImageJepa(JointModel):
    def __init__(self):
        super().__init__()
        self.svg_encoder = SVGEncoder(input_dim=D_row, embed_dim=512, output_dim=d_joint, num_layers=8, num_heads=8, mlp_ratio=4, dropout=0.1, max_seq_len=256)
        self.predictor = Predictor(embed_dim=d_joint)
        self.img_encoder = ImageEncoder(output_dim=d_joint)

    def forward(self, batch: dict[str, torch.Tensor]) -> TrainStep:
        svg_batch = batch["svg"]
        img_batch = batch["img"]

        # TODO: Mask input before sending to encoders - doesn't have to be the same masking
        z_svg = self.svg_encoder(svg_batch)
        z_svg = self.predictor(z_svg)

        with torch.no_grad():
            z_img = self.img_encoder(img_batch)

        loss = self.loss(z_svg, z_img.detach())

        logs = {"mse_loss": loss.item()}

        return TrainStep(loss=loss, logs=logs)
    
    def loss(self, z_svg, z_img):
        z_svg = F.normalize(z_svg, dim=-1)
        z_img = F.normalize(z_img, dim=-1)
        return F.mse_loss(z_svg, z_img)
    
    @torch.no_grad
    def encode_joint(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        svg_batch = batch["svg"]
        img_batch = batch["img"]

        x = self.svg_encoder(svg_batch)
        z_svg = self.predictor(x)

        z_img = self.img_encoder(img_batch)

        return {"svg": z_svg, "img": z_img}
    
    

