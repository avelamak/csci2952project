import torch
import torch.nn as nn
import torch.nn.functional as F
from vecssl.models.config import JepaConfig
from vecssl.models.base import JointModel, TrainStep
from vecssl.models.model import Encoder, DINOImageEncoder
from vecssl.util import _make_seq_first, _make_batch_first
from vecssl.models.basic_blocks import ResNet


class PredictorTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, hidden_dim, dropout):
        super().__init__()

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):  # x is of shape [batch, seq_len, emb_sz]
        transformer_out = self.transformer_encoder(x)
        transformer_out = self.layer_norm(transformer_out)
        z_pred = transformer_out.mean(dim=1)  # Pooling over seq_len
        z_pred = self.output_layer(z_pred)
        return z_pred


class PredictorMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        layers = []
        in_dim = embed_dim

        # Build MLP with (num_layers - 1) hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, embed_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):  # x: [batch, embed_dim]
        z_pred = self.mlp(x)
        return z_pred


class Jepa(JointModel):
    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg

        self.svg_encoder = Encoder(cfg)
        if cfg.use_resnet:
            self.resnet = ResNet(cfg.d_model)
        self.svg_projector = nn.Linear(self.svg_encoder.cfg.d_model, cfg.d_joint)

        if cfg.predictor_type == "transformer":
            self.predictor = PredictorTransformer(
                embed_dim=cfg.d_joint,
                num_heads=cfg.predictor_transformer_num_heads,
                num_layers=cfg.predictor_transformer_num_layers,
                hidden_dim=cfg.predictor_transformer_hidden_dim,
                dropout=cfg.predictor_transformer_dropout,
            )
        else:
            self.predictor = PredictorMLP(
                embed_dim=cfg.d_joint,
                hidden_dim=cfg.predictor_mlp_hidden_dim,
                num_layers=cfg.predictor_mlp_num_layers,
                dropout=cfg.predictor_mlp_dropout,
            )

        with torch.no_grad():
            self.image_encoder = DINOImageEncoder()
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # self.image_projector = nn.Linear(
            #     self.image_encoder.backbone.config.hidden_size, cfg.d_joint
            # )
            # for param in self.image_projector.parameters():
            #     param.requires_grad = False

    def forward(self, batch):
        device = next(self.parameters()).device

        commands = batch["commands"].to(device)
        args = batch["args"].to(device)
        images = batch["image"].to(device)

        commands_enc, args_enc = _make_seq_first(commands, args)
        z_svg = self.svg_encoder(commands_enc, args_enc)
        if self.cfg.use_resnet:
            z_svg = self.resnet(z_svg)
        z_svg = self.svg_projector(z_svg)
        z_svg = _make_batch_first(z_svg)
        z_svg = z_svg.squeeze()
        z_svg = self.predictor(z_svg)
        z_svg = F.normalize(z_svg, dim=-1)

        with torch.no_grad():
            z_img = self.image_encoder(images)
            # z_img = self.image_projector(z_img)
            z_img = F.normalize(z_img, dim=-1)

        loss = self.loss(z_svg, z_img)

        logs = {"mse_loss": loss.item()}
        return TrainStep(loss=loss, logs=logs)

    def loss(self, z_svg, z_img):
        return F.mse_loss(z_svg, z_img)

    @torch.no_grad()
    def encode_joint(self, batch):
        device = next(self.parameters()).device

        commands = batch["commands"].to(device)
        args = batch["args"].to(device)
        images = batch["image"].to(device)

        commands_enc, args_enc = _make_seq_first(commands, args)
        z_svg = self.svg_encoder(commands_enc, args_enc)
        if self.cfg.use_resnet:
            z_svg = self.resnet(z_svg)
        z_svg = self.svg_projector(z_svg)
        z_svg = _make_batch_first(z_svg).squeeze()
        z_svg = z_svg.squeeze()
        z_svg = F.normalize(z_svg, dim=-1)

        z_img = self.image_encoder(images)
        # z_img = self.image_projector(z_img)
        z_img = F.normalize(z_img, dim=-1)

        return {"svg": z_svg, "img": z_img}
