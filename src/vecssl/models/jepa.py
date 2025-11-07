import torch
import torch.nn as nn
import torch.nn.functional as F
from vecssl.models.config import _DefaultConfig, JepaConfig
from vecssl.models.base import JointModel, TrainStep
from transformers import AutoImageProcessor, AutoModel
from vecssl.models.model import Encoder
from vecssl.util import _make_seq_first, _make_batch_first

# TODO: update this architecture
class Predictor(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        z_pred = self.layers(x)
        return z_pred

class ImageEncoder(nn.Module):
    def __init__(self, emb_sz):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", do_rescale=False)
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.projector = nn.Linear(self.backbone.config.hidden_size, emb_sz)

        # For stop gradient
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x, seq_len):
        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(next(self.backbone.parameters()).device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        z_img = self.projector(cls_token_output)
        z_img = z_img.unsqueeze(1).repeat(1, seq_len, 1)  # Shape: [batch_size, seq_len, emb_sz]
        return z_img


class Jepa(JointModel):
    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg

        self.svg_encoder = Encoder(cfg)
        self.svg_projector = nn.Linear(self.svg_encoder.cfg.d_model, cfg.d_joint)
        self.predictor = Predictor(self.svg_encoder.cfg.d_model, cfg.d_joint) # TODO: what would be the action here?

        self.image_encoder = ImageEncoder(cfg.d_joint)

        
    def forward(self, batch):
        device = next(self.parameters()).device

        commands = batch['commands'].to(device)
        args = batch['args'].to(device)
        images = batch['image'].to(device)

        # TODO: need masking

        commands_enc, args_enc = _make_seq_first(commands, args)
        z_svg = self.svg_encoder(commands_enc, args_enc)
        z_svg = _make_batch_first(z_svg).squeeze()
        z_svg = self.predictor(z_svg)
        
        seq_len = z_svg.shape[1]
        z_img = self.image_encoder(images, seq_len)

        z_svg = F.normalize(z_svg, dim=-1)
        z_img = F.normalize(z_img, dim=-1)

        loss = self.loss(z_svg, z_img)

        logs = {"mse_loss": loss.item()}
        return TrainStep(loss=loss, logs=logs)

    def loss(self, z_svg, z_img):
        return F.mse_loss(z_svg, z_img)

    def encode_joint(self, batch):
        device = next(self.parameters()).device

        commands = batch['commands'].to(device)
        args = batch['args'].to(device)
        images = batch['image'].to(device)

        commands_enc, args_enc = _make_seq_first(commands, args)
        z_svg = self.svg_encoder(commands_enc, args_enc)
        z_svg = _make_batch_first(z_svg).squeeze()
        z_svg = self.predictor(z_svg)
        
        seq_len = z_svg.shape[1]
        z_img = self.image_encoder(images, seq_len)

        z_svg = F.normalize(z_svg, dim=-1)
        z_img = F.normalize(z_img, dim=-1)
        return {"svg": z_svg, "img": z_img}