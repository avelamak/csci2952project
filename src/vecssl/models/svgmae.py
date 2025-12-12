import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from vecssl.models.base import JointModel, TrainStep
from vecssl.models.config import SVGMAEConfig
from vecssl.models.model import SVGEmbedding, ConstEmbedding
from vecssl.models.basic_blocks import FCN
from vecssl.models.layers.transformer import TransformerEncoder, LayerNorm
from vecssl.models.layers.improved_transformer import TransformerEncoderLayerImproved
from vecssl.models.utils import _get_padding_mask, _get_key_padding_mask
from vecssl.util import _make_seq_first, _pack_group_batch, _unpack_group_batch
from vecssl.data.svg_tensor import SVGTensor


class SVGGroupEncoder(nn.Module):
    """Encodes SVG into per-group embeddings."""

    def __init__(self, cfg: SVGMAEConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = SVGEmbedding(cfg, cfg.max_seq_len, use_group=False)
        encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout
        )
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, commands: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
        N, G, S = commands.shape
        commands_sf, args_sf = _make_seq_first(commands, args)
        commands_packed, args_packed = _pack_group_batch(commands_sf, args_sf)

        padding_mask = _get_padding_mask(commands_packed, seq_dim=0)
        key_padding_mask = _get_key_padding_mask(commands_packed, seq_dim=0)

        src = self.embedding(commands_packed, args_packed)
        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask, memory2=None)

        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(
            dim=0, keepdim=True
        ).clamp(min=1.0)

        z = _unpack_group_batch(N, z)
        z = z.squeeze(0).permute(1, 0, 2)
        return z


class SVGGroupDecoder(nn.Module):
    """Decodes group latents back to tokens using one-shot logic."""

    def __init__(self, cfg: SVGMAEConfig):
        super().__init__()
        self.cfg = cfg
        self.query_embedding = ConstEmbedding(cfg, cfg.max_seq_len + 2)  # +2 for SOS/EOS
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.n_layers_decode)
        args_dim = cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def forward(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: (N, G_masked, d_model)
        # memory: (L_enc, N, d_model) <-- EXPECTS SEQ-FIRST

        N, G_masked, d_model = z.shape
        L_enc = memory.shape[0]  # Correctly reading L from dim 0

        z_flat = z.reshape(N * G_masked, d_model)
        tgt = self.query_embedding(z_flat.unsqueeze(0))

        # Repeat memory for each masked group
        # (L, N, D) -> (L, N, G, D) -> (L, N*G, D)
        memory_expanded = memory.unsqueeze(2).expand(-1, -1, G_masked, -1)
        memory_expanded = memory_expanded.reshape(L_enc, N * G_masked, d_model)

        if memory_key_padding_mask is not None:
            # (N, L) -> (N, G, L) -> (N*G, L)
            mem_mask = memory_key_padding_mask.unsqueeze(1).expand(-1, G_masked, -1)
            mem_mask = mem_mask.reshape(N * G_masked, L_enc)
        else:
            mem_mask = None

        out = self.decoder(
            tgt,
            memory_expanded,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=mem_mask,
        )

        command_logits, args_logits = self.fcn(out)
        S = command_logits.size(0)
        command_logits = command_logits.view(S, N, G_masked, -1).permute(1, 2, 0, 3)
        args_logits = args_logits.view(S, N, G_masked, self.cfg.n_args, -1).permute(1, 2, 0, 3, 4)

        return command_logits, args_logits


class SVGMAE(JointModel):
    def __init__(self, cfg: SVGMAEConfig):
        super().__init__()
        self.cfg = cfg
        self.svg_group_encoder = SVGGroupEncoder(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        mae_encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.mae_num_heads, int(cfg.mae_mlp_ratio * cfg.d_model), cfg.mae_dropout
        )
        mae_encoder_norm = LayerNorm(cfg.d_model)
        self.mae_encoder = TransformerEncoder(mae_encoder_layer, cfg.mae_depth, mae_encoder_norm)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.group_pos_embed = nn.Parameter(
            torch.zeros(1, cfg.max_num_groups, cfg.d_model), requires_grad=False
        )
        pos = torch.arange(cfg.max_num_groups).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, cfg.d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / cfg.d_model)
        )
        self.group_pos_embed.data[:, :, 0::2] = torch.sin(pos * div_term)
        self.group_pos_embed.data[:, :, 1::2] = torch.cos(pos * div_term)
        self.svg_decoder = SVGGroupDecoder(cfg)
        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)

    def _mask_groups(self, x, visibility_mask):
        N, G, D = x.shape
        device = x.device
        x = x + self.group_pos_embed[:, :G, :]
        len_keep = int(G * (1 - self.cfg.mask_ratio_svg))
        noise = torch.rand(N, G, device=device)
        noise[~visibility_mask] += 100.0
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_tokens = self.mask_token.repeat(N, ids_mask.shape[1], 1)
        mask_pos_embed = torch.gather(
            self.group_pos_embed.repeat(N, 1, 1),
            dim=1,
            index=ids_mask.unsqueeze(-1).repeat(1, 1, D),
        )
        mask_tokens = mask_tokens + mask_pos_embed
        mask = torch.ones([N, G], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask_tokens, ids_keep, ids_mask, mask, ids_restore

    def forward(self, batch: dict) -> TrainStep:
        commands = batch["commands"]
        args = batch["args"]
        N, G, S = commands.shape
        group_embs = self.svg_group_encoder(commands, args)
        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
        visibility = commands[:, :, 0] != EOS_idx
        x_visible, mask_tokens, ids_keep, ids_mask, mask_binary, _ = self._mask_groups(
            group_embs, visibility
        )
        cls_token = self.cls_token.expand(N, -1, -1)
        x_visible = torch.cat((cls_token, x_visible), dim=1)
        memory = self.mae_encoder(x_visible.transpose(0, 1))  # (L, N, D)
        pred_cmd_logits, pred_args_logits = self.svg_decoder(mask_tokens, memory)
        ids_mask_expanded = ids_mask.unsqueeze(-1).expand(-1, -1, S)
        tgt_commands = torch.gather(commands, 1, ids_mask_expanded)
        ids_mask_args = ids_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, self.cfg.n_args)
        tgt_args = torch.gather(args, 1, ids_mask_args)
        loss, logs = self._compute_svg_loss(
            pred_cmd_logits, pred_args_logits, tgt_commands, tgt_args
        )
        return TrainStep(loss=loss, logs=logs)

    def _compute_svg_loss(self, cmd_logits, args_logits, tgt_cmd, tgt_args):
        cmd_logits = cmd_logits.flatten(0, 1)
        args_logits = args_logits.flatten(0, 1)
        tgt_cmd = tgt_cmd.flatten(0, 1)
        tgt_args = tgt_args.flatten(0, 1)
        padding_mask = _get_padding_mask(tgt_cmd, seq_dim=-1, extended=True).bool().squeeze()
        loss_cmd = F.cross_entropy(cmd_logits.permute(0, 2, 1), tgt_cmd.long(), reduction="none")
        loss_cmd = (loss_cmd * padding_mask).sum() / padding_mask.sum().clamp(min=1.0)
        cmd_args_mask = self.cmd_args_mask[tgt_cmd.long()]
        valid_args_mask = padding_mask.unsqueeze(-1) & cmd_args_mask.bool()
        loss_args = F.cross_entropy(
            args_logits.permute(0, 3, 1, 2), (tgt_args + 1).long(), reduction="none"
        )
        loss_args = (loss_args * valid_args_mask).sum() / valid_args_mask.sum().clamp(min=1.0)
        loss = self.cfg.loss_cmd_weight * loss_cmd + self.cfg.loss_args_weight * loss_args
        return loss, {"loss_cmd": loss_cmd.item(), "loss_args": loss_args.item()}
