"""
SVG-only Masked Autoencoder (SVGMAE).

Masks path groups, encodes visible groups, reconstructs masked groups
to full command/args tokens using SVGLoss.
"""

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


class SVGGroupEncoder(nn.Module):
    """
    Encodes SVG into per-group embeddings using Stage 1 of existing Encoder pattern.

    Input: commands (N, G, S), args (N, G, S, n_args) - batch first
    Output: group_embeddings (N, G, d_model)
    """

    def __init__(self, cfg: SVGMAEConfig):
        super().__init__()
        self.cfg = cfg

        # SVG embedding (command + args -> d_model)
        self.embedding = SVGEmbedding(cfg, cfg.max_seq_len, use_group=False)

        # Stage 1 transformer encoder (processes each group independently)
        encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout
        )
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, commands: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
        """
        Args:
            commands: (N, G, S) command indices
            args: (N, G, S, n_args) argument values

        Returns:
            group_embeddings: (N, G, d_model) per-group embeddings
        """
        N, G, S = commands.shape

        # Convert to seq-first: (S, G, N)
        commands_sf, args_sf = _make_seq_first(commands, args)

        # Pack groups into batch: (S, G*N)
        commands_packed, args_packed = _pack_group_batch(commands_sf, args_sf)

        # Get masks
        padding_mask = _get_padding_mask(commands_packed, seq_dim=0)  # (S, 1)
        key_padding_mask = _get_key_padding_mask(commands_packed, seq_dim=0)  # (G*N,)

        # Embed and encode
        src = self.embedding(commands_packed, args_packed)  # (S, G*N, d_model)
        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask, memory2=None)

        # Pool over sequence dimension (weighted by padding mask)
        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(
            dim=0, keepdim=True
        ).clamp(min=1.0)  # (1, G*N, d_model)

        # Unpack back to (1, G, N, d_model)
        z = _unpack_group_batch(N, z)

        # Convert to batch-first: (N, G, d_model)
        z = z.squeeze(0).permute(1, 0, 2)  # (N, G, d_model)

        return z


class SVGGroupDecoder(nn.Module):
    """
    Decodes group latents back to commands/args tokens using cross-attention to encoder memory.

    For each masked group, generates a sequence of commands/args by:
    1. Using learnable position queries
    2. Cross-attending to the encoded visible groups (memory)
    3. Projecting to command/args logits

    Input: z (N, G_masked, d_model), memory (N, L_enc, d_model)
    Output: command_logits (N, G_masked, S, n_commands),
            args_logits (N, G_masked, S, n_args, args_dim)
    """

    def __init__(self, cfg: SVGMAEConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable query embeddings for each position in sequence (one-shot decoding)
        # +2 for SOS and EOS tokens
        self.const_embedding = ConstEmbedding(cfg, cfg.max_seq_len + 2)

        # Cross-attention decoder using standard TransformerDecoderLayer
        # This allows proper cross-attention to the encoder memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=False,  # seq-first format
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.n_layers_decode)

        # Output heads (same as existing FCN)
        args_dim = cfg.args_dim + 1  # +1 for PAD value shifted
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def forward(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (N, G_masked, d_model) mask tokens for masked groups (with encoder context added)
            memory: (N, L_enc, d_model) encoded visible groups from MAE encoder
            memory_key_padding_mask: (N, L_enc) optional padding mask for memory

        Returns:
            command_logits: (N, G_masked, S, n_commands)
            args_logits: (N, G_masked, S, n_args, args_dim)
        """
        N, G_masked, d_model = z.shape
        L_enc = memory.shape[1]
        S = self.cfg.max_seq_len + 2  # +2 for SOS and EOS tokens

        # For each masked group, we need to decode a sequence while cross-attending to memory
        # Strategy: pack groups into batch dimension, repeat memory for each group

        # z: (N, G_masked, d_model) -> (N * G_masked, d_model)
        z_flat = z.reshape(N * G_masked, d_model)

        # Repeat memory for each masked group: (N, L_enc, d_model) -> (N * G_masked, L_enc, d_model)
        memory_expanded = (
            memory.unsqueeze(1).expand(-1, G_masked, -1, -1).reshape(N * G_masked, L_enc, d_model)
        )
        memory_sf = memory_expanded.transpose(0, 1)  # (L_enc, N * G_masked, d_model)

        # Expand memory padding mask if provided
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = (
                memory_key_padding_mask.unsqueeze(1)
                .expand(-1, G_masked, -1)
                .reshape(N * G_masked, L_enc)
            )

        # Get learnable query tokens conditioned on z: (S, N * G_masked, d_model)
        tgt = self.const_embedding(z_flat.unsqueeze(0))  # (S, N * G_masked, d_model)

        # Decode with cross-attention to memory
        out = self.decoder(
            tgt,
            memory_sf,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (S, N * G_masked, d_model)

        # Get command and args logits
        command_logits, args_logits = self.fcn(
            out
        )  # (S, N*G_masked, n_commands), (S, N*G_masked, n_args, args_dim)

        # Reshape back to (N, G_masked, S, ...)
        command_logits = command_logits.view(S, N, G_masked, -1).permute(
            1, 2, 0, 3
        )  # (N, G_masked, S, n_commands)
        args_logits = args_logits.view(S, N, G_masked, self.cfg.n_args, -1).permute(
            1, 2, 0, 3, 4
        )  # (N, G_masked, S, n_args, args_dim)

        return command_logits, args_logits


class SVGMAE(JointModel):
    """
    SVG-only Masked Autoencoder.

    Masks some path groups, encodes visible groups, decodes masked groups to tokens.
    Uses SVGLoss (command CE + args CE) for training.
    """

    def __init__(self, cfg: SVGMAEConfig):
        super().__init__()
        self.cfg = cfg

        # SVG group encoder (Stage 1)
        self.svg_group_encoder = SVGGroupEncoder(cfg)

        # CLS token for overall representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # MAE encoder (processes visible groups + CLS)
        mae_encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model,
            cfg.mae_num_heads,
            int(cfg.mae_mlp_ratio * cfg.d_model),
            cfg.mae_dropout,
        )
        mae_encoder_norm = LayerNorm(cfg.d_model)
        self.mae_encoder = TransformerEncoder(mae_encoder_layer, cfg.mae_depth, mae_encoder_norm)

        # Learnable mask token (one per masked group)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # SVG decoder (predicts tokens for masked groups)
        self.svg_decoder = SVGGroupDecoder(cfg)

        # Register cmd_args_mask buffer for loss computation
        from vecssl.data.svg_tensor import SVGTensor

        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)
        self.args_dim = cfg.args_dim + 1

    def _mask_groups(
        self, group_embs: torch.Tensor, visibility: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask groups for MAE training.

        Args:
            group_embs: (N, G, d_model) per-group embeddings
            visibility: (N, G) bool mask of valid groups (optional)

        Returns:
            visible_embs: (N, K, d_model) embeddings of visible groups
            visible_indices: (N, K) indices of visible groups
            masked_indices: (N, M) indices of masked groups
            mask: (N, G) bool, True = masked
        """
        N, G, d_model = group_embs.shape
        device = group_embs.device

        # Default: all groups are valid
        if visibility is None:
            visibility = torch.ones(N, G, dtype=torch.bool, device=device)

        # Count valid groups per sample
        valid_counts = visibility.sum(dim=1)  # (N,)

        # Compute number of masked groups per sample
        num_masked = (valid_counts.float() * self.cfg.mask_ratio_svg).round().long()
        # At least 1 masked, at most (valid_counts - 1) to keep at least 1 visible
        num_masked = num_masked.clamp(min=1)
        num_masked = torch.minimum(num_masked, valid_counts - 1)

        # For samples with 0 or 1 valid groups, don't mask
        num_masked = torch.where(valid_counts <= 1, torch.zeros_like(num_masked), num_masked)

        # Random permutation for each sample
        rand_perm = torch.rand(N, G, device=device)
        rand_perm = rand_perm.masked_fill(~visibility, float("inf"))  # Invalid groups go to end
        sorted_indices = rand_perm.argsort(dim=1)  # (N, G)

        # Create mask: first (valid - num_masked) indices are visible, rest are masked
        num_visible = valid_counts - num_masked  # (N,)
        position_in_sort = torch.arange(G, device=device).unsqueeze(0).expand(N, -1)  # (N, G)
        mask = position_in_sort >= num_visible.unsqueeze(1)  # (N, G) True = masked

        # Reorder mask back to original group order
        inverse_indices = sorted_indices.argsort(dim=1)
        mask = mask.gather(1, inverse_indices)  # (N, G) in original order

        # Also mark invalid groups as masked (for consistency)
        mask = mask | ~visibility

        # Gather visible and masked embeddings
        # visible_embs: select groups where mask is False
        # This is variable length per sample, so we pad

        # For simplicity, use fixed-size tensors with padding
        visible_embs_list = []
        visible_indices_list = []
        masked_indices_list = []

        for i in range(N):
            vis_idx = (~mask[i]).nonzero(as_tuple=True)[0]  # indices of visible groups
            msk_idx = (mask[i] & visibility[i]).nonzero(as_tuple=True)[
                0
            ]  # indices of valid masked groups

            visible_embs_list.append(vis_idx)
            visible_indices_list.append(vis_idx)
            masked_indices_list.append(msk_idx)

        # Find max lengths
        max_vis = max(len(v) for v in visible_indices_list) if visible_indices_list else 1
        max_msk = max(len(m) for m in masked_indices_list) if masked_indices_list else 1

        # Pad and stack
        visible_indices = torch.zeros(N, max_vis, dtype=torch.long, device=device)
        masked_indices = torch.zeros(N, max_msk, dtype=torch.long, device=device)
        visible_mask_pad = torch.zeros(N, max_vis, dtype=torch.bool, device=device)
        masked_mask_pad = torch.zeros(N, max_msk, dtype=torch.bool, device=device)

        for i in range(N):
            v_len = len(visible_indices_list[i])
            m_len = len(masked_indices_list[i])
            if v_len > 0:
                visible_indices[i, :v_len] = visible_indices_list[i]
                visible_mask_pad[i, :v_len] = True
            if m_len > 0:
                masked_indices[i, :m_len] = masked_indices_list[i]
                masked_mask_pad[i, :m_len] = True

        # Gather embeddings
        visible_embs = group_embs.gather(
            1, visible_indices.unsqueeze(-1).expand(-1, -1, d_model)
        )  # (N, max_vis, d_model)

        return (
            visible_embs,
            visible_indices,
            masked_indices,
            mask,
            visible_mask_pad,
            masked_mask_pad,
        )

    def _compute_svg_loss(
        self,
        command_logits: torch.Tensor,
        args_logits: torch.Tensor,
        tgt_commands: torch.Tensor,
        tgt_args: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute SVGLoss on masked groups.

        Args:
            command_logits: (N, G_masked, S, n_commands)
            args_logits: (N, G_masked, S, n_args, args_dim)
            tgt_commands: (N, G_masked, S)
            tgt_args: (N, G_masked, S, n_args)
            valid_mask: (N, G_masked) bool, True = valid masked group

        Returns:
            loss: scalar tensor
            logs: dict of loss components
        """
        N, G_masked, S = tgt_commands.shape

        # Get padding mask for sequences within each group
        # Flatten to (N*G_masked, S) for easier processing
        tgt_cmd_flat = tgt_commands.view(N * G_masked, S)
        padding_mask = _get_padding_mask(
            tgt_cmd_flat.transpose(0, 1), seq_dim=0, extended=True
        )  # (S, 1)
        padding_mask = padding_mask.squeeze(-1).transpose(0, 1)  # (N*G_masked, S) - WRONG

        # Actually, we need per-sample padding mask
        # Simpler approach: use the valid_mask to filter valid groups
        # and compute CE on flattened valid positions

        # Create overall validity mask: (N, G_masked, S)
        # A position is valid if: group is valid AND position is not padded (before EOS)
        from vecssl.data.svg_tensor import SVGTensor

        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")

        # Find EOS positions
        is_eos = tgt_commands == EOS_idx  # (N, G_masked, S)
        # Cumsum to find positions after first EOS
        eos_cumsum = is_eos.cumsum(dim=-1)  # positions after first EOS have cumsum > 0
        # Valid positions: cumsum == 0 (before EOS) OR is_eos (include EOS itself)
        seq_valid = (eos_cumsum == 0) | ((eos_cumsum == 1) & is_eos)  # (N, G_masked, S)

        # Combine with group validity
        position_valid = seq_valid & valid_mask.unsqueeze(-1)  # (N, G_masked, S)

        # Command loss: skip SOS (index 0), so we compare logits[:, :, :-1] with targets[:, :, 1:]
        cmd_logits_shifted = command_logits[:, :, :-1, :]  # (N, G_masked, S-1, n_commands)
        tgt_cmd_shifted = tgt_commands[:, :, 1:]  # (N, G_masked, S-1)
        position_valid_shifted = position_valid[:, :, 1:]  # (N, G_masked, S-1)

        # Flatten and compute CE
        cmd_logits_flat = cmd_logits_shifted[position_valid_shifted]  # (num_valid, n_commands)
        tgt_cmd_flat = tgt_cmd_shifted[position_valid_shifted]  # (num_valid,)

        if cmd_logits_flat.numel() > 0:
            loss_cmd = F.cross_entropy(cmd_logits_flat, tgt_cmd_flat.long())
        else:
            loss_cmd = command_logits.new_tensor(0.0)

        # Args loss: only compute for valid args (use cmd_args_mask)
        args_logits_shifted = args_logits[:, :, :-1, :, :]  # (N, G_masked, S-1, n_args, args_dim)
        tgt_args_shifted = tgt_args[:, :, 1:, :]  # (N, G_masked, S-1, n_args)

        # Get mask for which args are valid per command
        cmd_args_mask = self.cmd_args_mask[tgt_cmd_shifted.long()]  # (N, G_masked, S-1, n_args)
        args_valid = (
            position_valid_shifted.unsqueeze(-1) & cmd_args_mask.bool()
        )  # (N, G_masked, S-1, n_args)

        args_logits_flat = args_logits_shifted[args_valid]  # (num_valid_args, args_dim)
        tgt_args_flat = tgt_args_shifted[args_valid]  # (num_valid_args,)

        if args_logits_flat.numel() > 0:
            # Shift args by +1 due to PAD_VAL = -1
            loss_args = F.cross_entropy(args_logits_flat, (tgt_args_flat + 1).long())
        else:
            loss_args = args_logits.new_tensor(0.0)

        # Weighted sum
        loss = self.cfg.loss_cmd_weight * loss_cmd + self.cfg.loss_args_weight * loss_args

        # Check for NaN (can happen with edge cases in masking)
        if torch.isnan(loss):
            loss = loss.new_tensor(0.0)
            loss_cmd = loss_cmd.new_tensor(0.0) if torch.isnan(loss_cmd) else loss_cmd
            loss_args = loss_args.new_tensor(0.0) if torch.isnan(loss_args) else loss_args

        logs = {
            "loss_cmd": loss_cmd.item(),
            "loss_args": loss_args.item(),
        }

        return loss, logs

    def forward(self, batch: dict) -> TrainStep:
        """
        Forward pass for training.

        Args:
            batch: dict with "commands" (N, G, S), "args" (N, G, S, n_args)

        Returns:
            TrainStep with loss and logs
        """
        device = next(self.parameters()).device

        commands = batch["commands"].to(device)  # (N, G, S)
        args = batch["args"].to(device)  # (N, G, S, n_args)

        N, G, S = commands.shape

        # 1. Get per-group embeddings
        group_embs = self.svg_group_encoder(commands, args)  # (N, G, d_model)

        # 2. Get visibility mask (which groups are actually valid)
        # Check first command - if it's EOS, the group is empty
        from vecssl.data.svg_tensor import SVGTensor

        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
        visibility = commands[:, :, 0] != EOS_idx  # (N, G)

        # 3. Mask groups
        (
            visible_embs,
            visible_indices,
            masked_indices,
            mask,
            visible_mask_pad,
            masked_mask_pad,
        ) = self._mask_groups(group_embs, visibility)

        # 4. Prepare encoder input: [CLS] + visible_groups
        cls_token = self.cls_token.expand(N, -1, -1)  # (N, 1, d_model)
        enc_input = torch.cat([cls_token, visible_embs], dim=1)  # (N, 1+K, d_model)

        # Create key padding mask for encoder (True = padding)
        cls_valid = torch.ones(N, 1, dtype=torch.bool, device=device)
        enc_key_padding_mask = ~torch.cat([cls_valid, visible_mask_pad], dim=1)  # (N, 1+K)

        # 5. Encode with MAE encoder
        # Convert to seq-first: (1+K, N, d_model)
        enc_input_sf = enc_input.transpose(0, 1)
        memory = self.mae_encoder(
            enc_input_sf, mask=None, src_key_padding_mask=enc_key_padding_mask, memory2=None
        )
        memory = memory.transpose(0, 1)  # (N, 1+K, d_model)

        # 6. Prepare decoder input: mask tokens for masked groups
        num_masked = masked_mask_pad.sum(dim=1).max().item()  # Max number of masked groups
        if num_masked == 0:
            # Edge case: no masked groups (shouldn't happen in normal training)
            loss = group_embs.new_tensor(0.0)
            logs = {"mae_loss": 0.0, "loss_cmd": 0.0, "loss_args": 0.0}
            return TrainStep(loss=loss, logs=logs)

        # Get embeddings for masked groups from original group_embs (these are targets)
        # Use mask_token as input to decoder (it will cross-attend to memory for context)
        masked_z = self.mask_token.expand(N, num_masked, -1)  # (N, M, d_model)

        # 7. Decode masked groups to command/args tokens
        # The decoder cross-attends to the encoder memory (encoded visible groups + CLS)
        command_logits, args_logits = self.svg_decoder(
            masked_z, memory=memory, memory_key_padding_mask=enc_key_padding_mask
        )

        # 8. Get target commands/args for masked groups
        masked_indices_exp = masked_indices[:, :num_masked]
        tgt_commands = commands.gather(
            1, masked_indices_exp.unsqueeze(-1).expand(-1, -1, S)
        )  # (N, M, S)
        tgt_args = args.gather(
            1, masked_indices_exp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, self.cfg.n_args)
        )  # (N, M, S, n_args)

        # 9. Compute loss
        loss, loss_logs = self._compute_svg_loss(
            command_logits, args_logits, tgt_commands, tgt_args, masked_mask_pad[:, :num_masked]
        )

        logs = {"mae_loss": loss.item(), **loss_logs}

        return TrainStep(loss=loss, logs=logs)

    @torch.no_grad()
    def encode_joint(self, batch: dict) -> dict:
        """
        Encode SVG to latent embedding (no masking).

        Args:
            batch: dict with "commands", "args"

        Returns:
            dict with "svg": (N, d_model) normalized embedding
        """
        device = next(self.parameters()).device

        commands = batch["commands"].to(device)
        args = batch["args"].to(device)

        N = commands.shape[0]

        # Get per-group embeddings
        group_embs = self.svg_group_encoder(commands, args)  # (N, G, d_model)

        # Get visibility mask
        from vecssl.data.svg_tensor import SVGTensor

        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
        visibility = commands[:, :, 0] != EOS_idx  # (N, G)

        # Encode all groups (no masking)
        cls_token = self.cls_token.expand(N, -1, -1)  # (N, 1, d_model)
        enc_input = torch.cat([cls_token, group_embs], dim=1)  # (N, 1+G, d_model)

        # Key padding mask
        cls_valid = torch.ones(N, 1, dtype=torch.bool, device=device)
        enc_key_padding_mask = ~torch.cat([cls_valid, visibility], dim=1)  # (N, 1+G)

        # Encode
        enc_input_sf = enc_input.transpose(0, 1)
        memory = self.mae_encoder(
            enc_input_sf, mask=None, src_key_padding_mask=enc_key_padding_mask, memory2=None
        )
        memory = memory.transpose(0, 1)

        cls_embed = memory[:, 0, :]  # (N, d_model)
        cls_embed = F.normalize(cls_embed, dim=-1)

        return {"svg": cls_embed}
