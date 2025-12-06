"""
Multi-modal Masked Autoencoder (MultiMAE) for SVG + Image.

Masks SVG path groups and DINO image patches, encodes visible tokens from both modalities,
reconstructs masked SVG groups to full command/args tokens using SVGLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List

from transformers import AutoImageProcessor, AutoModel

from vecssl.models.base import JointModel, TrainStep
from vecssl.models.config import MultiMAEConfig
from vecssl.models.svgmae import SVGGroupEncoder, SVGGroupDecoder
from vecssl.models.layers.transformer import TransformerEncoder, LayerNorm
from vecssl.models.layers.improved_transformer import TransformerEncoderLayerImproved


class DINOImagePatchEncoder(nn.Module):
    """
    Extracts DINO patch tokens (excludes CLS and register tokens).

    Returns: (N, num_patches, hidden_size) where hidden_size=768 for dinov2-base
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", train_backbone: bool = False):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, do_rescale=False, use_fast=True
        )
        self.backbone = AutoModel.from_pretrained(model_name)

        if not train_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)

    def forward(self, images: Union[torch.Tensor, List]) -> torch.Tensor:
        """
        Args:
            images: (N, 3, H, W) tensor or list of PIL images

        Returns:
            patch_tokens: (N, num_patches, hidden_size)
        """
        device = next(self.backbone.parameters()).device

        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.set_grad_enabled(self.backbone.training):
            outputs = self.backbone(**inputs)
            hs = outputs.last_hidden_state  # (N, 1 + num_reg + num_patches, hidden_size)

        # Strip CLS and register tokens: [CLS, reg..., patches...]
        start = 1 + self.num_register_tokens
        patch_tokens = hs[:, start:, :]  # (N, num_patches, hidden_size)

        return patch_tokens

    @property
    def num_patches(self) -> int:
        """Return number of patches for default 224x224 input (14x14 = 196 for patch_size=16)"""
        return 196  # Default for dinov2-base with 224x224 input


class MultiMAE(JointModel):
    """
    Multi-modal MAE with SVG groups + DINO image patches.

    Only reconstructs masked SVG groups (not image patches).
    Uses SVGLoss (command CE + args CE) for training.
    """

    def __init__(self, cfg: MultiMAEConfig):
        super().__init__()
        self.cfg = cfg

        # SVG components (reused from SVGMAE)
        self.svg_group_encoder = SVGGroupEncoder(cfg)
        self.svg_decoder = SVGGroupDecoder(cfg)

        # Image components
        if not cfg.use_precomputed_dino_patches:
            self.image_encoder = DINOImagePatchEncoder(
                model_name=cfg.dino_model_name, train_backbone=cfg.train_dino
            )
        else:
            self.image_encoder = None

        # Project DINO patches to d_model
        self.img_proj = nn.Linear(cfg.img_proj_dim, cfg.d_model, bias=False)

        # Modality embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.mod_embed_svg = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.mod_embed_img = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mod_embed_svg, std=0.02)
        nn.init.trunc_normal_(self.mod_embed_img, std=0.02)

        # Shared MAE encoder (processes visible SVG + visible image + CLS)
        mae_encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model,
            cfg.mae_num_heads,
            int(cfg.mae_mlp_ratio * cfg.d_model),
            cfg.mae_dropout,
        )
        mae_encoder_norm = LayerNorm(cfg.d_model)
        self.mae_encoder = TransformerEncoder(mae_encoder_layer, cfg.mae_depth, mae_encoder_norm)

        # Learnable mask token for SVG groups
        self.svg_mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.svg_mask_token, std=0.02)

        # Register cmd_args_mask buffer for loss computation
        from vecssl.data.svg_tensor import SVGTensor

        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)
        self.args_dim = cfg.args_dim + 1

    def _mask_svg_groups(
        self, group_embs: torch.Tensor, visibility: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask SVG groups for MAE training.

        Args:
            group_embs: (N, G, d_model)
            visibility: (N, G) bool

        Returns:
            visible_embs, visible_mask_pad, masked_indices, masked_mask_pad, all_mask
        """
        N, G, d_model = group_embs.shape
        device = group_embs.device

        valid_counts = visibility.sum(dim=1)
        num_masked = (valid_counts.float() * self.cfg.mask_ratio_svg).round().long()
        # At least 1 masked, at most (valid_counts - 1) to keep at least 1 visible
        num_masked = num_masked.clamp(min=1)
        num_masked = torch.minimum(num_masked, valid_counts - 1)
        num_masked = torch.where(valid_counts <= 1, torch.zeros_like(num_masked), num_masked)

        rand_perm = torch.rand(N, G, device=device)
        rand_perm = rand_perm.masked_fill(~visibility, float("inf"))
        sorted_indices = rand_perm.argsort(dim=1)

        num_visible = valid_counts - num_masked
        position_in_sort = torch.arange(G, device=device).unsqueeze(0).expand(N, -1)
        mask = position_in_sort >= num_visible.unsqueeze(1)

        inverse_indices = sorted_indices.argsort(dim=1)
        mask = mask.gather(1, inverse_indices)
        mask = mask | ~visibility

        # Gather visible and masked
        visible_indices_list = []
        masked_indices_list = []

        for i in range(N):
            vis_idx = (~mask[i]).nonzero(as_tuple=True)[0]
            msk_idx = (mask[i] & visibility[i]).nonzero(as_tuple=True)[0]
            visible_indices_list.append(vis_idx)
            masked_indices_list.append(msk_idx)

        max_vis = max(len(v) for v in visible_indices_list) if visible_indices_list else 1
        max_msk = max(len(m) for m in masked_indices_list) if masked_indices_list else 1

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

        visible_embs = group_embs.gather(1, visible_indices.unsqueeze(-1).expand(-1, -1, d_model))

        return visible_embs, visible_mask_pad, masked_indices, masked_mask_pad, mask

    def _mask_image_patches(self, img_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask image patches.

        Args:
            img_tokens: (N, P, d_model)

        Returns:
            visible_tokens: (N, K_img, d_model)
            visible_mask: (N, K_img) bool
        """
        N, P, d_model = img_tokens.shape
        device = img_tokens.device

        num_keep = max(1, round(P * (1.0 - self.cfg.mask_ratio_img)))
        num_keep = min(num_keep, P - 1)

        # Random permutation
        perm = torch.rand(N, P, device=device).argsort(dim=1)
        keep_indices = perm[:, :num_keep]  # (N, num_keep)

        # Gather visible tokens
        visible_tokens = img_tokens.gather(
            1, keep_indices.unsqueeze(-1).expand(-1, -1, d_model)
        )  # (N, num_keep, d_model)

        # All visible (no padding needed as num_keep is same for all)
        visible_mask = torch.ones(N, num_keep, dtype=torch.bool, device=device)

        return visible_tokens, visible_mask

    def _compute_svg_loss(
        self,
        command_logits: torch.Tensor,
        args_logits: torch.Tensor,
        tgt_commands: torch.Tensor,
        tgt_args: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute SVGLoss on masked groups (same as SVGMAE)."""
        N, G_masked, S = tgt_commands.shape

        from vecssl.data.svg_tensor import SVGTensor

        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")

        is_eos = tgt_commands == EOS_idx
        eos_cumsum = is_eos.cumsum(dim=-1)
        seq_valid = (eos_cumsum == 0) | ((eos_cumsum == 1) & is_eos)
        position_valid = seq_valid & valid_mask.unsqueeze(-1)

        # Command loss
        cmd_logits_shifted = command_logits[:, :, :-1, :]
        tgt_cmd_shifted = tgt_commands[:, :, 1:]
        position_valid_shifted = position_valid[:, :, 1:]

        cmd_logits_flat = cmd_logits_shifted[position_valid_shifted]
        tgt_cmd_flat = tgt_cmd_shifted[position_valid_shifted]

        if cmd_logits_flat.numel() > 0:
            loss_cmd = F.cross_entropy(cmd_logits_flat, tgt_cmd_flat.long())
        else:
            loss_cmd = command_logits.new_tensor(0.0)

        # Args loss
        args_logits_shifted = args_logits[:, :, :-1, :, :]
        tgt_args_shifted = tgt_args[:, :, 1:, :]

        cmd_args_mask = self.cmd_args_mask[tgt_cmd_shifted.long()]
        args_valid = position_valid_shifted.unsqueeze(-1) & cmd_args_mask.bool()

        args_logits_flat = args_logits_shifted[args_valid]
        tgt_args_flat = tgt_args_shifted[args_valid]

        if args_logits_flat.numel() > 0:
            loss_args = F.cross_entropy(args_logits_flat, (tgt_args_flat + 1).long())
        else:
            loss_args = args_logits.new_tensor(0.0)

        loss = self.cfg.loss_cmd_weight * loss_cmd + self.cfg.loss_args_weight * loss_args

        # Check for NaN (can happen with edge cases in masking)
        if torch.isnan(loss):
            loss = loss.new_tensor(0.0)
            loss_cmd = loss_cmd.new_tensor(0.0) if torch.isnan(loss_cmd) else loss_cmd
            loss_args = loss_args.new_tensor(0.0) if torch.isnan(loss_args) else loss_args

        logs = {"loss_cmd": loss_cmd.item(), "loss_args": loss_args.item()}

        return loss, logs

    def forward(self, batch: dict) -> TrainStep:
        """
        Forward pass for training.

        Args:
            batch: dict with "commands", "args", "image" (or "dino_patches")

        Returns:
            TrainStep with loss and logs
        """
        device = next(self.parameters()).device

        commands = batch["commands"].to(device)
        args = batch["args"].to(device)

        N, G, S = commands.shape

        # 1. Get per-group SVG embeddings
        group_embs = self.svg_group_encoder(commands, args)  # (N, G, d_model)

        # 2. Get image patch embeddings
        if "dino_patches" in batch:
            img_patches = batch["dino_patches"].to(device)  # (N, P, 768)
        else:
            images = batch["image"].to(device)
            img_patches = self.image_encoder(images)  # (N, P, 768)

        img_tokens = self.img_proj(img_patches)  # (N, P, d_model)

        # 3. Get visibility mask for SVG groups
        from vecssl.data.svg_tensor import SVGTensor

        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
        svg_visibility = commands[:, :, 0] != EOS_idx  # (N, G)

        # 4. Mask SVG groups and image patches
        (
            visible_svg,
            svg_visible_mask,
            masked_svg_indices,
            svg_masked_mask,
            svg_mask,
        ) = self._mask_svg_groups(group_embs, svg_visibility)

        visible_img, img_visible_mask = self._mask_image_patches(img_tokens)

        # 5. Add modality embeddings
        visible_svg = visible_svg + self.mod_embed_svg
        visible_img = visible_img + self.mod_embed_img

        # 6. Build encoder input: [CLS, visible_svg, visible_img]
        cls_token = self.cls_token.expand(N, -1, -1)
        enc_input = torch.cat(
            [cls_token, visible_svg, visible_img], dim=1
        )  # (N, 1+K_svg+K_img, d_model)

        # Key padding mask
        cls_valid = torch.ones(N, 1, dtype=torch.bool, device=device)
        enc_key_padding_mask = ~torch.cat([cls_valid, svg_visible_mask, img_visible_mask], dim=1)

        # 7. Encode with MAE encoder
        enc_input_sf = enc_input.transpose(0, 1)
        memory = self.mae_encoder(
            enc_input_sf, mask=None, src_key_padding_mask=enc_key_padding_mask, memory2=None
        )
        memory = memory.transpose(0, 1)  # (N, 1+K_svg+K_img, d_model)

        # 8. Decode masked SVG groups
        num_masked = svg_masked_mask.sum(dim=1).max().item()
        if num_masked == 0:
            loss = group_embs.new_tensor(0.0)
            logs = {"mae_loss": 0.0, "loss_cmd": 0.0, "loss_args": 0.0}
            return TrainStep(loss=loss, logs=logs)

        # Use mask_token as input to decoder (it will cross-attend to memory for context)
        masked_z = self.svg_mask_token.expand(N, num_masked, -1)
        # The decoder cross-attends to the encoder memory (encoded visible groups/patches + CLS)
        command_logits, args_logits = self.svg_decoder(
            masked_z, memory=memory, memory_key_padding_mask=enc_key_padding_mask
        )

        # 9. Get target commands/args for masked groups
        masked_indices_exp = masked_svg_indices[:, :num_masked]
        tgt_commands = commands.gather(1, masked_indices_exp.unsqueeze(-1).expand(-1, -1, S))
        tgt_args = args.gather(
            1, masked_indices_exp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, self.cfg.n_args)
        )

        # 10. Compute loss
        loss, loss_logs = self._compute_svg_loss(
            command_logits, args_logits, tgt_commands, tgt_args, svg_masked_mask[:, :num_masked]
        )

        logs = {"mae_loss": loss.item(), **loss_logs}

        return TrainStep(loss=loss, logs=logs)

    @torch.no_grad()
    def encode_joint(self, batch: dict, use_images: bool = True) -> dict:
        """
        Encode SVG (optionally with image context) to latent embedding.

        Args:
            batch: dict with "commands", "args", and optionally "image"/"dino_patches"
            use_images: whether to include image context

        Returns:
            dict with "svg": (N, d_model) normalized embedding
        """
        device = next(self.parameters()).device

        commands = batch["commands"].to(device)
        args = batch["args"].to(device)

        N, G, S = commands.shape

        # Get per-group SVG embeddings
        group_embs = self.svg_group_encoder(commands, args)

        # Get visibility mask
        from vecssl.data.svg_tensor import SVGTensor

        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
        svg_visibility = commands[:, :, 0] != EOS_idx

        # Add modality embedding
        svg_tokens = group_embs + self.mod_embed_svg

        # Optionally get image tokens
        if use_images and ("image" in batch or "dino_patches" in batch):
            if "dino_patches" in batch:
                img_patches = batch["dino_patches"].to(device)
            else:
                images = batch["image"].to(device)
                img_patches = self.image_encoder(images)
            img_tokens = self.img_proj(img_patches) + self.mod_embed_img

            # Key padding mask
            cls_valid = torch.ones(N, 1, dtype=torch.bool, device=device)
            img_valid = torch.ones(N, img_tokens.shape[1], dtype=torch.bool, device=device)
            enc_key_padding_mask = ~torch.cat([cls_valid, svg_visibility, img_valid], dim=1)

            # Encoder input
            cls_token = self.cls_token.expand(N, -1, -1)
            enc_input = torch.cat([cls_token, svg_tokens, img_tokens], dim=1)
        else:
            # SVG only
            cls_valid = torch.ones(N, 1, dtype=torch.bool, device=device)
            enc_key_padding_mask = ~torch.cat([cls_valid, svg_visibility], dim=1)

            cls_token = self.cls_token.expand(N, -1, -1)
            enc_input = torch.cat([cls_token, svg_tokens], dim=1)

        # Encode
        enc_input_sf = enc_input.transpose(0, 1)
        memory = self.mae_encoder(
            enc_input_sf, mask=None, src_key_padding_mask=enc_key_padding_mask, memory2=None
        )
        memory = memory.transpose(0, 1)

        cls_embed = memory[:, 0, :]
        cls_embed = F.normalize(cls_embed, dim=-1)

        return {"svg": cls_embed}
