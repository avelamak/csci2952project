import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List

from transformers import AutoImageProcessor, AutoModel

from vecssl.models.base import JointModel, TrainStep
from vecssl.models.config import MultiMAEConfig
from vecssl.models.svgmae import SVGGroupEncoder, SVGGroupDecoder
from vecssl.models.layers.transformer import TransformerEncoder, LayerNorm
from vecssl.models.layers.improved_transformer import TransformerEncoderLayerImproved
from vecssl.data.svg_tensor import SVGTensor
from .utils import _sample_categorical


class DINOImagePatchEncoder(nn.Module):
    """
    Extracts DINO patch tokens (excludes CLS and register tokens).
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
        device = next(self.backbone.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.set_grad_enabled(self.backbone.training):
            outputs = self.backbone(**inputs)
            hs = outputs.last_hidden_state

        start = 1 + self.num_register_tokens
        patch_tokens = hs[:, start:, :]
        return patch_tokens

    @property
    def num_patches(self) -> int:
        return 196


class MultiMAE(JointModel):
    def __init__(self, cfg: MultiMAEConfig):
        super().__init__()
        self.cfg = cfg

        # SVG components
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

        # Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.mod_embed_svg = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.mod_embed_img = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.svg_mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mod_embed_svg, std=0.02)
        nn.init.trunc_normal_(self.mod_embed_img, std=0.02)
        nn.init.trunc_normal_(self.svg_mask_token, std=0.02)

        # Group Positional Embeddings (Essential for MAE)
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

        # MAE encoder
        mae_encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model,
            cfg.mae_num_heads,
            int(cfg.mae_mlp_ratio * cfg.d_model),
            cfg.mae_dropout,
        )
        mae_encoder_norm = LayerNorm(cfg.d_model)
        self.mae_encoder = TransformerEncoder(mae_encoder_layer, cfg.mae_depth, mae_encoder_norm)

        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)
        self.args_dim = cfg.args_dim + 1

    def _mask_svg_groups(self, x, visibility_mask):
        """
        Adaptive masking strategy to handle N=1..4 cases.
        """
        N, G, D = x.shape
        device = x.device

        # 1. Add Positional Embeddings BEFORE masking
        x = x + self.group_pos_embed[:, :G, :]

        # 2. Determine Valid Counts
        valid_counts = visibility_mask.sum(dim=1)  # (N,)

        # 3. Adaptive Masking Logic
        # Case A: N <= 2 (Cross-Modal). Mask EVERYTHING (0 visible). Rely on Image.
        # Case B: 3 <= N <= 4 (Gentle). Mask 50% max.
        # Case C: N > 4 (Standard). Use config ratio (e.g. 75%).

        target_masked = (valid_counts.float() * self.cfg.mask_ratio_svg).round().long()

        # Apply Gentle cap for Case B
        gentle_masked = (valid_counts.float() * 0.5).round().long()
        target_masked = torch.where(
            (valid_counts > 2) & (valid_counts <= 4), gentle_masked, target_masked
        )

        # Enforce keeping 1 visible by default...
        num_masked = torch.minimum(target_masked, valid_counts - 1)

        # ...BUT override for Case A (N<=2): Allow masking all valid groups
        num_masked = torch.where(valid_counts <= 2, valid_counts, num_masked)

        # Final safety: Don't mask if empty (N=0 padding)
        num_masked = torch.where(valid_counts == 0, torch.zeros_like(num_masked), num_masked)

        # 4. Sorting & Indexing
        noise = torch.rand(N, G, device=device)
        noise[~visibility_mask] += 100.0  # Push invalid padding to end

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # We cannot just slice [:, :len_keep] because len_keep varies per sample now!
        # We must create boolean masks based on `num_masked`

        # Create rank matrix: (N, G) -> 0, 1, 2, ...
        ranks = torch.arange(G, device=device).unsqueeze(0).expand(N, -1)

        # Mask is True if rank >= (valid - masked)
        # e.g. Valid=4, Mask=1 -> Keep=3. Ranks 0,1,2 are Keep. Ranks 3 is Mask.
        num_keep = valid_counts - num_masked
        is_masked = ranks >= num_keep.unsqueeze(1)

        # Also ensure we don't select padding as "masked" targets
        is_masked = is_masked & (ranks < valid_counts.unsqueeze(1))

        # Now we have boolean mask in SHUFFLED order.
        # We need to gather indices. Since PyTorch ragged is hard, we'll keep using
        # gather but we need to handle variable lengths carefully.

        # Actually, for the Encoder, we just need to drop the masked ones.
        # But `gather` requires fixed L dimension.
        # Simple fix: We pad the *visible* set to max_keep and the *masked* set to max_mask.

        # Let's go back to lists for robust variable length handling (batch overhead is low here)
        ids_keep_list = []
        ids_mask_list = []

        for i in range(N):
            n_k = num_keep[i].item()
            n_m = num_masked[i].item()
            ids_keep_list.append(ids_shuffle[i, :n_k])
            ids_mask_list.append(ids_shuffle[i, n_k : n_k + n_m])

        max_k = max([len(x) for x in ids_keep_list]) if ids_keep_list else 0
        max_m = max([len(x) for x in ids_mask_list]) if ids_mask_list else 0

        # Pad indices with 0 (safe because we will use a key_padding_mask later)
        ids_keep = torch.zeros(N, max_k, dtype=torch.long, device=device)
        ids_mask = torch.zeros(N, max_m, dtype=torch.long, device=device)

        keep_padding_mask = torch.ones(N, max_k, dtype=torch.bool, device=device)  # True=Pad
        mask_padding_mask = torch.ones(N, max_m, dtype=torch.bool, device=device)  # True=Pad

        for i in range(N):
            len_k = len(ids_keep_list[i])
            len_m = len(ids_mask_list[i])
            if len_k > 0:
                ids_keep[i, :len_k] = ids_keep_list[i]
                keep_padding_mask[i, :len_k] = False
            if len_m > 0:
                ids_mask[i, :len_m] = ids_mask_list[i]
                mask_padding_mask[i, :len_m] = False

        # 5. Gather Data
        # Visible
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Mask Tokens (with positional info)
        mask_tokens = self.svg_mask_token.repeat(N, max_m, 1)
        mask_pos_embed = torch.gather(
            self.group_pos_embed.repeat(N, 1, 1),
            dim=1,
            index=ids_mask.unsqueeze(-1).repeat(1, 1, D),
        )
        mask_tokens = mask_tokens + mask_pos_embed

        return x_masked, keep_padding_mask, mask_tokens, ids_mask, mask_padding_mask, ids_restore

    def _mask_image_patches(self, img_tokens: torch.Tensor):
        N, P, d_model = img_tokens.shape
        device = img_tokens.device
        num_keep = max(1, round(P * (1.0 - self.cfg.mask_ratio_img)))

        noise = torch.rand(N, P, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        visible_tokens = torch.gather(img_tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d_model))
        # Image is always valid (no padding)
        visible_mask = torch.zeros(N, num_keep, dtype=torch.bool, device=device)
        return visible_tokens, visible_mask

    def forward(self, batch: dict) -> TrainStep:
        device = next(self.parameters()).device
        commands = batch["commands"].to(device)
        args = batch["args"].to(device)
        N, G, S = commands.shape

        # 1. Encode SVG
        group_embs = self.svg_group_encoder(commands, args)

        # 2. Encode Image
        if "dino_patches" in batch:
            img_patches = batch["dino_patches"].to(device)
        else:
            images = batch["image"].to(device)
            img_patches = self.image_encoder(images)
        img_tokens = self.img_proj(img_patches)

        # 3. Masking
        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
        svg_visibility = commands[:, :, 0] != EOS_idx

        # Adaptive Masking
        (x_vis_svg, svg_pad_mask, svg_mask_tokens, ids_mask_svg, mask_tokens_pad_mask, _) = (
            self._mask_svg_groups(group_embs, svg_visibility)
        )

        x_vis_img, img_pad_mask = self._mask_image_patches(img_tokens)

        # 4. Modality Embeddings & Concat
        # Note: If x_vis_svg is empty (max_k=0), adding bias is fine (broadcasts to empty)
        if x_vis_svg.shape[1] > 0:
            x_vis_svg = x_vis_svg + self.mod_embed_svg

        x_vis_img = x_vis_img + self.mod_embed_img
        cls_token = self.cls_token.expand(N, -1, -1)

        # Handle empty SVG case for concatenation
        if x_vis_svg.shape[1] == 0:
            enc_input = torch.cat([cls_token, x_vis_img], dim=1)
            cls_pad = torch.zeros(N, 1, dtype=torch.bool, device=device)
            enc_key_padding_mask = torch.cat([cls_pad, img_pad_mask], dim=1)
        else:
            enc_input = torch.cat([cls_token, x_vis_svg, x_vis_img], dim=1)
            cls_pad = torch.zeros(N, 1, dtype=torch.bool, device=device)
            enc_key_padding_mask = torch.cat([cls_pad, svg_pad_mask, img_pad_mask], dim=1)

        # 5. MAE Encoder
        memory = self.mae_encoder(
            enc_input.transpose(0, 1), src_key_padding_mask=enc_key_padding_mask
        )  # (L, N, D)

        # 6. Decoder (Only decode SVG holes)
        # If no masks (rare/padding), return 0 loss
        if svg_mask_tokens.shape[1] == 0:
            loss = group_embs.new_tensor(0.0)
            return TrainStep(loss=loss, logs={"mae_loss": 0.0})

        pred_cmd_logits, pred_args_logits = self.svg_decoder(
            svg_mask_tokens, memory, memory_key_padding_mask=enc_key_padding_mask
        )

        # 7. Targets
        ids_mask_exp = ids_mask_svg.unsqueeze(-1).expand(-1, -1, S)
        tgt_commands = torch.gather(commands, 1, ids_mask_exp)

        ids_mask_args = ids_mask_svg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, self.cfg.n_args)
        tgt_args = torch.gather(args, 1, ids_mask_args)

        # 8. Loss (pass mask_tokens_pad_mask to ignore padding in mask tokens)
        loss, logs = self._compute_svg_loss(
            pred_cmd_logits, pred_args_logits, tgt_commands, tgt_args, mask_tokens_pad_mask
        )

        return TrainStep(loss=loss, logs={"mae_loss": loss.item(), **logs})

    def _compute_svg_loss(self, cmd_logits, args_logits, tgt_cmd, tgt_args, mask_pad_mask):
        """
        Computes loss, ignoring padded mask tokens.
        """
        # mask_pad_mask: True = Padding (Ignore)
        valid_group_mask = ~mask_pad_mask  # True = Valid

        # Flatten
        cmd_logits = cmd_logits.flatten(0, 1)
        args_logits = args_logits.flatten(0, 1)
        tgt_cmd = tgt_cmd.flatten(0, 1)
        tgt_args = tgt_args.flatten(0, 1)
        valid_group_mask = valid_group_mask.flatten(0, 1)

        # Sequence Padding Mask (EOS)
        from vecssl.models.utils import _get_padding_mask

        seq_padding_mask = _get_padding_mask(tgt_cmd, seq_dim=-1, extended=True).bool().squeeze()

        # Combined Validity: Valid Group AND Valid Sequence Position
        total_valid = seq_padding_mask & valid_group_mask.unsqueeze(-1)

        # Command Loss
        loss_cmd = F.cross_entropy(cmd_logits.permute(0, 2, 1), tgt_cmd.long(), reduction="none")
        loss_cmd = (loss_cmd * total_valid).sum() / total_valid.sum().clamp(min=1.0)

        # Args Loss
        cmd_args_mask = self.cmd_args_mask[tgt_cmd.long()]
        total_valid_args = total_valid.unsqueeze(-1) & cmd_args_mask.bool()

        loss_args = F.cross_entropy(
            args_logits.permute(0, 3, 1, 2), (tgt_args + 1).long(), reduction="none"
        )
        loss_args = (loss_args * total_valid_args).sum() / total_valid_args.sum().clamp(min=1.0)

        loss = self.cfg.loss_cmd_weight * loss_cmd + self.cfg.loss_args_weight * loss_args
        return loss, {"loss_cmd": loss_cmd.item(), "loss_args": loss_args.item()}

    @torch.no_grad()
    def greedy_reconstruct(self, batch: dict):
        device = next(self.parameters()).device
        # Ensure commands are Long, args are Float (standard for SVG tensors)
        commands = batch["commands"].to(device).long()
        args = batch["args"].to(device).float()
        N, G, S = commands.shape

        # Encode
        group_embs = self.svg_group_encoder(commands, args)
        if "dino_patches" in batch:
            img_patches = batch["dino_patches"].to(device)
        else:
            images = batch["image"].to(device)
            img_patches = self.image_encoder(images)
        img_tokens = self.img_proj(img_patches)

        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
        svg_visibility = commands[:, :, 0] != EOS_idx

        # Masking
        (
            x_vis_svg,
            svg_pad_mask,
            svg_mask_tokens,
            ids_mask_svg,
            mask_tokens_pad_mask,
            ids_restore_svg,
        ) = self._mask_svg_groups(group_embs, svg_visibility)
        x_vis_img, img_pad_mask = self._mask_image_patches(img_tokens)

        # Embed & Concat
        if x_vis_svg.shape[1] > 0:
            x_vis_svg = x_vis_svg + self.mod_embed_svg
        x_vis_img = x_vis_img + self.mod_embed_img
        cls_token = self.cls_token.expand(N, -1, -1)

        if x_vis_svg.shape[1] == 0:
            enc_input = torch.cat([cls_token, x_vis_img], dim=1)
            cls_pad = torch.zeros(N, 1, dtype=torch.bool, device=device)
            enc_key_padding_mask = torch.cat([cls_pad, img_pad_mask], dim=1)
        else:
            enc_input = torch.cat([cls_token, x_vis_svg, x_vis_img], dim=1)
            cls_pad = torch.zeros(N, 1, dtype=torch.bool, device=device)
            enc_key_padding_mask = torch.cat([cls_pad, svg_pad_mask, img_pad_mask], dim=1)

        # MAE
        memory = self.mae_encoder(
            enc_input.transpose(0, 1), src_key_padding_mask=enc_key_padding_mask
        )
        pred_cmd_logits, pred_args_logits = self.svg_decoder(
            svg_mask_tokens, memory, memory_key_padding_mask=enc_key_padding_mask
        )

        # Sample
        pred_cmd, pred_args = _sample_categorical(0.0001, pred_cmd_logits, pred_args_logits)
        pred_args -= 1

        # Make valid
        mask = self.cmd_args_mask[pred_cmd.long()].bool()
        pred_args[~mask] = -1

        # Reconstruct
        # Start from GT everywhere, then overwrite only masked groups
        # This preserves EOS tokens in padded groups (preventing spurious paths)
        final_cmd = commands.clone().long()
        final_args = args.clone().float()

        for i in range(N):
            real_n_mask = (~mask_tokens_pad_mask[i]).sum().item()
            real_n_vis = (~svg_pad_mask[i]).sum().item()

            ids_shuffle_i = ids_restore_svg[i].argsort()  # == original ids_shuffle

            # Masked indices, only these get predictions
            idx_mask = ids_shuffle_i[real_n_vis : real_n_vis + real_n_mask]

            # Only overwrite masked groups (visible groups already have GT from clone)
            final_cmd[i, idx_mask] = pred_cmd[i, :real_n_mask].long()
            final_args[i, idx_mask] = pred_args[i, :real_n_mask].float()

        return final_cmd, final_args

    @torch.no_grad()
    def encode_joint(self, batch: dict, use_images: bool = True) -> dict:
        device = next(self.parameters()).device
        commands = batch["commands"].to(device)
        args = batch["args"].to(device)
        N, G, S = commands.shape

        group_embs = self.svg_group_encoder(commands, args)

        # Add Pos Embed (Crucial for frozen encoder to make sense of geometry)
        group_embs = group_embs + self.group_pos_embed[:, :G, :]

        EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
        svg_visibility = commands[:, :, 0] != EOS_idx

        svg_tokens = group_embs + self.mod_embed_svg

        if use_images and ("image" in batch or "dino_patches" in batch):
            if "dino_patches" in batch:
                img_patches = batch["dino_patches"].to(device)
            else:
                images = batch["image"].to(device)
                img_patches = self.image_encoder(images)
            img_tokens = self.img_proj(img_patches) + self.mod_embed_img

            cls_valid = torch.zeros(N, 1, dtype=torch.bool, device=device)
            # Mask invalid SVG groups
            svg_padding = ~svg_visibility
            img_valid = torch.zeros(N, img_tokens.shape[1], dtype=torch.bool, device=device)

            enc_key_padding_mask = torch.cat([cls_valid, svg_padding, img_valid], dim=1)

            cls_token = self.cls_token.expand(N, -1, -1)
            enc_input = torch.cat([cls_token, svg_tokens, img_tokens], dim=1)
        else:
            cls_valid = torch.zeros(N, 1, dtype=torch.bool, device=device)
            svg_padding = ~svg_visibility
            enc_key_padding_mask = torch.cat([cls_valid, svg_padding], dim=1)

            cls_token = self.cls_token.expand(N, -1, -1)
            enc_input = torch.cat([cls_token, svg_tokens], dim=1)

        memory = self.mae_encoder(
            enc_input.transpose(0, 1), src_key_padding_mask=enc_key_padding_mask
        )
        memory = memory.transpose(0, 1)

        cls_embed = memory[:, 0, :]
        cls_embed = F.normalize(cls_embed, dim=-1)

        return {"svg": cls_embed}
