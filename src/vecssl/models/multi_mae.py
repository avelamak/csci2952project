from vecssl.models.base import JointModel, TrainStep

from vecssl.models.model import Encoder, Decoder, ImageDecoderMAE
from torchvision.models import vit_b_16
import torch
import torch.nn as nn
import torch.nn.functional as F
from vecssl.util import _make_seq_first
# from timm.models.vision_transformer import PatchEmbed, Block


class MultiMAEModel(JointModel):
    def __init__(self, cfg, logger):
        super().__init__()
        self.cfg = cfg
        self.logger = logger

        self.logger.info("[bold cyan]Init MultiMae[/bold cyan]", extra={"markup": True})

        # Init the image encoder
        # maybe try ViT instead over here? or should I try using Dino?
        # self.image_encoder = #init image encoder here

        self.image_encoder = vit_b_16(weights=None)  # No pre-trained weights
        self.image_encoder.heads = nn.Identity()  # Remove classification head
        self.image_encoder = self.image_encoder.float()

        ##what is this step doing??
        # img_dim = self.image_encoder.hidden_dim
        self.logger.info(
            "[bold cyan]Image Encoder Init Successful[/bold cyan]", extra={"markup": True}
        )

        # Init the svg encoder
        self.svg_encoder = Encoder(cfg)
        self.logger.info(
            "[bold cyan]SVG Encoder Init Successful[/bold cyan]", extra={"markup": True}
        )

        # Init the image decoder to reconstruct

        self.image_decoder = ImageDecoderMAE(cfg)
        self.logger.info(
            "[bold cyan]Image Decoder Init Successful[/bold cyan]", extra={"markup": True}
        )

        # Init the svg decoder to reconstruct the svg
        self.svg_decoder = Decoder(cfg)
        self.logger.info(
            "[bold cyan]SVG Decoder Init Successful[/bold cyan]", extra={"markup": True}
        )
        # self.fusion = fusion_module

    def forward_img_encoder(self, x, mask_ratio, center_masking=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if center_masking:
            x, mask, ids_restore = self.random_center_masking(x, mask_ratio)

        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def mask_svgs(
        self, commands, args, mask_ratio=0.15, pad_val=-1, mask_token_cmd=None, mask_token_args=None
    ):
        """
        The goal is to first mask out entire commands and their correspodning args.
        """
        seq_len, num_groups, batch_size = commands.shape
        # n_args = args.shape[-1]

        commands_masked = commands.clone()
        args_masked = args.clone()

        num_to_mask = max(1, int(seq_len * mask_ratio))

        mask_positions = torch.zeros_like(
            commands, dtype=torch.bool
        )  # [seq_len, num_groups, batch_size]

        for g in range(num_groups):
            for b in range(batch_size):
                perm = torch.randperm(seq_len)[:num_to_mask]
                mask_positions[perm, g, b] = True

        commands_masked[mask_positions] = pad_val
        args_masked[mask_positions] = pad_val

        return commands_masked, args_masked, mask_positions

    def forward(self, batch):
        """
        Forward pass for multi-modal (image + SVG) masked autoencoder.
        Logs shapes and sanity-check info using rich logger.
        """

        device = next(self.image_encoder.parameters()).device
        # print(batch)

        # img = batch["image"].to(device)
        args = batch["args"].to(device)
        cmds = batch["commands"].to(device)

        # print(
        #     f"This is args {args.shape} this is commands shape {cmds.shape}, this is img: {img.shape}"
        # )

        # make it seq first
        commands_enc, args_enc = _make_seq_first(cmds, args)

        ##masking time

        # print(
        #     f"This is args shape after seq first {args_enc.shape} this is commands shape after seq first {commands_enc.shape}, this is img: {img.shape}"
        # )
        # what do we do here first, encode the SVG?
        self.logger.info("[bold cyan]SVG Encoding...[/bold cyan]", extra={"markup": True})

        # what dim should z_svg be in [1,1,4,256], [4,256]?
        # default returns [1,1,4,256]
        z_svg = self.svg_encoder(commands_enc, args_enc)

        self.logger.info(
            "[bold cyan]SVG embedded in latent dim z[/bold cyan]", extra={"markup": True}
        )

        self.logger.info(
            f"[bold cyan]SVG embedded in latent dim z {z_svg.shape}[/bold cyan]",
            extra={"markup": True},
        )

        # let's reconstruct the svg, the logits that Jack was talking about
        recon_cmds_logits, recon_args_logits, *rest = self.svg_decoder(
            z_svg, commands_enc, args_enc
        )

        seq_len, num_groups, batch_size, n_args, arg_dim = recon_args_logits.shape

        self.logger.info(
            f"[bold cyan]recon_cmds_logits shape {recon_cmds_logits.shape}[/bold cyan]",
            extra={"markup": True},
        )

        self.logger.info(
            f"[bold cyan]recon_args_logits shape {recon_args_logits.shape}[/bold cyan]",
            extra={"markup": True},
        )

        PAD_VAL = -1

        # -------------------------
        # COMMANDS LOSS
        # -------------------------
        # recon_cmds_logits: [seq_len, num_groups, batch, n_commands]
        # commands_enc:      [seq_len_total, num_groups, batch]

        seq_len, num_groups, batch_size, n_commands = recon_cmds_logits.shape

        # Flatten logits to [total_valid_commands, n_commands]
        cmd_logits_flat = recon_cmds_logits.permute(2, 1, 0, 3).reshape(-1, n_commands)

        # Flatten targets to [total_commands]
        cmd_targets_flat = commands_enc[:seq_len].permute(2, 1, 0).reshape(-1).long()

        ##THIS WAS THE SOURCE OF SHAPE ERRORS, HOW BEST TO DEAL WITH PAD_VAL when working with reconstructions?
        ##Currently only compute loss on unpadded values
        # Mask padding
        mask = cmd_targets_flat != PAD_VAL
        cmd_logits_flat = cmd_logits_flat[mask]
        cmd_targets_flat = cmd_targets_flat[mask]

        # Compute CE
        loss_cmd = F.cross_entropy(cmd_logits_flat, cmd_targets_flat)

        # -------------------------
        # ARGS LOSS
        # -------------------------
        # recon_args_logits: [seq_len, num_groups, batch, n_args, arg_dim]
        # args_enc:          [seq_len_total, num_groups, batch, n_args]

        seq_len, num_groups, batch_size, n_args, arg_dim = recon_args_logits.shape

        # Flatten logits to [total_args, arg_dim]
        args_logits_flat = recon_args_logits.permute(2, 1, 0, 3, 4).reshape(-1, arg_dim)

        # Flatten targets to [total_args]
        args_targets_flat = args_enc[:seq_len].permute(2, 1, 0, 3).reshape(-1).long()

        ##THIS WAS THE SOURCE OF SHAPE ERRORS, HOW BEST TO DEAL WITH PAD_VAL when working with reconstructions?
        ##Currently only compute loss on unpadded values
        # Mask padding
        mask = args_targets_flat != PAD_VAL
        args_logits_flat = args_logits_flat[mask]
        args_targets_flat = args_targets_flat[mask]

        # Compute CE
        loss_args = F.cross_entropy(args_logits_flat, args_targets_flat)

        # -------------------------
        # TOTAL LOSS
        # -------------------------
        loss = loss_cmd + loss_args

        return TrainStep(
            loss=loss,
            logs={
                "loss": loss.item(),
                "loss_cmds": loss_cmd.item(),
                "loss_args": loss_args.item(),
            },
        )

    def check_gradients(self):
        """Check gradient flow through the model"""
        gradient_info = {}
        total_params = 0
        params_with_grad = 0
        max_grad = 0.0
        min_grad = float("inf")

        for name, param in self.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
                    grad_norm = param.grad.norm().item()
                    gradient_info[name] = {
                        "shape": tuple(param.shape),
                        "grad_norm": grad_norm,
                        "has_grad": True,
                    }
                    max_grad = max(max_grad, grad_norm)
                    min_grad = min(min_grad, grad_norm)
                else:
                    gradient_info[name] = {"shape": tuple(param.shape), "has_grad": False}

        summary = {
            "total_params": total_params,
            "params_with_grad": params_with_grad,
            "max_grad_norm": max_grad,
            "min_grad_norm": min_grad if min_grad != float("inf") else 0.0,
            "gradient_flow_percentage": (params_with_grad / total_params * 100)
            if total_params > 0
            else 0,
        }

        return gradient_info, summary

    # def forward_2(self):
    # self.logger.info("[bold cyan]Starting forward pass...[/bold cyan]", extra={"markup": True})
    # device = next(self.image_encoder.parameters()).device
    # # --- (1) Extract batch data ---
    # img = batch["image"].to(device)  # (B, 3, H, W)
    # # svg_cmds = batch["commands"].to(device)  # (B, N_groups, seq_len)
    # # svg_args = batch["args"].to(device)

    # img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
    # img_masked, img_mask = self.mask_image(img)

    # self.logger.info(
    #     f"[bold yellow]Input image shape:[/bold yellow] {tuple(img.shape)}",
    #     extra={"markup": True},
    # )

    # # --- (2) Apply masking ---
    # # img_masked, img_mask = self.mask_image(img)
    # # img_masked = F.interpolate(img_masked, size=(224, 224), mode='bilinear', align_corners=False)

    # self.logger.info(
    #     f"[bold yellow]Masked image shape:[/bold yellow] {tuple(img_masked.shape)}",
    #     extra={"markup": True},
    # )
    # self.logger.info(
    #     f"[bold yellow]Mask shape:[/bold yellow] {tuple(img_mask.shape)}",
    #     extra={"markup": True},
    # )

    # # --- (3) Encode visible tokens ---
    # img_latents = self.image_encoder(img_masked)
    # self.logger.info(
    #     f"[bold green]Image latents shape:[/bold green] {tuple(img_latents.shape)}",
    #     extra={"markup": True},
    # )

    # # For now, skip SVG encoder & fusion — use only image latents
    # if img_latents.shape[1] != img_mask.shape[1]:
    #     img_latents = img_latents[:, 1:, :]
    # fused = img_latents

    # # --- (4) Decode / reconstruct ---
    # img_recon = self.image_decoder(fused, img_mask)
    # self.logger.info(
    #     f"[bold magenta]Reconstructed image shape:[/bold magenta] {tuple(img_recon.shape)}",
    #     extra={"markup": True},
    # )

    # # --- (5) Visualize (sanity check) ---
    # B = 0
    # self.logger.info(
    #     "[bold blue]Visualizing reconstruction for sanity check...[/bold blue]",
    #     extra={"markup": True},
    # )

    # self.show_images(
    #     original=img[B],
    #     masked=img_masked[B],
    #     recon=img_recon[B],
    #     title="Sanity check reconstruction",
    # )

    # self.logger.info(
    #     "[bold cyan]Forward pass completed successfully.[/bold cyan]", extra={"markup": True}
    # )
    # return 0

    def encode_joint(self, batch):
        pass


#     def mask_image(self, img):
#         """
#         img: [B, C, H, W]
#         Returns:
#             img_masked: [B, C, H, W] masked image
#             mask_patches: [B, N] boolean patch mask
#         """
#         B, C, H, W = img.shape
#         p = 16  # patch size
#         n_patches_H = H // p
#         n_patches_W = W // p
#         N = n_patches_H * n_patches_W

#         # create random patch mask
#         mask_patches = torch.zeros(B, N, dtype=torch.bool, device=img.device)
#         for b in range(B):
#             n_mask = int(0.25 * N)
#             idx = torch.randperm(N)[:n_mask]
#             mask_patches[b, idx] = True

#         # reshape img into patches
#         img_patches = img.unfold(2, p, p).unfold(3, p, p)  # [B,C,H//p,W//p,p,p]
#         img_patches_flat = img_patches.reshape(B, C, N, p, p)

#         # expand mask to match patch dims
#         mask_expanded = (
#             mask_patches.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, C, -1, p, p)
#         )

#         # zero out masked patches
#         img_patches_flat = torch.where(
#             mask_expanded, torch.zeros_like(img_patches_flat), img_patches_flat
#         )

#         # reshape back to image
#         img_masked = img_patches_flat.reshape(B, C, n_patches_H, n_patches_W, p, p)
#         img_masked = img_masked.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

#         return img_masked, mask_patches

#     def mask_svg(self, svg_cmds, svg_args, mask_ratio=0.3):
#         # Simple token-level masking
#         B, N, D = svg_args.shape
#         len_keep = int(N * (1 - mask_ratio))
#         noise = torch.rand(B, N, device=svg_args.device)
#         ids_shuffle = torch.argsort(noise, dim=1)
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#         ids_keep = ids_shuffle[:, :len_keep]

#         svg_masked = torch.gather(svg_args, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
#         mask = torch.ones([B, N], device=svg_args.device)
#         mask[:, :len_keep] = 0
#         mask = torch.gather(mask, dim=1, index=ids_restore)
#         mask = mask.bool()

#         return (svg_cmds, svg_masked), mask

#     def show_images(self, original, masked, recon=None, title=""):
#         # all inputs expected in range [0,1]
#         to_pil = T.ToPILImage()

#         plt.figure(figsize=(12, 4))
#         plt.subplot(1, 3 if recon is not None else 2, 1)
#         plt.imshow(to_pil(original.cpu()))
#         plt.title("Original")

#         plt.subplot(1, 3 if recon is not None else 2, 2)
#         plt.imshow(to_pil(masked.cpu()))
#         plt.title("Masked")

#         if recon is not None:
#             plt.subplot(1, 3, 3)
#             plt.imshow(to_pil(recon.cpu()))
#             plt.title("Reconstructed")

#         plt.suptitle(title)
#         plt.axis("off")
#         plt.show()


# #
