from vecssl.models.base import JointModel, TrainStep

from vecssl.models.model import MAEEncoder, MAEDecoder
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

        self.mask_token_cmd = nn.Parameter(torch.randn(self.cfg.d_model))

        # Mask token for arguments (one vector per argument slot)
        self.mask_token_args = nn.Parameter(torch.randn(self.cfg.n_args, self.cfg.args_dim))

        ##what is this step doing??
        # img_dim = self.image_encoder.hidden_dim
        self.logger.info(
            "[bold cyan]Image Encoder Init Successful[/bold cyan]", extra={"markup": True}
        )

        # Init the svg encoder
        self.svg_encoder = MAEEncoder(cfg)
        self.logger.info(
            "[bold cyan]SVG Encoder Init Successful[/bold cyan]", extra={"markup": True}
        )

        # Init the image decoder to reconstruct

        # self.image_decoder = ImageDecoderMAE(cfg)
        self.logger.info(
            "[bold cyan]Image Decoder Init Successful[/bold cyan]", extra={"markup": True}
        )

        # Init the svg decoder to reconstruct the svg
        self.svg_decoder = MAEDecoder(cfg)
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

    def mask_svgs_wrong(
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
        args_masked[mask_positions] = 0

        return commands_masked, args_masked, mask_positions

    def mask_svgs(
        self, commands, args, mask_ratio=0.15, pad_val=-1, mask_token_cmd=None, mask_token_args=None
    ):
        """
        The goal is to first mask out entire commands and their correspodning args.
        """
        # if we are masking out singular commands in the matrix, how are we going to encode the masked command in the commands tensor, and the masked commands + row in the args tensor?, cause the encoder works on taking the whole tensor into the latent space
        # with image patches it made sense cause the encoder would work on each patch and what was being masked was the same unit as the encoder (it was all patches)

        # how do I mask specific rows and cols in the tensor and then somehow add them back for my decoder in the latent space? It doesn't make a lot of sense because it isn't as structurally similar to masking patches
        # If I understand the pipeline correctly it goes:
        # you have your input say 100 arbitrary dims
        # say you mask 20 of these 100 dims
        # you record the positional embedding of these 20 dims
        # you give your 80 unmasked dims to the encoder
        # it encodes each of them into 80 latent dim (say dim z) vectors
        #
        # The you separately encode the 20 dims (how is this done, how are the masked tokens embedded into the same latent dim? Is it using the same encoder?)
        # so you just don't encode them at all
        # instead you create 20 learned mask tokens (trainable embeddings), these are also learned
        # then you add positional embeddings to all of them

        # now you have 20 masked embeddings in latent z dim
        # then you concatenate these 20 masked embeddings with the 80 unmasked dims to get a masked representation of your 100 dim input in latent space
        # your decoder then takes these 100 dims and tries to recreate the original 100 dims image (including the masked tokens)

        # return commands_masked, args_masked, mask_positions

    def measure_euclidean_latent_dim(self, z_svg, masked, logger, debug=True):
        B, G, S, C = z_svg.shape

        # Debug logs
        logger.info(
            f"[bold red]Torch is nan {torch.isnan(z_svg).sum()}[/bold red]", extra={"markup": True}
        )

        # If no groups are masked, treat all as unmasked
        if masked is None:
            masked_idxs = []
            unmasked_idxs = torch.arange(G, device=z_svg.device)
        else:
            masked = torch.as_tensor(masked, device=z_svg.device)
            masked_idxs = masked.tolist()
            all_groups = torch.arange(G, device=z_svg.device)
            mask = torch.ones(G, dtype=torch.bool, device=z_svg.device)
            mask[masked] = False
            unmasked_idxs = all_groups[mask]

        # Extract latent vectors
        if masked_idxs:
            z_masked = z_svg[:, masked_idxs, :, :]  # [B, M, S, C]
            dist_masked = torch.cdist(z_masked.reshape(-1, C), z_masked.reshape(-1, C)).mean()
        else:
            dist_masked = None

        if len(unmasked_idxs) > 0:
            z_unmasked = z_svg[:, unmasked_idxs, :, :]  # [B, G-M, S, C]
            dist_unmasked = torch.cdist(z_unmasked.reshape(-1, C), z_unmasked.reshape(-1, C)).mean()
        else:
            dist_unmasked = None

        # Logging
        if dist_masked is not None:
            logger.info(
                f"[bold red]MaskedMasked mean distance: {dist_masked}[/bold red]",
                extra={"markup": True},
            )
        if dist_unmasked is not None:
            logger.info(
                f"[bold red]UnmaskedUnmasked mean distance: {dist_unmasked}[/bold red]",
                extra={"markup": True},
            )

        return dist_masked, dist_unmasked

    def measure_cosine_similarity(self, z_svgs, masked_groups, logger, debug=True):
        """
        z_svgs: [B, G, S, C] latent embeddings
        masked_groups: list or tensor of masked group indices
        """
        B, G, S, C = z_svgs.shape

        # ensure tensor
        if not torch.is_tensor(masked_groups):
            masked_groups = torch.tensor(masked_groups, device=z_svgs.device)

        # Compute unmasked indices
        all_groups = torch.arange(G, device=z_svgs.device)
        mask = torch.ones(G, dtype=torch.bool, device=z_svgs.device)
        mask[masked_groups] = False
        unmasked_groups = all_groups[mask]

        # Slice latents
        z_masked = z_svgs[:, masked_groups]  # [B, M, S, C]
        z_unmasked = z_svgs[:, unmasked_groups]  # [B, G-M, S, C]

        # Flatten across groups & sequence
        z_masked_flat = z_masked.reshape(-1, C)  # [M*S, C]
        z_unmasked_flat = z_unmasked.reshape(-1, C)  # [(G-M)*S, C]

        # Compute pairwise cosine similarity
        # similarity matrix = (N x C) @ (C x N) → N x N
        sim_masked = torch.matmul(z_masked_flat, z_masked_flat.T)
        sim_unmasked = torch.matmul(z_unmasked_flat, z_unmasked_flat.T)

        # normalize (cosine similarity)
        z_masked_norm = z_masked_flat.norm(dim=1, keepdim=True)
        z_unmasked_norm = z_unmasked_flat.norm(dim=1, keepdim=True)

        cos_masked = sim_masked / (z_masked_norm @ z_masked_norm.T + 1e-8)
        cos_unmasked = sim_unmasked / (z_unmasked_norm @ z_unmasked_norm.T + 1e-8)

        # take mean, remove diagonal entries (self-similarity = 1)
        mean_masked = cos_masked[~torch.eye(cos_masked.shape[0], dtype=bool)].mean()
        mean_unmasked = cos_unmasked[~torch.eye(cos_unmasked.shape[0], dtype=bool)].mean()

        logger.info(
            f"[red]MaskedMasked mean cosine sim:   {mean_masked}[/red]", extra={"markup": True}
        )
        logger.info(
            f"[green]UnmaskedUnmasked mean cosine: {mean_unmasked}[/green]", extra={"markup": True}
        )

        return mean_masked, mean_unmasked

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
        S, G, N = cmds.shape
        # GN = G * N

        # print(
        #     f"This is args {args.shape} this is commands shape {cmds.shape}, this is img: {img.shape}"
        # )

        # make it seq first
        # self.logger.info("[bold red]Making embeddings[/bold red]", extra={"markup": True})
        commands_enc, args_enc = _make_seq_first(cmds, args)

        ##masking time
        # let's generate all embeddings first
        # src = self.svg_encoder.create_embeddings(cmds, args)

        # print(
        #     f"This is args shape after seq first {args_enc.shape} this is commands shape after seq first {commands_enc.shape}, this is img: {img.shape}"
        # )
        # what do we do here first, encode the SVG?
        # self.logger.info("[bold cyan]SVG Encoding...[/bold cyan]", extra={"markup": True})

        # what dim should z_svg be in [1,1,4,256], [4,256]?
        # default returns [1,1,4,256]
        # breakpoint()
        z_svg, masked_groups = self.svg_encoder(commands_enc, args_enc, logger=self.logger)
        # compute cosine similarity to see how the encoder is learning about the space
        # masked_cos_sim, unmasked_cos_sim = self.measure_cosine_similarity(z_svg, masked_groups, logger)
        masked_euclid_dis, unmasked_euclid_dis = self.measure_euclidean_latent_dim(
            z_svg, masked_groups, logger=self.logger
        )

        # self.logger.info(
        #     "[bold cyan]SVG embedded in latent dim z[/bold cyan]", extra={"markup": True}
        # )

        self.logger.info(
            f"[bold cyan]SVG embedded in latent dim z {z_svg.shape}[/bold cyan]",
            extra={"markup": True},
        )

        self.logger.info(
            f"[bold cyan]SVG groups masked {masked_groups}[/bold cyan]",
            extra={"markup": True},
        )

        # let's reconstruct the svg, the logits that Jack was talking about
        recon_cmds_logits, recon_args_logits, *rest = self.svg_decoder(
            z_svg, commands_enc, args_enc, logger=self.logger
        )

        seq_len, num_groups, batch_size, n_args, arg_dim = recon_args_logits.shape

        self.logger.info(
            f"[bold cyan]og cmds shape {commands_enc.shape}[/bold cyan]",
            extra={"markup": True},
        )

        self.logger.info(
            f"[bold cyan]og args  shape {args_enc.shape}[/bold cyan]",
            extra={"markup": True},
        )

        self.logger.info(
            f"[bold cyan]recon_cmds_logits shape {recon_cmds_logits.shape}[/bold cyan]",
            extra={"markup": True},
        )

        self.logger.info(
            f"[bold cyan]recon_args_logits shape {recon_args_logits.shape}[/bold cyan]",
            extra={"markup": True},
        )

        # -------------------------
        # COMMANDS LOSS
        # -------------------------
        # recon_cmds_logits: [seq_len, num_groups, batch, n_commands]
        # commands_enc:      [seq_len_total, num_groups, batch]

        seq_len, num_groups, batch_size, n_commands = recon_cmds_logits.shape

        # loss_cmd = F.cross_entropy(masked_cmd_logits, masked_cmd_targets)

        # Flatten logits to [total_valid_commands, n_commands]
        group_mask = torch.zeros(num_groups, dtype=bool)
        group_mask[masked_groups] = True

        masked_cmd_logits = recon_cmds_logits[:, group_mask, :, :]
        masked_cmd_targets = commands_enc[:seq_len][:, group_mask, :]

        masked_args_logits = recon_args_logits[:, group_mask, :, :, :]
        masked_args_targets = args_enc[:seq_len][:, group_mask, :, :]
        # self.logger.info(
        #     f"[bold blue]masked_cmd_logits  shape {masked_cmd_logits.shape}[/bold blue]",
        #     extra={"markup": True},
        # )
        # self.logger.info(
        #     f"[bold blue]masked_cmd_targets  shape {masked_cmd_targets.shape}[/bold blue]",
        #     extra={"markup": True},
        # )

        # self.logger.info(
        #     f"[bold blue]group mask {group_mask}[/bold blue]",
        #     extra={"markup": True},
        # )
        cmd_logits_flat = masked_cmd_logits.permute(2, 1, 0, 3).reshape(-1, n_commands)
        self.logger.info(
            f"[bold cyan]flat logits  shape {cmd_logits_flat.shape}[/bold cyan]",
            extra={"markup": True},
        )

        # Flatten targets to [total_commands]
        cmd_targets_flat = masked_cmd_targets.permute(2, 1, 0).reshape(-1).long()
        self.logger.info(
            f"[bold cyan]flat target  shape {cmd_targets_flat.shape}[/bold cyan]",
            extra={"markup": True},
        )
        loss_cmd = F.cross_entropy(cmd_logits_flat, cmd_targets_flat)

        ##THIS WAS THE SOURCE OF SHAPE ERRORS, HOW BEST TO DEAL WITH PAD_VAL when working with reconstructions?
        ##Currently only compute loss on unpadded values

        # Compute CE

        # cmd_targets_safe = cmd_targets_flat.clone()
        # cmd_targets_safe[cmd_targets_safe == pad_val] = IGNORE

        # Compute cross-entropy with ignore index
        # loss_cmd = F.cross_entropy(cmd_logits_flat, cmd_targets_safe, ignore_index=IGNORE)

        # interestingly commands doesnt have any -1 values?

        # -------------------------
        # ARGS LOSS
        # -------------------------

        IGNORE = -100
        pad_val = -1
        seq_len, num_groups, batch_size, n_args, arg_dim = recon_args_logits.shape

        # Flatten logits to [total_args, arg_dim]
        args_logits_flat = masked_args_logits.permute(2, 1, 0, 3, 4).reshape(-1, arg_dim)

        # Flatten targets to [total_args]
        args_targets_flat = masked_args_targets.permute(2, 1, 0, 3).reshape(-1).long()

        ##THIS WAS THE SOURCE OF SHAPE ERRORS, HOW BEST TO DEAL WITH PAD_VAL when working with reconstructions?
        ##Currently only compute loss on unpadded values

        args_targets_safe = args_targets_flat.clone()
        args_targets_safe[args_targets_safe == pad_val] = IGNORE

        loss_args = F.cross_entropy(args_logits_flat, args_targets_safe, ignore_index=IGNORE)

        # self.logger.info(
        #     f"[bold green]args target shape before loss{args_targets_safe.shape}[/bold green]",
        #     extra={"markup": True},
        # )

        # self.logger.info(
        #     f"[bold green]recon_args_logits shape {recon_args_logits.shape}[/bold green]",
        #     extra={"markup": True},
        # )

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
