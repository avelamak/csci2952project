from vecssl.models.base import JointModel
from vecssl.models.model import Encoder, Decoder, ImageDecoderMAE
from torchvision.models import vit_b_16
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F


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

    def forward(self, batch):
        """
        Forward pass for multi-modal (image + SVG) masked autoencoder.
        Logs shapes and sanity-check info using rich logger.
        """

        self.logger.info("[bold cyan]Starting forward pass...[/bold cyan]", extra={"markup": True})
        device = next(self.image_encoder.parameters()).device
        # --- (1) Extract batch data ---
        img = batch["image"].to(device)  # (B, 3, H, W)
        # svg_cmds = batch["commands"].to(device)  # (B, N_groups, seq_len)
        # svg_args = batch["args"].to(device)

        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img_masked, img_mask = self.mask_image(img)

        self.logger.info(
            f"[bold yellow]Input image shape:[/bold yellow] {tuple(img.shape)}",
            extra={"markup": True},
        )

        # --- (2) Apply masking ---
        # img_masked, img_mask = self.mask_image(img)
        # img_masked = F.interpolate(img_masked, size=(224, 224), mode='bilinear', align_corners=False)

        self.logger.info(
            f"[bold yellow]Masked image shape:[/bold yellow] {tuple(img_masked.shape)}",
            extra={"markup": True},
        )
        self.logger.info(
            f"[bold yellow]Mask shape:[/bold yellow] {tuple(img_mask.shape)}",
            extra={"markup": True},
        )

        # --- (3) Encode visible tokens ---
        img_latents = self.image_encoder(img_masked)
        self.logger.info(
            f"[bold green]Image latents shape:[/bold green] {tuple(img_latents.shape)}",
            extra={"markup": True},
        )

        # For now, skip SVG encoder & fusion — use only image latents
        if img_latents.shape[1] != img_mask.shape[1]:
            img_latents = img_latents[:, 1:, :]
        fused = img_latents

        # --- (4) Decode / reconstruct ---
        img_recon = self.image_decoder(fused, img_mask)
        self.logger.info(
            f"[bold magenta]Reconstructed image shape:[/bold magenta] {tuple(img_recon.shape)}",
            extra={"markup": True},
        )

        # --- (5) Visualize (sanity check) ---
        B = 0
        self.logger.info(
            "[bold blue]Visualizing reconstruction for sanity check...[/bold blue]",
            extra={"markup": True},
        )

        self.show_images(
            original=img[B],
            masked=img_masked[B],
            recon=img_recon[B],
            title="Sanity check reconstruction",
        )

        self.logger.info(
            "[bold cyan]Forward pass completed successfully.[/bold cyan]", extra={"markup": True}
        )
        return 0

    def encode_joint(self, batch):
        pass

    def mask_image(self, img):
        """
        img: [B, C, H, W]
        Returns:
            img_masked: [B, C, H, W] masked image
            mask_patches: [B, N] boolean patch mask
        """
        B, C, H, W = img.shape
        p = 16  # patch size
        n_patches_H = H // p
        n_patches_W = W // p
        N = n_patches_H * n_patches_W

        # create random patch mask
        mask_patches = torch.zeros(B, N, dtype=torch.bool, device=img.device)
        for b in range(B):
            n_mask = int(0.25 * N)
            idx = torch.randperm(N)[:n_mask]
            mask_patches[b, idx] = True

        # reshape img into patches
        img_patches = img.unfold(2, p, p).unfold(3, p, p)  # [B,C,H//p,W//p,p,p]
        img_patches_flat = img_patches.reshape(B, C, N, p, p)

        # expand mask to match patch dims
        mask_expanded = (
            mask_patches.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, C, -1, p, p)
        )

        # zero out masked patches
        img_patches_flat = torch.where(
            mask_expanded, torch.zeros_like(img_patches_flat), img_patches_flat
        )

        # reshape back to image
        img_masked = img_patches_flat.reshape(B, C, n_patches_H, n_patches_W, p, p)
        img_masked = img_masked.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

        return img_masked, mask_patches

    def mask_svg(self, svg_cmds, svg_args, mask_ratio=0.3):
        # Simple token-level masking
        B, N, D = svg_args.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=svg_args.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        svg_masked = torch.gather(svg_args, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones([B, N], device=svg_args.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.bool()

        return (svg_cmds, svg_masked), mask

    def show_images(self, original, masked, recon=None, title=""):
        # all inputs expected in range [0,1]
        to_pil = T.ToPILImage()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3 if recon is not None else 2, 1)
        plt.imshow(to_pil(original.cpu()))
        plt.title("Original")

        plt.subplot(1, 3 if recon is not None else 2, 2)
        plt.imshow(to_pil(masked.cpu()))
        plt.title("Masked")

        if recon is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(to_pil(recon.cpu()))
            plt.title("Reconstructed")

        plt.suptitle(title)
        plt.axis("off")
        plt.show()


#
