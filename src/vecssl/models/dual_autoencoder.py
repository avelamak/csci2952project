"""
Dual Autoencoder model for joint image-SVG representation learning.

This model consists of:
- Image encoder/decoder: CNN-based autoencoder for images
- SVG encoder/decoder: Simple embedding-based autoencoder for SVG
- Shared latent space: Both modalities encode to the same dimension
"""

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from vecssl.models.base import JointModel, TrainStep


class ImageEncoder(nn.Module):
    """CNN encoder for images (128x128 RGB -> latent_dim)"""

    def __init__(self, latent_dim: int = 128, input_channels: int = 3):
        super().__init__()
        # Input: (B, 3, 128, 128)
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            # 8x8 -> 4x4
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Flatten: 4x4x512 = 8192
            nn.Flatten(),
            # Project to latent_dim
            nn.Linear(512 * 4 * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ImageDecoder(nn.Module):
    """CNN decoder for images (latent_dim -> 128x128 RGB)"""

    def __init__(self, latent_dim: int = 128, output_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        # Project from latent_dim to 4x4x512
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1], will be scaled to [0, 1] if needed
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        # Project to spatial features
        x = self.fc(z)
        x = x.view(B, 512, 4, 4)
        return self.decoder(x)


class SVGEncoder(nn.Module):
    """Simple encoder for SVG features.
    
    For now, we assume SVG is represented as a feature vector.
    In practice, this could be a transformer or sequence encoder.
    """

    def __init__(self, svg_feature_dim: int = 512, latent_dim: int = 128):
        super().__init__()
        # Simple MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(svg_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SVGDecoder(nn.Module):
    """Simple decoder for SVG features.
    
    Decodes from latent space back to SVG feature space.
    """

    def __init__(self, latent_dim: int = 128, svg_feature_dim: int = 512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, svg_feature_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class DualAutoencoder(JointModel):
    """
    Dual Autoencoder model for joint image-SVG representation learning.
    
    The model learns to:
    1. Encode images to latent space
    2. Encode SVG to latent space (same dimension)
    3. Decode latent codes back to images/SVG
    4. Align representations in the shared latent space
    """

    def __init__(
        self,
        latent_dim: int = 128,
        img_channels: int = 3,
        svg_feature_dim: int = 512,
        recon_weight: float = 1.0,
        align_weight: float = 0.1,
    ):
        """
        Args:
            latent_dim: Dimension of shared latent space
            img_channels: Number of image channels (3 for RGB)
            svg_feature_dim: Dimension of SVG feature vector (if SVG is pre-processed)
            recon_weight: Weight for reconstruction loss
            align_weight: Weight for alignment loss (contrastive)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.svg_feature_dim = svg_feature_dim
        self.recon_weight = recon_weight
        self.align_weight = align_weight

        # Image encoder/decoder
        self.img_encoder = ImageEncoder(latent_dim=latent_dim, input_channels=img_channels)
        self.img_decoder = ImageDecoder(latent_dim=latent_dim, output_channels=img_channels)

        # SVG encoder/decoder
        self.svg_encoder = SVGEncoder(svg_feature_dim=svg_feature_dim, latent_dim=latent_dim)
        self.svg_decoder = SVGDecoder(latent_dim=latent_dim, svg_feature_dim=svg_feature_dim)

    def forward(self, batch: dict[str, Any]) -> TrainStep:
        """
        Forward pass computing reconstruction and alignment losses.
        
        Expected batch keys:
            - 'image': torch.Tensor of shape (B, C, H, W) - images
            - 'svg': torch.Tensor of shape (B, svg_feature_dim) - SVG features
                   OR dict with 'features' key
        """
        # Extract inputs
        img = batch.get("image")
        svg_data = batch.get("svg") or batch.get("svg_features")

        losses = []
        logs = {}

        z_img = None
        z_svg = None
        svg_features = None

        # Image reconstruction
        if img is not None:
            # Normalize image to [-1, 1] if it's in [0, 1]
            if img.max() <= 1.0:
                img = img * 2.0 - 1.0

            # Encode and decode image
            z_img = self.img_encoder(img)
            img_recon = self.img_decoder(z_img)

            # Reconstruction loss (L2)
            img_loss = F.mse_loss(img_recon, img)
            losses.append(self.recon_weight * img_loss)
            logs["img_recon_loss"] = float(img_loss.detach())

        # SVG reconstruction
        if svg_data is not None:
            # Handle SVG input: could be tensor or dict
            if isinstance(svg_data, dict):
                svg_features = svg_data.get("features")
            else:
                svg_features = svg_data

            # If SVG is not a tensor, create a dummy feature vector
            # In practice, you'd want proper SVG processing here
            if svg_features is None or not isinstance(svg_features, torch.Tensor):
                # Create dummy features from SVG string or use zeros
                B = img.shape[0] if img is not None else 1
                device = img.device if img is not None else torch.device("cpu")
                svg_features = torch.zeros(
                    B, self.svg_feature_dim, device=device
                )

            # Encode and decode SVG
            z_svg = self.svg_encoder(svg_features)
            svg_recon = self.svg_decoder(z_svg)

            # Reconstruction loss
            svg_loss = F.mse_loss(svg_recon, svg_features)
            losses.append(self.recon_weight * svg_loss)
            logs["svg_recon_loss"] = float(svg_loss.detach())

        # Alignment loss (contrastive): align image and SVG in latent space
        if z_img is not None and z_svg is not None:
            # L2 normalize for contrastive learning
            z_img_norm = F.normalize(z_img, p=2, dim=1)
            z_svg_norm = F.normalize(z_svg, p=2, dim=1)

            # Contrastive loss: positive pairs should be similar
            # Simple cosine similarity loss (maximize similarity)
            align_loss = -torch.mean(
                torch.sum(z_img_norm * z_svg_norm, dim=1)
            )  # Negative cosine similarity
            losses.append(self.align_weight * align_loss)
            logs["align_loss"] = float(align_loss.detach())

        # Total loss
        total_loss = sum(losses)
        logs["total_loss"] = float(total_loss.detach())

        return TrainStep(loss=total_loss, logs=logs)

    def encode_joint(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Encode both image and SVG to shared latent space.
        Returns L2-normalized encodings.
        """
        z_img = None
        z_svg = None

        img = batch.get("image")
        if img is not None:
            # Normalize to [-1, 1] if needed
            if img.max() <= 1.0:
                img = img * 2.0 - 1.0
            z_img = self.img_encoder(img)
            z_img = F.normalize(z_img, p=2, dim=1)

        svg_data = batch.get("svg") or batch.get("svg_features")
        if svg_data is not None:
            if isinstance(svg_data, dict):
                svg_features = svg_data.get("features")
            else:
                svg_features = svg_data

            if svg_features is not None and isinstance(svg_features, torch.Tensor):
                z_svg = self.svg_encoder(svg_features)
                z_svg = F.normalize(z_svg, p=2, dim=1)

        result = {}
        if z_img is not None:
            result["img"] = z_img
        if z_svg is not None:
            result["svg"] = z_svg

        return result

    def to_z_edit(self, batch: dict[str, Any]) -> torch.Tensor:
        """Encode SVG to edit space (same as latent space for now)."""
        svg_data = batch.get("svg") or batch.get("svg_features")
        if isinstance(svg_data, dict):
            svg_features = svg_data.get("features")
        else:
            svg_features = svg_data

        if svg_features is None or not isinstance(svg_features, torch.Tensor):
            B = batch.get("batch_size", 1)
            device = next(self.parameters()).device
            svg_features = torch.zeros(
                B, self.svg_feature_dim, device=device
            )

        z_edit = self.svg_encoder(svg_features)
        return z_edit

    def decode_svg(
        self, z_edit: torch.Tensor, N_max: int, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Decode z_edit to SVG features.
        
        Args:
            z_edit: Edit latent code (B, latent_dim)
            N_max: Maximum number of elements (not used in simple version)
            mask: Mask for which elements to decode (not used in simple version)
        
        Returns:
            Dictionary with 'features' key containing decoded SVG features
        """
        svg_features = self.svg_decoder(z_edit)
        return {"features": svg_features}

