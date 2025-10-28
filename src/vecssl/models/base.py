from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class TrainStep:
    loss: torch.Tensor
    logs: Optional[dict[str, float]] = None
    extras: Optional[dict[str, Any]] = None


class JointModel(nn.Module):
    def forward(self, batch: dict[str, Any]) -> TrainStep:
        raise NotImplementedError

    # For eval/inference, optional for now
    def encode_joint(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Optional. Returns e.g. {"img": z_img [B,d], "svg": z_svg [B,d]} (L2-normalized).
        """
        raise NotImplementedError

    # Editing
    def to_z_edit(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Optional. Returns z_edit [B, d_edit] from inputs (e.g., svg)."""
        raise NotImplementedError

    def decode_svg(
        self, z_edit: torch.Tensor, N_max: int, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Optional. Decode z_edit to SVG fields/geometry."""
        raise NotImplementedError
