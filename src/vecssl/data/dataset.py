"""
Data utils for vecssl

Currently we have:
 - SVGX-Core-250k

"""

from typing import Dict, Any, List, Tuple, Optional
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import io
import torchvision.transforms as transforms


from dataclasses import dataclass
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
import re
import numpy as np


"""SVG matrix representation following SVGFusion §3.1.

Converts SVG XML into a structured matrix with:
- Element/command indices (discrete)
- Normalized geometry parameters (continuous)
- Colors, opacity, stroke (continuous)
- Optional affine transforms

One row per primitive/command, preserving input order.
"""

# SVG element type vocabulary
ELEMENT_TYPES = [
    "<PAD>",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "g",
    "use",
    "text",
    "defs",
    "clipPath",
    "mask",
    "linearGradient",
    "radialGradient",
    "pattern",
    "image",
    "symbol",
    "marker",
    "<UNK>",
]

# Path command vocabulary
PATH_COMMANDS = ["<PAD>", "M", "L", "H", "V", "C", "S", "Q", "T", "A", "Z", "<UNK>"]


@dataclass
class SvgMatSpec:
    """Specification for SVG matrix representation."""

    # Vocabulary sizes for discrete fields
    n_elements: int = len(ELEMENT_TYPES)
    n_commands: int = len(PATH_COMMANDS)

    # Geometry columns (normalized to [0, 1] or [-1, 1])
    geom_cols: List[str] = None

    # Style columns (colors, stroke, opacity)
    style_cols: List[str] = None

    # Whether to include affine transform parameters
    include_affine: bool = True

    # Maximum number of rows (K in the paper)
    max_rows: int = 256

    def __post_init__(self):
        if self.geom_cols is None:
            # Geometry: x, y, x1, y1, x2, y2, radius
            self.geom_cols = ["x", "y", "x1", "y1", "x2", "y2", "r"]

        if self.style_cols is None:
            # Style: fill RGBA, stroke width, visible flag
            self.style_cols = ["fill_r", "fill_g", "fill_b", "fill_a", "stroke_w", "visible"]


class SvgMatBuilder(nn.Module):
    """Builds matrix representation from SVG and provides embedding."""

    def __init__(self, spec: SvgMatSpec):
        super().__init__()
        self.spec = spec

        # Discrete column indices
        self.col_element = 0
        self.col_command = 1

        # Continuous column start indices
        self.col_geom_start = 2
        self.col_style_start = 2 + len(spec.geom_cols)

        if spec.include_affine:
            # Affine: [a, b, c, d, e, f] for matrix(a,b,c,d,e,f)
            self.col_affine_start = self.col_style_start + len(spec.style_cols)
            self.n_cols_total = self.col_affine_start + 6
        else:
            self.n_cols_total = self.col_style_start + len(spec.style_cols)

        # Learned embeddings for discrete fields
        # self.element_embed = nn.Embedding(spec.n_elements, 64)
        # self.command_embed = nn.Embedding(spec.n_commands, 64)

        # Linear projection for continuous fields
        # n_continuous = self.n_cols_total - 2  # exclude element and command indices
        # self.continuous_proj = nn.Linear(n_continuous, 128)

        # Output embedding dimension (used by encoders)
        # self.d_embed = 64 + 64 + 128  # element + command + continuous

    def parse_xml_to_matrix(
        self, svg_text: str, viewport: Tuple[int, int] = (512, 512)
    ) -> Dict[str, torch.Tensor]:
        """Parse SVG XML and convert to matrix representation.

        Args:
            svg_text: SVG XML string
            viewport: (width, height) for normalization

        Returns:
            Dictionary with:
                - 'svg_mat': [N, D] tensor with N rows, D columns
                - 'svg_mat_mask': [N] boolean mask (True for valid rows)
        """
        try:
            root = ET.fromstring(svg_text)
        except ET.ParseError:
            # Return empty matrix for invalid SVG
            return self._empty_matrix()

        # Parse all elements and build rows
        rows = []
        self._parse_element(root, rows, viewport)

        if not rows:
            return self._empty_matrix()

        # Convert to tensor and pad/truncate
        matrix = np.array(rows, dtype=np.float32)
        n_rows = min(len(matrix), self.spec.max_rows)

        # Create padded matrix
        svg_mat = np.zeros((self.spec.max_rows, self.n_cols_total), dtype=np.float32)
        svg_mat[:n_rows] = matrix[:n_rows]

        # Create mask (True for valid rows)
        svg_mat_mask = np.zeros(self.spec.max_rows, dtype=bool)
        svg_mat_mask[:n_rows] = True

        return {
            "svg_mat": torch.from_numpy(svg_mat),
            "svg_mat_mask": torch.from_numpy(svg_mat_mask),
        }

    def _empty_matrix(self) -> Dict[str, torch.Tensor]:
        """Return empty matrix (all padding)."""
        return {
            "svg_mat": torch.zeros((self.spec.max_rows, self.n_cols_total)),
            "svg_mat_mask": torch.zeros(self.spec.max_rows, dtype=torch.bool),
        }

    def _parse_element(
        self,
        elem: ET.Element,
        rows: List[List[float]],
        viewport: Tuple[int, int],
        inherited_style: Optional[Dict] = None,
    ):
        """Recursively parse SVG element and add rows to matrix."""
        if inherited_style is None:
            inherited_style = {}

        # Get element type index
        tag = elem.tag.split("}")[-1]  # Remove namespace
        elem_idx = (
            ELEMENT_TYPES.index(tag) if tag in ELEMENT_TYPES else ELEMENT_TYPES.index("<UNK>")
        )

        # Merge inherited and local styles
        style = inherited_style.copy()
        style.update(self._parse_style(elem))

        # Handle different element types
        if tag == "path":
            self._parse_path(elem, rows, elem_idx, style, viewport)
        elif tag in ["rect", "circle", "ellipse", "line"]:
            self._parse_shape(elem, rows, elem_idx, tag, style, viewport)
        elif tag == "g":
            # Group: recurse to children
            for child in elem:
                self._parse_element(child, rows, viewport, style)
        else:
            # Other elements: create single row
            row = self._create_row(elem_idx, 0, [0] * len(self.spec.geom_cols), style, viewport)
            rows.append(row)

        # Recursively handle children (except for 'g' already handled)
        if tag != "g":
            for child in elem:
                self._parse_element(child, rows, viewport, style)

    def _parse_path(
        self,
        elem: ET.Element,
        rows: List[List[float]],
        elem_idx: int,
        style: Dict,
        viewport: Tuple[int, int],
    ):
        """Parse path element and create row for each command."""
        d = elem.get("d", "")
        if not d:
            return

        # Simple path command parsing (M, L, C, Z)
        # This is a simplified parser - full implementation would handle all commands
        commands = re.findall(r"[MLHVCSQTAZ][^MLHVCSQTAZ]*", d, re.IGNORECASE)

        for cmd_str in commands[: self.spec.max_rows]:  # Limit per path
            cmd_char = cmd_str[0].upper()
            cmd_idx = (
                PATH_COMMANDS.index(cmd_char)
                if cmd_char in PATH_COMMANDS
                else PATH_COMMANDS.index("<UNK>")
            )

            # Extract numeric parameters
            params = re.findall(r"-?\d+\.?\d*", cmd_str)
            params = [float(p) for p in params]

            # Normalize parameters to geometry columns
            geom = self._normalize_params(params, viewport)

            row = self._create_row(elem_idx, cmd_idx, geom, style, viewport)
            rows.append(row)

    def _parse_shape(
        self,
        elem: ET.Element,
        rows: List[List[float]],
        elem_idx: int,
        shape_type: str,
        style: Dict,
        viewport: Tuple[int, int],
    ):
        """Parse basic shape element."""
        geom = [0.0] * len(self.spec.geom_cols)

        # Extract shape-specific parameters
        if shape_type == "rect":
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
            w = float(elem.get("width", 0))
            h = float(elem.get("height", 0))
            geom[0] = x / viewport[0]  # x
            geom[1] = y / viewport[1]  # y
            geom[2] = (x + w) / viewport[0]  # x1
            geom[3] = (y + h) / viewport[1]  # y1
        elif shape_type == "circle":
            cx = float(elem.get("cx", 0))
            cy = float(elem.get("cy", 0))
            r = float(elem.get("r", 0))
            geom[0] = cx / viewport[0]
            geom[1] = cy / viewport[1]
            geom[6] = r / max(viewport)
        elif shape_type == "ellipse":
            cx = float(elem.get("cx", 0))
            cy = float(elem.get("cy", 0))
            rx = float(elem.get("rx", 0))
            ry = float(elem.get("ry", 0))
            geom[0] = cx / viewport[0]
            geom[1] = cy / viewport[1]
            geom[2] = rx / viewport[0]
            geom[3] = ry / viewport[1]

        row = self._create_row(elem_idx, 0, geom, style, viewport)
        rows.append(row)

    def _parse_style(self, elem: ET.Element) -> Dict:
        """Extract style attributes from element."""
        style = {}

        # Direct attributes
        for attr in ["fill", "stroke", "stroke-width", "opacity", "fill-opacity"]:
            val = elem.get(attr)
            if val:
                style[attr] = val

        # Style attribute
        style_attr = elem.get("style", "")
        if style_attr:
            for item in style_attr.split(";"):
                if ":" in item:
                    key, val = item.split(":", 1)
                    style[key.strip()] = val.strip()

        return style

    def _normalize_params(self, params: List[float], viewport: Tuple[int, int]) -> List[float]:
        """Normalize path parameters to [0, 1] based on viewport."""
        geom = [0.0] * len(self.spec.geom_cols)

        for i, p in enumerate(params[: len(geom)]):
            # Alternate x/y normalization
            if i % 2 == 0:
                geom[i] = p / viewport[0]  # x-coordinate
            else:
                geom[i] = p / viewport[1]  # y-coordinate

        # Clip to reasonable range
        geom = [max(-2.0, min(2.0, g)) for g in geom]

        return geom

    def _create_row(
        self, elem_idx: int, cmd_idx: int, geom: List[float], style: Dict, viewport: Tuple[int, int]
    ) -> List[float]:
        """Create a single matrix row."""
        row = [float(elem_idx), float(cmd_idx)]

        # Add geometry
        row.extend(geom[: len(self.spec.geom_cols)])

        # Add style
        fill_color = self._parse_color(style.get("fill", "#000000"))
        stroke_w = float(style.get("stroke-width", "1").replace("px", ""))
        stroke_w = stroke_w / max(viewport)  # Normalize stroke

        row.extend(
            [
                fill_color[0] / 255.0,  # fill_r
                fill_color[1] / 255.0,  # fill_g
                fill_color[2] / 255.0,  # fill_b
                fill_color[3],  # fill_a (already 0-1)
                stroke_w,  # stroke_w
                1.0,  # visible
            ]
        )

        # Add affine if needed
        if self.spec.include_affine:
            # Default identity transform
            row.extend([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        return row

    def _parse_color(self, color_str: str) -> Tuple[float, float, float, float]:
        """Parse color string to RGBA."""
        if color_str == "none":
            return (0, 0, 0, 0)

        # Handle hex colors
        if color_str.startswith("#"):
            color_str = color_str[1:]
            if len(color_str) == 3:
                color_str = "".join([c * 2 for c in color_str])
            if len(color_str) == 6:
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                return (r, g, b, 1.0)

        # Default black
        return (0, 0, 0, 1.0)

    # def embed(self, svg_mat: torch.Tensor) -> torch.Tensor:
    #     """Convert matrix [B, N, D] to dense embeddings [B, N, d_embed].

    #     Args:
    #         svg_mat: [B, N, D] tensor with discrete indices and continuous values

    #     Returns:
    #         [B, N, d_embed] embedded representation
    #     """
    #     B, N, D = svg_mat.shape

    #     # Extract discrete indices (as long)
    #     elem_idx = svg_mat[:, :, self.col_element].long()  # [B, N]
    #     cmd_idx = svg_mat[:, :, self.col_command].long()   # [B, N]

    #     # Clip indices to vocabulary
    #     elem_idx = elem_idx.clamp(0, self.spec.n_elements - 1)
    #     cmd_idx = cmd_idx.clamp(0, self.spec.n_commands - 1)

    #     # Embed discrete fields
    #     elem_emb = self.element_embed(elem_idx)  # [B, N, 64]
    #     cmd_emb = self.command_embed(cmd_idx)    # [B, N, 64]

    #     # Project continuous fields
    #     continuous = svg_mat[:, :, 2:]  # [B, N, D-2]
    #     cont_emb = self.continuous_proj(continuous)  # [B, N, 128]

    #     # Concatenate all embeddings
    #     full_emb = torch.cat([elem_emb, cmd_emb, cont_emb], dim=-1)  # [B, N, d_embed]

    #     return full_emb

    def forward(
        self, svg_text: str, viewport: Tuple[int, int] = (512, 512)
    ) -> Dict[str, torch.Tensor]:
        """End-to-end: parse and embed a single SVG.

        Args:
            svg_text: SVG XML string
            viewport: (width, height) for normalization

        Returns:
            Dictionary with 'svg_mat', 'svg_mat_mask', and 'svg_emb'
        """
        parsed = self.parse_xml_to_matrix(svg_text, viewport)

        # Add batch dimension and embed
        svg_mat = parsed["svg_mat"].unsqueeze(0)  # [1, N, D]
        # svg_emb = self.embed(svg_mat)  # [1, N, d_embed]

        return {
            "svg_mat": svg_mat.squeeze(0),
            "svg_mat_mask": parsed["svg_mat_mask"],
            # 'svg_emb': svg_emb.squeeze(0)
        }


class SVGXDataset(Dataset):
    """
    Dataset wrapper for SVGX-Core-250k.
    This dataset contains ~250k SVG vector graphics with:
    - SVG paths and code
    - BLIP captions
    - Rendered images (128x128)
    - Descriptions
    - Source information
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        enable_svg_matrix: bool = True,
        svg_spec: Optional[SvgMatSpec] = None,
    ) -> None:
        """
        Initialize the SVGX dataset.

        Args:
            split: Dataset split to use (e.g., "train", "test")
            cache_dir: Directory to cache the downloaded dataset
            transform: Optional transform to apply to images (if None, uses ToTensor())
        """
        self.dataset: datasets.Dataset = load_dataset(  # type: ignore
            "xingxm/SVGX-Core-250k",
            split=split,
            cache_dir=cache_dir,
        )

        # Default transform converts PIL Image to tensor
        self.transform = transform if transform is not None else transforms.ToTensor()

        # SVG -> matrix builder
        self.enable_svg_matrix = enable_svg_matrix
        if self.enable_svg_matrix:
            spec = svg_spec or SvgMatSpec()
            self.svg_builder = SvgMatBuilder(spec)
        else:
            self.svg_builder = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]

        # Convert image bytes to PIL Image if needed, then to tensor
        if "image" in sample:
            if isinstance(sample["image"], (bytes, io.BytesIO)):
                sample["image"] = Image.open(io.BytesIO(sample["image"]))

            # Convert PIL Image to tensor
            if isinstance(sample["image"], Image.Image):
                sample["image"] = self.transform(sample["image"])

        if self.enable_svg_matrix and self.svg_builder:
            svg_text = sample.get("svg_code", "")
            if svg_text:
                svg_data = self.svg_builder(svg_text)
                sample["svg_mat"] = svg_data["svg_mat"]
                sample["svg_mat_mask"] = svg_data["svg_mat_mask"]
        return sample

    def get_sample_info(self, idx: int) -> dict[str, Any]:
        sample = self[idx]
        info = {
            "index": idx,
            "has_image": "image" in sample,
            "has_svg": "svg" in sample or "svg_code" in sample,
            "has_caption": "caption" in sample or "blip_caption" in sample,
            "keys": list(sample.keys()),
        }
        return info
