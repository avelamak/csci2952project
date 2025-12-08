"""
Common utilities for evaluation scripts.

Provides:
- custom_collate: Standard collate for SVGXDataset with label support
- load_encoder: Load frozen JEPA or Contrastive encoder
- compute_embeddings: Batch encode dataset to latent vectors
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset
from vecssl.models.config import (
    ContrastiveConfig,
    JepaConfig,
    SVGMAEConfig,
    MultiMAEConfig,
    _DefaultConfig,
)
from vecssl.models.contrastive import ContrastiveModel
from vecssl.models.jepa import Jepa
from vecssl.models.svgmae import SVGMAE
from vecssl.models.multimae import MultiMAE

logger = logging.getLogger(__name__)


def custom_collate(batch):
    """
    Standard collate for SVGXDataset.

    Stacks tensors, keeps lists for non-tensor fields.
    Handles optional fields like dino_embedding, label, family_label.
    """
    collated = {
        "commands": torch.stack([item["commands"] for item in batch]),
        "args": torch.stack([item["args"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "tensors": [item["tensors"] for item in batch],
        "uuid": [item["uuid"] for item in batch],
        "name": [item["name"] for item in batch],
        "source": [item["source"] for item in batch],
        "glyph_label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "family_label": torch.tensor([item["family_label"] for item in batch], dtype=torch.long),
    }

    # Handle optional DINO embeddings (precomputed CLS embedding)
    if "dino_embedding" in batch[0]:
        collated["dino_embedding"] = torch.stack([item["dino_embedding"] for item in batch])

    # Handle optional DINO patches (precomputed patch embeddings for MultiMAE)
    if "dino_patches" in batch[0]:
        collated["dino_patches"] = torch.stack([item["dino_patches"] for item in batch])

    # Handle optional label fields
    if "label" in batch[0]:
        collated["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    if "family_label" in batch[0]:
        collated["family_label"] = torch.tensor(
            [item["family_label"] for item in batch], dtype=torch.long
        )

    return collated


def load_encoder(
    checkpoint_path: Path,
    encoder_type: str,
) -> tuple:
    """
    Load frozen encoder model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        encoder_type: "jepa" or "contrastive" or "autoencoder"

    Returns:
        tuple: (model, config)
    """
    logger.info(f"Loading {encoder_type} encoder from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg_obj = ckpt.get("cfg")

    if encoder_type == "jepa":
        if isinstance(cfg_obj, JepaConfig):
            cfg = cfg_obj
        elif isinstance(cfg_obj, dict):
            cfg = JepaConfig()
            for k, v in cfg_obj.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        else:
            logger.warning("No config in checkpoint, using default JepaConfig")
            cfg = JepaConfig()
        model = Jepa(cfg)

    elif encoder_type == "contrastive":
        if isinstance(cfg_obj, ContrastiveConfig):
            cfg = cfg_obj
        elif isinstance(cfg_obj, dict):
            cfg = ContrastiveConfig()
            for k, v in cfg_obj.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        else:
            logger.warning("No config in checkpoint, using default ContrastiveConfig")
            cfg = ContrastiveConfig()
        model = ContrastiveModel(cfg)

    elif encoder_type == "autoencoder":
        from test_svg_autoencoder import SimpleSVGAutoencoder

        if isinstance(cfg_obj, _DefaultConfig):
            cfg = cfg_obj
        elif isinstance(cfg_obj, dict):
            cfg = _DefaultConfig()
            for k, v in cfg_obj.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        else:
            logger.warning("No config in checkpoint, using default _DefaultConfig")
            cfg = _DefaultConfig()
            cfg.encode_stages = 2
            cfg.decode_stages = 2
            cfg.use_vae = True

        model = SimpleSVGAutoencoder(cfg)

    elif encoder_type == "svgmae":
        if isinstance(cfg_obj, SVGMAEConfig):
            cfg = cfg_obj
        elif isinstance(cfg_obj, dict):
            cfg = SVGMAEConfig()
            for k, v in cfg_obj.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        else:
            logger.warning("No config in checkpoint, using default SVGMAEConfig")
            cfg = SVGMAEConfig()
        model = SVGMAE(cfg)

    elif encoder_type == "multimae":
        if isinstance(cfg_obj, MultiMAEConfig):
            cfg = cfg_obj
        elif isinstance(cfg_obj, dict):
            cfg = MultiMAEConfig()
            for k, v in cfg_obj.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        else:
            logger.warning("No config in checkpoint, using default MultiMAEConfig")
            cfg = MultiMAEConfig()
        model = MultiMAE(cfg)

    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. Must be 'jepa', 'contrastive', 'autoencoder', 'svgmae', or 'multimae'"
        )

    model.load_state_dict(ckpt["model"])
    model.eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    epoch = ckpt.get("epoch", "unknown")
    logger.info(f"Loaded {encoder_type} from epoch {epoch}")

    return model, cfg


@torch.no_grad()
def compute_embeddings(
    model,
    loader: DataLoader,
    device: torch.device,
    modality: str = "svg",
) -> tuple[torch.Tensor, list, list]:
    """
    Compute embeddings for entire dataset.

    Args:
        model: Encoder model with encode_joint() method
        loader: DataLoader
        device: torch device
        modality: "svg" or "img" - which embedding to return

    Returns:
        tuple: (embeddings [N, d], labels [N], uuids [N])
    """
    embeddings = []
    labels = []
    uuids = []

    model.eval()
    for batch in loader:
        # Move tensors to device
        batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        joint = model.encode_joint(batch_device)
        z = joint[modality]  # [B, d_joint]
        z = F.normalize(z, dim=-1)

        embeddings.append(z.cpu())
        labels.extend(batch["label"].tolist())
        uuids.extend(batch["uuid"])

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, labels, uuids


def create_eval_dataloader(
    svg_dir: str,
    img_dir: str,
    meta_filepath: str,
    max_num_groups: int = 8,
    max_seq_len: int = 40,
    batch_size: int = 64,
    num_workers: int = 4,
    split: str = "test",
    seed: int = 42,
    shuffle: bool = False,
    stratify_by: str | None = None,
    min_class_count: int = 2,
) -> DataLoader:
    """
    Create a dataloader for evaluation.

    Args:
        svg_dir: Path to SVG directory
        img_dir: Path to image directory
        meta_filepath: Path to metadata CSV
        max_num_groups: Max number of path groups
        max_seq_len: Max sequence length per group
        batch_size: Batch size
        num_workers: Number of dataloader workers
        split: Data split to use ("train", "val", or "test")
        seed: Random seed for reproducible splits
        shuffle: Whether to shuffle the data
        stratify_by: Column to stratify split by (e.g., "family_label")
        min_class_count: Minimum samples per class for stratified split

    Returns:
        DataLoader
    """
    dataset = SVGXDataset(
        svg_dir=svg_dir,
        img_dir=img_dir,
        meta_filepath=meta_filepath,
        max_num_groups=max_num_groups,
        max_seq_len=max_seq_len,
        split=split,
        seed=seed,
        already_preprocessed=True,
        stratify_by=stratify_by,
        min_class_count=min_class_count,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate,
        drop_last=False,
    )

    logger.info(f"Created eval dataloader ({split}): {len(dataset)} samples")
    return loader


def add_common_args(parser):
    """Add common CLI arguments for evaluation scripts."""
    # Dataset args
    parser.add_argument("--svg-dir", type=str, required=True, help="SVG directory")
    parser.add_argument("--img-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--meta", type=str, required=True, help="Metadata CSV")
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max path groups")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")

    # Runtime args
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    return parser
