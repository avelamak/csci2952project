"""
Reconstruction Evaluation Script

Evaluate autoencoder reconstruction quality with:
- Command accuracy (exact token match)
- Arguments L1/L2 error

Usage:
    python scripts/eval_reconstruction.py \
        --ae-ckpt checkpoints/decoder/best_model.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv
"""

import argparse
import logging
from pathlib import Path

import torch

from vecssl.data.dataset import SVGXDataset
from vecssl.models.config import _DefaultConfig
from vecssl.models.model import SVGTransformer
from vecssl.util import setup_logging

from eval_utils import add_common_args, custom_collate
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_autoencoder(checkpoint_path: Path, device: torch.device):
    """
    Load autoencoder from checkpoint.

    Handles both SimpleSVGAutoencoder wrapper and raw SVGTransformer checkpoints.
    """
    logger.info(f"Loading autoencoder from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Get config
    cfg_obj = ckpt.get("cfg")
    if isinstance(cfg_obj, _DefaultConfig):
        cfg = cfg_obj
    elif isinstance(cfg_obj, dict):
        cfg = _DefaultConfig()
        for k, v in cfg_obj.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        logger.warning("No config in checkpoint, using defaults")
        cfg = _DefaultConfig()
        cfg.encode_stages = 2
        cfg.decode_stages = 2
        cfg.use_vae = True

    # Create model
    model = SVGTransformer(cfg)

    # Load state dict - handle wrapped model (SimpleSVGAutoencoder)
    state_dict = ckpt["model"]

    # Check if this is a wrapped model (has 'model.' prefix)
    if any(k.startswith("model.") for k in state_dict.keys()):
        # Remove 'model.' prefix
        state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(
        f"Loaded autoencoder: encode_stages={cfg.encode_stages}, decode_stages={cfg.decode_stages}"
    )
    return model, cfg


@torch.no_grad()
def reconstruct_batch(model, batch, device):
    """
    Encode and decode a batch, return predictions and targets.

    Returns:
        tuple: (pred_cmds, pred_args, tgt_cmds, tgt_args)
    """
    commands = batch["commands"].to(device)  # [B, G, S]
    args = batch["args"].to(device)  # [B, G, S, n_args]

    # Use greedy sampling with encoding from input
    commands_y, args_y = model.greedy_sample(
        commands_enc=commands,
        args_enc=args,
        commands_dec=None,
        args_dec=None,
        z=None,
        concat_groups=False,
        temperature=0.0001,
    )

    return commands_y.cpu(), args_y.cpu(), commands.cpu(), args.cpu()


def command_accuracy(pred_cmds, tgt_cmds, pad_val=-1):
    """
    Compute command token accuracy.

    Args:
        pred_cmds: [B, G, S] predicted commands
        tgt_cmds: [B, G, S] target commands
        pad_val: Padding value to ignore

    Returns:
        float: Accuracy
    """
    # Align sequence lengths
    S = min(pred_cmds.size(-1), tgt_cmds.size(-1))
    pred = pred_cmds[..., :S]
    tgt = tgt_cmds[..., :S]

    # Create mask for valid (non-padded) positions
    mask = tgt != pad_val
    correct = (pred == tgt) & mask

    acc = correct.sum().float() / mask.sum().clamp(min=1)
    return acc.item()


def arg_errors(pred_args, tgt_args, pad_val=-1):
    """
    Compute argument L1 and L2 errors.

    Args:
        pred_args: [B, G, S, n_args] predicted arguments
        tgt_args: [B, G, S, n_args] target arguments
        pad_val: Padding value to ignore

    Returns:
        tuple: (L1 error, L2 error)
    """
    # Align sequence lengths
    S = min(pred_args.size(-2), tgt_args.size(-2))
    pred = pred_args[..., :S, :].float()
    tgt = tgt_args[..., :S, :].float()

    # Mask valid args
    mask = tgt != pad_val
    diff = (pred - tgt) * mask

    l1 = diff.abs().sum() / mask.sum().clamp(min=1)
    l2 = torch.sqrt((diff**2).sum() / mask.sum().clamp(min=1))

    return l1.item(), l2.item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate SVG reconstruction quality")
    parser = add_common_args(parser)

    # Model args
    parser.add_argument(
        "--ae-ckpt",
        type=str,
        required=True,
        help="Path to autoencoder checkpoint",
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("Reconstruction Evaluation")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load autoencoder
    model, cfg = load_autoencoder(Path(args.ae_ckpt), device)

    # Create dataloader
    dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        train_ratio=1.0,
        already_preprocessed=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        drop_last=False,
    )

    logger.info(f"Evaluating on {len(dataset)} samples...")

    # Accumulate metrics
    total_cmd_acc = 0.0
    total_l1 = 0.0
    total_l2 = 0.0
    n_batches = 0

    for batch in loader:
        pred_cmds, pred_args, tgt_cmds, tgt_args = reconstruct_batch(model, batch, device)

        ca = command_accuracy(pred_cmds, tgt_cmds, pad_val=-1)
        l1, l2 = arg_errors(pred_args, tgt_args, pad_val=-1)

        total_cmd_acc += ca
        total_l1 += l1
        total_l2 += l2
        n_batches += 1

        if n_batches % 10 == 0:
            logger.info(f"Processed {n_batches * args.batch_size} samples...")

    # Report results
    logger.info("=" * 60)
    logger.info("Results:")
    logger.info(f"  Command accuracy: {total_cmd_acc / n_batches:.4f}")
    logger.info(f"  Args L1 error:    {total_l1 / n_batches:.4f}")
    logger.info(f"  Args L2 error:    {total_l2 / n_batches:.4f}")
    logger.info("=" * 60)

    # Future: Add hooks for geometric metrics
    # - Chamfer distance: sample points from pred/gt SVGs, compute Chamfer
    # - PSNR: render pred/gt to images, compute PSNR
    logger.info("Note: Chamfer distance and PSNR not yet implemented.")
    logger.info("Done.")


if __name__ == "__main__":
    main()
