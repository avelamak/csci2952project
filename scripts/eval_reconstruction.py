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
from vecssl.data.geom import Bbox
from vecssl.data.svg import SVG
from vecssl.data.svg_tensor import SVGTensor
from vecssl.models.config import _DefaultConfig
from vecssl.models.model import SVGTransformer
from vecssl.util import setup_logging

from eval_utils import add_common_args, custom_collate
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def decode_svg_from_cmd_args(
    commands: torch.Tensor,
    args: torch.Tensor,
    viewbox_size: int = 256,
    pad_val: int = -1,
) -> SVG:
    """
    Decode commands + args into an SVG, keeping all groups and stripping
    EOS/SOS/padded rows so SVG.from_tensor only sees real drawing commands.

    Args:
        commands: [T], [G, S], or [B, G, S] long tensor of command indices
        args: [T, n_args], [G, S, n_args], or [B, G, S, n_args] float tensor
        viewbox_size: Size of the SVG viewbox (default 256)
        pad_val: Padding value for args (default -1)

    Returns:
        SVG object that can be saved via .save_svg()
    """
    commands = commands.detach().cpu().long()
    args = args.detach().cpu().float()

    # Strip batch dim if present: [B, G, S] -> [G, S]
    print(f"{commands.shape=}")
    print(f"{args.shape=}")
    print(f"{commands=}")
    print(f"{args}")
    if commands.ndim == 3:
        commands = commands[0]
        args = args[0]
    # If grouped, flatten (G, S) -> T
    print(f"{commands.shape=}")
    print(f"{args.shape=}")
    print(f"{commands=}")
    print(f"{args}")
    # If we have groups [G, S], flatten to [T]
    if commands.ndim == 2:
        G, S = commands.shape
        commands = commands.reshape(-1)  # [T]
        args = args.reshape(-1, args.size(-1))  # [T, n_args]
    elif commands.ndim == 1:
        # Already flat [T]
        pass
    else:
        raise ValueError(f"Unexpected commands ndim={commands.ndim} in decode")

    cmds_list = SVGTensor.COMMANDS_SIMPLIFIED
    eos_idx = cmds_list.index("EOS")
    sos_idx = cmds_list.index("SOS")

    # 1) Drop EOS/SOS meta tokens (including EOS-as-padding)
    meta_mask = (commands != eos_idx) & (commands != sos_idx)

    # 2) Drop rows where all args are PAD (-1): completely empty
    if pad_val is not None:
        nonempty_mask = ~(args == pad_val).all(dim=-1)
    else:
        nonempty_mask = torch.ones_like(commands, dtype=torch.bool)

    keep_mask = meta_mask & nonempty_mask

    commands = commands[keep_mask]
    args = args[keep_mask]

    if commands.numel() == 0:
        # Nothing real to draw
        return SVG([], viewbox=Bbox(viewbox_size))

    svg_tensor = SVGTensor.from_cmd_args(commands, args, PAD_VAL=pad_val)

    svg = SVG.from_tensor(
        svg_tensor.data,
        viewbox=Bbox(viewbox_size),
        allow_empty=True,
    )

    return svg


def save_reconstruction_results(
    save_dir: Path,
    uuids: list,
    pred_cmds: torch.Tensor,
    pred_args: torch.Tensor,
    tgt_cmds: torch.Tensor,
    tgt_args: torch.Tensor,
):
    """Save ground truth and reconstructed SVGs."""
    save_dir.mkdir(parents=True, exist_ok=True)

    B = pred_cmds.size(0)
    for i in range(B):
        uuid = uuids[i]
        out_dir = save_dir / uuid
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            gt_svg = decode_svg_from_cmd_args(tgt_cmds[i], tgt_args[i])
            gt_svg.save_svg(out_dir / "gt.svg")
        except Exception as e:
            logger.warning(f"Failed to save GT SVG for {uuid}: {e}")

        try:
            recon_svg = decode_svg_from_cmd_args(pred_cmds[i], pred_args[i])
            recon_svg.save_svg(out_dir / "recon.svg")
        except Exception as e:
            logger.warning(f"Failed to save recon SVG for {uuid}: {e}")


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


def command_accuracy(pred_cmds: torch.Tensor, tgt_cmds: torch.Tensor) -> float:
    """
    Compute command token accuracy over real drawing commands (m, l, c, a).

    COMMANDS_SIMPLIFIED = ["m", "l", "c", "a", "EOS", "SOS", "z"]
    Real commands have indices 0-3 (< EOS index).

    Args:
        pred_cmds: [B, G, S] predicted commands
        tgt_cmds: [B, G, S] target commands

    Returns:
        float: Accuracy
    """
    # Align sequence lengths
    S = min(pred_cmds.size(-1), tgt_cmds.size(-1))
    pred = pred_cmds[..., :S]
    tgt = tgt_cmds[..., :S]

    # Real commands are indices 0-3 (m, l, c, a), anything >= EOS is special
    eos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")  # 4
    mask = tgt < eos_idx

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
    parser.add_argument(
        "--save-reconstruction-dir",
        type=str,
        default=None,
        help="Directory to save reconstructed SVGs (gt.svg + recon.svg per sample)",
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

    # Create dataloader using test split for proper evaluation
    dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        split="test",
        seed=42,
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

    should_save = args.save_reconstruction_dir is not None

    for batch in loader:
        pred_cmds, pred_args, tgt_cmds, tgt_args = reconstruct_batch(model, batch, device)

        ca = command_accuracy(pred_cmds, tgt_cmds)
        l1, l2 = arg_errors(pred_args, tgt_args, pad_val=-1)

        total_cmd_acc += ca
        total_l1 += l1
        total_l2 += l2
        n_batches += 1

        # Save reconstructed SVGs if requested
        if should_save:
            save_reconstruction_results(
                save_dir=Path(args.save_reconstruction_dir),
                uuids=batch["uuid"],
                pred_cmds=pred_cmds,
                pred_args=pred_args,
                tgt_cmds=tgt_cmds,
                tgt_args=tgt_args,
            )

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
