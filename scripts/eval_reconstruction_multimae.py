"""
MultiMAE Reconstruction Evaluation Script

Goal:
    Run the MultiMAE model in a *training-like* MAE scenario:
      - Apply the same masking strategy as in training
      - Decode the masked SVG groups from image + visible SVG (if any)
      - Compare ground truth vs reconstructed SVG

Usage (dataset mode):
    python scripts/eval_reconstruction_multimae.py \
        --ckpt checkpoints/multimae/best_model.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv \
        --idx 0 --out-dir recon_multimae --save-png

Usage (.pt mode):
    python scripts/eval_reconstruction_multimae.py \
        --ckpt checkpoints/multimae/best_model.pt \
        --pt sample.pt --img sample.png \
        --out-dir recon_multimae --save-png

Notes:
    - This uses model.greedy_reconstruct(), which internally:
        * runs svg_group_encoder + image encoder
        * applies _mask_svg_groups() and _mask_image_patches()
        * runs mae_encoder
        * decodes ONLY masked SVG groups with svg_decoder
        * merges visible groups (original) + predicted groups into final_cmd/final_args
    - For 1 to 2 visible SVG groups, the masking code masks *all* of them
      (cross-modal regime → pure prediction from image+CLS).
"""

import argparse
import logging
from pathlib import Path

import torch

from vecssl.data.dataset import SVGXDataset
from vecssl.data.geom import Bbox
from vecssl.data.svg import SVG
from vecssl.data.svg_tensor import SVGTensor
from vecssl.models.config import MultiMAEConfig
from vecssl.models.multimae import MultiMAE
from vecssl.util import setup_logging

import cairosvg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------


def svg_to_png(svg_path: str, png_path: str):
    """Convert SVG to PNG for easier viewing."""
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path, background_color="#ffffff")
    except Exception as e:
        logger.warning(f"Failed to convert {svg_path} to PNG: {e}")


def load_pt_file(
    pt_path: str,
    max_num_groups: int = 8,
    max_seq_len: int = 40,
    pad_val: int = -1,
) -> dict:
    """
    Load a .pt file (from preprocess_fonts.py with --to_tensor) and convert to model input format.

    Expects: {"t_sep": list of tensors, "fillings": list of ints}

    Returns:
        dict with:
            - "commands": (G, S)
            - "args": (G, S, n_args)
            - "uuid", "name"
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    t_sep = data["t_sep"]  # list of tensors, each (seq_len, 14)
    fillings = data["fillings"]  # list of ints

    # Pad if there are too few groups
    pad_len = max(max_num_groups - len(t_sep), 0)
    t_sep = list(t_sep)
    t_sep.extend([torch.empty(0, 14)] * pad_len)
    fillings = list(fillings)
    fillings.extend([0] * pad_len)

    processed = []
    for t, f in zip(t_sep[:max_num_groups], fillings[:max_num_groups], strict=True):
        svg_t = SVGTensor.from_data(t, PAD_VAL=pad_val, filling=f)
        svg_t = svg_t.add_eos().add_sos().pad(seq_len=max_seq_len + 2)
        processed.append(svg_t)

    commands = torch.stack([t.cmds() for t in processed])  # (G, S)
    args = torch.stack([t.args() for t in processed])  # (G, S, n_args)

    return {
        "commands": commands,
        "args": args,
        "uuid": Path(pt_path).stem,
        "name": Path(pt_path).stem,
    }


def load_multimae(checkpoint_path: Path, device: torch.device):
    """Load MultiMAE from checkpoint, restoring cfg and state_dict."""
    logger.info(f"Loading MultiMAE from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_obj = ckpt.get("cfg")
    if isinstance(cfg_obj, MultiMAEConfig):
        cfg = cfg_obj
    elif isinstance(cfg_obj, dict):
        cfg = MultiMAEConfig()
        for k, v in cfg_obj.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        logger.warning("No config in checkpoint, using defaults")
        cfg = MultiMAEConfig()

    model = MultiMAE(cfg)

    state_dict = ckpt["model"]
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, cfg


def decode_svg_from_cmd_args(commands, args, viewbox_size=256, pad_val=-1) -> SVG:
    """
    Decode commands and args tensors back to an SVG object.

    Filters out SOS/EOS/PAD tokens.

    Note:
        - If commands has shape (G, S), this currently decodes ONLY the first group.
          This is enough to sanity-check recon on single-path glyphs.
          You can extend this to draw all groups if desired.
    """
    # Handle multi-group case: use group 0
    if commands.ndim == 2:
        commands = commands[0]
        args = args[0]

    commands = commands.detach().cpu().long()
    args = args.detach().cpu().float()

    eos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
    sos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("SOS")

    valid_mask = (commands != pad_val) & (commands != eos_idx) & (commands != sos_idx)

    if not valid_mask.any():
        return SVG([], viewbox=Bbox(viewbox_size))

    commands = commands[valid_mask]
    args = args[valid_mask]

    try:
        svg_tensor = SVGTensor.from_cmd_args(commands, args, PAD_VAL=pad_val)
        tensor_data = svg_tensor.data
        svg = SVG.from_tensor(tensor_data, viewbox=Bbox(viewbox_size), allow_empty=True)
    except Exception as e:
        logger.warning(f"Failed to decode SVG: {e}")
        svg = SVG([], viewbox=Bbox(viewbox_size))

    return svg


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MultiMAE Reconstruction Evaluation")

    # Dataset mode
    parser.add_argument("--svg-dir", type=str, default=None, help="SVG directory")
    parser.add_argument("--img-dir", type=str, default=None, help="Image directory")
    parser.add_argument("--meta", type=str, default=None, help="Metadata CSV")
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max path groups")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument("--idx", type=int, default=None, help="Dataset index to reconstruct")

    parser.add_argument(
        "--use-precomputed-dino-patches",
        action="store_true",
        help="Use precomputed DINO patch embeddings from dataset",
    )
    parser.add_argument(
        "--dino-patches-dir",
        type=str,
        default=None,
        help="Directory containing precomputed DINO patches (.pt files)",
    )

    # .pt mode
    parser.add_argument("--pt", type=str, default=None, help="Path to .pt file sample")
    parser.add_argument(
        "--img",
        type=str,
        default=None,
        help="Path to image for the .pt sample (required if --pt and no dino_patches)",
    )

    # Model / output
    parser.add_argument("--ckpt", type=str, required=True, help="MultiMAE checkpoint path")
    parser.add_argument("--out-dir", type=str, default="recon_multimae", help="Output directory")
    parser.add_argument("--viewbox-size", type=int, default=256, help="SVG viewbox size")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--save-png", action="store_true", help="Also save PNG versions")

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("MultiMAE Reconstruction Evaluation")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, cfg = load_multimae(Path(args.ckpt), device)
    logger.info(f"Loaded MultiMAE with d_model={cfg.d_model}, max_num_groups={cfg.max_num_groups}")

    # Determine mode
    use_pt = args.pt is not None
    use_dataset = args.svg_dir is not None and args.meta is not None

    if not (use_pt ^ use_dataset):
        raise ValueError(
            "Specify exactly one of: "
            "(--pt [and optionally --img]) or "
            "(--svg-dir/--meta/--idx for dataset mode)."
        )

    # -----------------------------------------------------
    # Load single item
    # -----------------------------------------------------
    if use_pt:
        # .pt sample
        item = load_pt_file(
            args.pt,
            max_num_groups=args.max_num_groups,
            max_seq_len=args.max_seq_len,
        )

        # Attach image if provided
        if args.img is not None:
            from PIL import Image
            import torchvision.transforms as transforms

            to_tensor = transforms.ToTensor()
            img = Image.open(args.img).convert("RGB")
            item["image"] = to_tensor(img)
        else:
            # For a "realistic" training scenario, you probably want an image.
            logger.warning(
                "No --img provided for .pt sample. "
                "This will fail if MultiMAE expects an image (no dino_patches)."
            )

        name = item.get("name", Path(args.pt).stem)
        logger.info(f"Loaded .pt sample: {name}")

    else:
        # Dataset sample
        if args.idx is None:
            raise ValueError("--idx is required when using dataset mode")

        dataset = SVGXDataset(
            svg_dir=args.svg_dir,
            img_dir=args.img_dir,
            meta_filepath=args.meta,
            max_num_groups=args.max_num_groups,
            max_seq_len=args.max_seq_len,
            split="test",
            seed=42,
            already_preprocessed=True,
            use_precomputed_dino_patches=args.use_precomputed_dino_patches,
            dino_patches_dir=args.dino_patches_dir,
        )

        logger.info(f"Dataset size: {len(dataset)}")
        item = dataset[args.idx]
        name = item.get("name", item.get("uuid", f"idx{args.idx}"))
        logger.info(f"Loaded dataset sample: {name} (idx {args.idx})")

    # -----------------------------------------------------
    # Build batch for model.greedy_reconstruct
    # -----------------------------------------------------
    commands = item["commands"].unsqueeze(0)  # (1, G, S)
    args_svg = item["args"].unsqueeze(0)  # (1, G, S, n_args)

    batch = {
        "commands": commands,
        "args": args_svg,
    }

    if "dino_patches" in item:
        batch["dino_patches"] = item["dino_patches"].unsqueeze(0)
    elif "image" in item:
        batch["image"] = item["image"].unsqueeze(0)
    else:
        raise ValueError(
            "Sample does not contain 'image' or 'dino_patches'. "
            "MultiMAE expects one of them for image conditioning."
        )

    # -----------------------------------------------------
    # Run greedy reconstruction (training-like MAE scenario)
    # -----------------------------------------------------
    with torch.no_grad():
        recon_cmd, recon_args = model.greedy_reconstruct(batch)  # (1, G, S), (1, G, S, n_args)

    # -----------------------------------------------------
    # Save GT and reconstruction
    # -----------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ground truth SVG
    svg_gt = decode_svg_from_cmd_args(commands[0], args_svg[0], viewbox_size=args.viewbox_size)
    gt_path = out_dir / f"{name}_gt.svg"
    svg_gt.save_svg(str(gt_path))
    logger.info(f"Saved ground truth SVG to {gt_path}")
    if args.save_png:
        svg_to_png(str(gt_path), str(gt_path.with_suffix(".png")))

    # Reconstructed SVG
    svg_recon = decode_svg_from_cmd_args(
        recon_cmd[0], recon_args[0], viewbox_size=args.viewbox_size
    )
    recon_path = out_dir / f"{name}_recon.svg"
    svg_recon.save_svg(str(recon_path))
    logger.info(f"Saved reconstructed SVG to {recon_path}")
    if args.save_png:
        svg_to_png(str(recon_path), str(recon_path.with_suffix(".png")))

    logger.info("Done.")


if __name__ == "__main__":
    main()
