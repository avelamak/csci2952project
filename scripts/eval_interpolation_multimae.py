"""
Latent Interpolation Script for MultiMAE

Visualize latent space by interpolating between two SVGs in the group embedding space.

Usage (from dataset):
    python scripts/eval_interpolation_multimae.py \
        --ckpt checkpoints/multimae/best_model.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv \
        --idx-a 0 --idx-b 100 --num-steps 10 --out-dir interp_multimae

Usage (from .pt files):
    python scripts/eval_interpolation_multimae.py \
        --ckpt checkpoints/multimae/best_model.pt \
        --pt-a sample_a.pt --pt-b sample_b.pt \
        --num-steps 10 --out-dir interp_multimae
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
from vecssl.models.utils import _sample_categorical
from vecssl.util import setup_logging

import cairosvg

logger = logging.getLogger(__name__)


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

    The .pt file contains: {"t_sep": list of tensors, "fillings": list of ints}

    Returns a dict with "commands" and "args" tensors ready for the model.
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    t_sep = data["t_sep"]  # list of tensors, each (seq_len, 14)
    fillings = data["fillings"]  # list of ints

    # Pad if there are too few groups
    pad_len = max(max_num_groups - len(t_sep), 0)
    t_sep = list(t_sep)  # ensure it's a mutable list
    t_sep.extend([torch.empty(0, 14)] * pad_len)
    fillings = list(fillings)
    fillings.extend([0] * pad_len)

    # Convert to SVGTensor format with SOS/EOS/padding
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
    """Load MultiMAE from checkpoint."""
    logger.info(f"Loading MultiMAE from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config
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

    # Handle wrapped model state dict
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

    Args:
        commands: [S] or [G, S] long tensor of command indices
        args: [S, 11] or [G, S, 11] float tensor of arguments
        viewbox_size: Size of the SVG viewbox
        pad_val: Padding value used

    Returns:
        SVG object
    """
    # Handle multi-group case
    if commands.ndim == 2:
        commands = commands[0]
        args = args[0]

    commands = commands.detach().cpu()
    args = args.detach().cpu().float()

    # Filter out special tokens (SOS, EOS, padding)
    eos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
    sos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("SOS")
    valid_mask = (commands != pad_val) & (commands != eos_idx) & (commands != sos_idx)

    if not valid_mask.any():
        return SVG([], viewbox=Bbox(viewbox_size))

    # Use mask to filter out invalid tokens
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


# def decode_svg_from_cmd_args(commands, args, viewbox_size=256, pad_val=-1) -> SVG:
#     """
#     Decode commands and args tensors back to an SVG object.
#     Filters out SOS/EOS/PAD tokens.
#     """
#     if commands.ndim == 2:
#         commands = commands[0]
#         args = args[0]

#     commands = commands.detach().cpu().long()
#     args = args.detach().cpu().float()

#     cmd_vocab = SVGTensor.COMMANDS_SIMPLIFIED
#     try:
#         SOS_idx = cmd_vocab.index("SOS")
#     except ValueError:
#         SOS_idx = -999
#     try:
#         EOS_idx = cmd_vocab.index("EOS")
#     except ValueError:
#         EOS_idx = -999

#     valid_cmds = []
#     valid_args = []

#     for i, c_idx in enumerate(commands.tolist()):
#         if c_idx == EOS_idx:
#             break
#         if c_idx == SOS_idx or c_idx == pad_val or c_idx < 0:
#             continue
#         valid_cmds.append(c_idx)
#         valid_args.append(args[i])

#     if len(valid_cmds) == 0:
#         return SVG([], viewbox=Bbox(viewbox_size))

#     commands_t = torch.tensor(valid_cmds, dtype=torch.long)
#     args_t = torch.stack(valid_args)

#     try:
#         svg_tensor = SVGTensor.from_cmd_args(commands_t, args_t, PAD_VAL=pad_val)
#         tensor_data = svg_tensor.data
#         svg = SVG.from_tensor(tensor_data, viewbox=Bbox(viewbox_size), allow_empty=True)
#     except Exception as e:
#         logger.warning(f"Failed to decode SVG: {e}")
#         svg = SVG([], viewbox=Bbox(viewbox_size))

#     return svg


@torch.no_grad()
def encode_to_group_embeddings(model: MultiMAE, item: dict, device: torch.device):
    """
    Encode a single sample to SVG group embeddings.

    Returns:
        group_embs: (1, G, D) - Group embeddings with positional encoding
        img_tokens: (1, P, D) - Image patch tokens (projected), or None if no image
        visibility_mask: (1, G) - Which groups are valid
        commands: (1, G, S) - Original commands
        args: (1, G, S, n_args) - Original args
    """
    commands = item["commands"].unsqueeze(0).to(device)  # (1, G, S)
    args = item["args"].unsqueeze(0).to(device)  # (1, G, S, n_args)

    # Get group embeddings
    group_embs = model.svg_group_encoder(commands, args)  # (1, G, D)

    # Add positional embeddings
    G = group_embs.shape[1]
    group_embs = group_embs + model.group_pos_embed[:, :G, :]

    # Get image tokens (if available)
    img_tokens = None
    if "dino_patches" in item:
        img_patches = item["dino_patches"].unsqueeze(0).to(device)
        img_tokens = model.img_proj(img_patches)
    elif "image" in item:
        images = item["image"].unsqueeze(0).to(device)
        img_patches = model.image_encoder(images)
        img_tokens = model.img_proj(img_patches)
    else:
        # No image provided - create dummy zero tokens
        # This allows SVG-only interpolation without images
        logger.warning("No image provided, using zero image tokens (SVG-only mode)")
        num_patches = 196  # Standard for DINOv2 with 224x224 input
        img_tokens = torch.zeros(1, num_patches, model.cfg.d_model, device=device)

    # Visibility mask
    EOS_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
    visibility_mask = commands[:, :, 0] != EOS_idx  # (1, G)

    return group_embs, img_tokens, visibility_mask, commands, args


@torch.no_grad()
def decode_from_embeddings(
    model: MultiMAE,
    group_embs: torch.Tensor,
    img_tokens: torch.Tensor,
    visibility_mask: torch.Tensor,
    temperature: float = 0.0001,
):
    """
    Decode SVG group embeddings back to commands and args.

    This creates query tokens for all groups and uses the decoder
    to reconstruct the full SVG.

    Args:
        model: MultiMAE model
        group_embs: (N, G, D) - Group embeddings (already has positional encoding)
        img_tokens: (N, P, D) - Image patch tokens
        visibility_mask: (N, G) - Which groups are valid
        temperature: Sampling temperature

    Returns:
        commands: (N, G, S) - Predicted command indices
        args: (N, G, S, n_args) - Predicted arguments
    """
    device = group_embs.device
    N, G, D = group_embs.shape
    # S = model.cfg.max_seq_len

    # Add modality embeddings
    svg_tokens = group_embs + model.mod_embed_svg
    img_tokens = img_tokens + model.mod_embed_img

    # Build encoder input (CLS + SVG + IMG)
    cls_token = model.cls_token.expand(N, -1, -1)
    enc_input = torch.cat([cls_token, svg_tokens, img_tokens], dim=1)  # (N, 1+G+P, D)

    # Padding mask: True = padding (ignore)
    cls_pad = torch.zeros(N, 1, dtype=torch.bool, device=device)
    svg_pad = ~visibility_mask  # Invalid groups are padding
    img_pad = torch.zeros(N, img_tokens.shape[1], dtype=torch.bool, device=device)
    enc_key_padding_mask = torch.cat([cls_pad, svg_pad, img_pad], dim=1)

    # Run MAE encoder
    memory = model.mae_encoder(
        enc_input.transpose(0, 1), src_key_padding_mask=enc_key_padding_mask
    )  # (L, N, D)

    # Create query tokens for ALL groups (we want to decode everything)
    # Use mask tokens + positional embeddings as queries
    query_tokens = model.svg_mask_token.repeat(N, G, 1)  # (N, G, D)
    query_tokens = query_tokens + model.group_pos_embed[:, :G, :]

    # Decode
    pred_cmd_logits, pred_args_logits = model.svg_decoder(
        query_tokens, memory, memory_key_padding_mask=enc_key_padding_mask
    )

    # Sample from logits
    pred_cmd, pred_args = _sample_categorical(temperature, pred_cmd_logits, pred_args_logits)
    pred_args = pred_args - 1  # Shift back from [0, 256] to [-1, 255]

    # Mask invalid args
    cmd_args_mask = model.cmd_args_mask[pred_cmd.long()].bool()
    pred_args[~cmd_args_mask] = -1

    return pred_cmd, pred_args


def interpolate_embeddings(emb_a: torch.Tensor, emb_b: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation between two embedding tensors."""
    return (1 - t) * emb_a + t * emb_b


def main():
    parser = argparse.ArgumentParser(description="Latent interpolation for MultiMAE")

    # Dataset args (for loading from dataset)
    parser.add_argument("--svg-dir", type=str, default=None, help="SVG directory")
    parser.add_argument("--img-dir", type=str, default=None, help="Image directory")
    parser.add_argument("--meta", type=str, default=None, help="Metadata CSV")
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max path groups")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument(
        "--use-precomputed-dino-patches",
        action="store_true",
        help="Use precomputed DINO patch embeddings",
    )
    parser.add_argument(
        "--dino-patches-dir",
        type=str,
        default=None,
        help="Directory containing precomputed DINO patches",
    )

    # Direct .pt file input (alternative to dataset)
    parser.add_argument("--pt-a", type=str, default=None, help="Path to .pt file for sample A")
    parser.add_argument("--pt-b", type=str, default=None, help="Path to .pt file for sample B")
    parser.add_argument(
        "--img-a",
        type=str,
        default=None,
        help="Path to image for sample A (optional, for cross-modal)",
    )
    parser.add_argument(
        "--img-b",
        type=str,
        default=None,
        help="Path to image for sample B (optional, for cross-modal)",
    )

    # Model args
    parser.add_argument("--ckpt", type=str, required=True, help="MultiMAE checkpoint path")

    # Interpolation args (for dataset mode)
    parser.add_argument("--idx-a", type=int, default=None, help="Index of first SVG (dataset mode)")
    parser.add_argument(
        "--idx-b", type=int, default=None, help="Index of second SVG (dataset mode)"
    )
    parser.add_argument("--num-steps", type=int, default=10, help="Number of interpolation steps")
    parser.add_argument("--out-dir", type=str, default="interp_multimae", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.0001, help="Sampling temperature")

    # Interpolation mode
    parser.add_argument(
        "--interp-mode",
        type=str,
        choices=["svg_only", "both", "img_only"],
        default="svg_only",
        help="What to interpolate: svg_only (SVG embeddings), both (SVG + image), img_only (image embeddings only)",
    )

    # Runtime args
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--save-png", action="store_true", help="Also save PNG versions")

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("MultiMAE Latent Interpolation")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, cfg = load_multimae(Path(args.ckpt), device)
    logger.info(f"Loaded MultiMAE with d_model={cfg.d_model}, max_num_groups={cfg.max_num_groups}")

    # Determine input mode: .pt files or dataset
    use_pt_files = args.pt_a is not None and args.pt_b is not None
    use_dataset = args.svg_dir is not None and args.meta is not None

    if use_pt_files:
        # Load from .pt files directly
        logger.info("Loading samples from .pt files...")
        item_a = load_pt_file(
            args.pt_a,
            max_num_groups=args.max_num_groups,
            max_seq_len=args.max_seq_len,
        )
        item_b = load_pt_file(
            args.pt_b,
            max_num_groups=args.max_num_groups,
            max_seq_len=args.max_seq_len,
        )

        # Load images if provided (for cross-modal interpolation)
        if args.img_a:
            from PIL import Image
            import torchvision.transforms as transforms

            to_tensor = transforms.ToTensor()
            img_a = Image.open(args.img_a).convert("RGB")
            item_a["image"] = to_tensor(img_a)
        if args.img_b:
            from PIL import Image
            import torchvision.transforms as transforms

            to_tensor = transforms.ToTensor()
            img_b = Image.open(args.img_b).convert("RGB")
            item_b["image"] = to_tensor(img_b)

        logger.info(f"Sample A: {item_a['name']} (from {args.pt_a})")
        logger.info(f"Sample B: {item_b['name']} (from {args.pt_b})")

    elif use_dataset:
        # Load from dataset
        if args.idx_a is None or args.idx_b is None:
            raise ValueError("--idx-a and --idx-b are required when using dataset mode")

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
        item_a = dataset[args.idx_a]
        item_b = dataset[args.idx_b]

        logger.info(f"Sample A: {item_a.get('name', item_a['uuid'])} (idx {args.idx_a})")
        logger.info(f"Sample B: {item_b.get('name', item_b['uuid'])} (idx {args.idx_b})")
    else:
        raise ValueError(
            "Must provide either --pt-a/--pt-b for .pt file mode, "
            "or --svg-dir/--meta/--idx-a/--idx-b for dataset mode"
        )

    logger.info(f"Interpolation mode: {args.interp_mode}")

    # Encode both samples
    group_embs_a, img_tokens_a, vis_mask_a, cmd_a, args_a = encode_to_group_embeddings(
        model, item_a, device
    )
    group_embs_b, img_tokens_b, vis_mask_b, cmd_b, args_b = encode_to_group_embeddings(
        model, item_b, device
    )

    logger.info(f"Group embeddings A shape: {group_embs_a.shape}")
    logger.info(f"Group embeddings B shape: {group_embs_b.shape}")
    logger.info(f"Image tokens A shape: {img_tokens_a.shape}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original samples (ground truth)
    for tag, cmd, arg in [("a_gt", cmd_a, args_a), ("b_gt", cmd_b, args_b)]:
        svg = decode_svg_from_cmd_args(cmd[0], arg[0])
        out_path = out_dir / f"interp_{tag}.svg"
        svg.save_svg(str(out_path))
        logger.info(f"Saved {out_path}")
        if args.save_png:
            svg_to_png(str(out_path), str(out_path.with_suffix(".png")))

    # Save reconstructions of endpoints (to verify encoder-decoder roundtrip)
    for tag, group_embs, img_tokens, vis_mask in [
        ("a_recon", group_embs_a, img_tokens_a, vis_mask_a),
        ("b_recon", group_embs_b, img_tokens_b, vis_mask_b),
    ]:
        pred_cmd, pred_args = decode_from_embeddings(
            model, group_embs, img_tokens, vis_mask, args.temperature
        )
        svg = decode_svg_from_cmd_args(pred_cmd[0], pred_args[0])
        out_path = out_dir / f"interp_{tag}.svg"
        svg.save_svg(str(out_path))
        logger.info(f"Saved {out_path}")
        if args.save_png:
            svg_to_png(str(out_path), str(out_path.with_suffix(".png")))

    # Interpolation
    # Use the union of visibility masks (conservative: if either has a valid group, keep it)
    vis_mask_interp = vis_mask_a | vis_mask_b

    logger.info(f"Starting interpolation with {args.num_steps} steps...")

    for i, t in enumerate(torch.linspace(0.0, 1.0, args.num_steps)):
        t_val = t.item()

        # Interpolate based on mode
        if args.interp_mode == "svg_only":
            group_embs_t = interpolate_embeddings(group_embs_a, group_embs_b, t_val)
            img_tokens_t = img_tokens_a  # Keep image from A
        elif args.interp_mode == "both":
            group_embs_t = interpolate_embeddings(group_embs_a, group_embs_b, t_val)
            img_tokens_t = interpolate_embeddings(img_tokens_a, img_tokens_b, t_val)
        elif args.interp_mode == "img_only":
            group_embs_t = group_embs_a  # Keep SVG from A
            img_tokens_t = interpolate_embeddings(img_tokens_a, img_tokens_b, t_val)

        # Decode
        pred_cmd_t, pred_args_t = decode_from_embeddings(
            model, group_embs_t, img_tokens_t, vis_mask_interp, args.temperature
        )

        # Save
        svg_t = decode_svg_from_cmd_args(pred_cmd_t[0], pred_args_t[0])
        out_path = out_dir / f"interp_{i:02d}_t{t_val:.2f}.svg"
        svg_t.save_svg(str(out_path))

        if args.save_png:
            svg_to_png(str(out_path), str(out_path.with_suffix(".png")))

    logger.info(f"Saved {args.num_steps} interpolation steps to {out_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
