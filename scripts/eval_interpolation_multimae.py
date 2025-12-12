"""
DecoderFromMultiMAE Latent Interpolation Script

Visualize latent space by interpolating between two SVGs in the CLS embedding space.

Flow:
    1. Encode two SVG samples via frozen MultiMAE encoder -> CLS embeddings
    2. Linearly interpolate in embedding space
    3. Decode interpolated embeddings via SVGTransformer

Usage (from dataset):
    python scripts/eval_interpolation_multimae.py \
        --ckpt checkpoints/multimae_decoder/checkpoint.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv \
        --idx-a 0 --idx-b 100 --num-steps 10 --out-dir interp_multimae

Usage (from .pt files):
    python scripts/eval_interpolation_multimae.py \
        --ckpt checkpoints/multimae_decoder/checkpoint.pt \
        --pt-a sample_a.pt --pt-b sample_b.pt \
        --num-steps 10 --out-dir interp_multimae
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from vecssl.data.dataset import SVGXDataset
from vecssl.data.geom import Bbox
from vecssl.data.svg import SVG
from vecssl.data.svg_path import SVGPath
from vecssl.data.svg_tensor import SVGTensor
from vecssl.models.config import MultiMAEConfig, _DefaultConfig
from vecssl.models.multimae import MultiMAE
from vecssl.models.model import SVGTransformer
from vecssl.models.loss import SVGLoss
from vecssl.util import setup_logging

import cairosvg

logger = logging.getLogger(__name__)


# =============================================================================
# DecoderFromMultiMAE Model Definition
# =============================================================================


class DecoderFromMultiMAE(nn.Module):
    """
    Frozen MultiMAE encoder + trainable SVG decoder.

    Flow:
        1. Get CLS embedding from frozen MultiMAE via encode_joint()
        2. Reshape CLS to match decoder expected input
        3. Decode to SVG commands/args
    """

    def __init__(
        self,
        frozen_encoder: MultiMAE,
        decoder_cfg: _DefaultConfig,
    ):
        super().__init__()
        self.encoder = frozen_encoder
        self.cfg = decoder_cfg

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Create SVGTransformer for decoder
        self.svg_transformer = SVGTransformer(decoder_cfg)

        # Freeze the encoder part of SVGTransformer (we use MultiMAE instead)
        if hasattr(self.svg_transformer, "encoder"):
            for p in self.svg_transformer.encoder.parameters():
                p.requires_grad = False

        # Create loss function (needed for checkpoint loading)
        self.loss_fn = SVGLoss(decoder_cfg)


# =============================================================================
# Utilities
# =============================================================================


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


def load_decoder_from_multimae(checkpoint_path: Path, device: torch.device):
    """
    Load DecoderFromMultiMAE from checkpoint.

    Handles checkpoints from train_decoder_from_multimae.py which contain:
    - encoder.* keys (MultiMAE)
    - svg_transformer.* keys (SVGTransformer decoder)
    - loss_fn.* keys (SVGLoss buffers)
    """
    logger.info(f"Loading DecoderFromMultiMAE from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get configs from checkpoint
    multimae_cfg = ckpt.get("multimae_cfg")
    if multimae_cfg is None:
        # Try to reconstruct from checkpoint keys or use defaults
        multimae_cfg = ckpt.get("cfg")
        if isinstance(multimae_cfg, MultiMAEConfig):
            pass
        elif isinstance(multimae_cfg, dict):
            cfg_temp = MultiMAEConfig()
            for k, v in multimae_cfg.items():
                if hasattr(cfg_temp, k):
                    setattr(cfg_temp, k, v)
            multimae_cfg = cfg_temp
        else:
            logger.warning("No MultiMAE config in checkpoint, using defaults")
            multimae_cfg = MultiMAEConfig()

    decoder_cfg = ckpt.get("decoder_cfg")
    if decoder_cfg is None:
        # Use defaults
        logger.warning("No decoder config in checkpoint, using defaults")
        decoder_cfg = _DefaultConfig()
        decoder_cfg.encode_stages = 2
        decoder_cfg.decode_stages = 2
        decoder_cfg.use_vae = False
        decoder_cfg.pred_mode = "one_shot"

    # Create model
    frozen_encoder = MultiMAE(multimae_cfg)
    model = DecoderFromMultiMAE(frozen_encoder, decoder_cfg)

    # Load state dict - handle 'model.' prefix if present
    state_dict = ckpt["model"]
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Loaded DecoderFromMultiMAE (d_model={multimae_cfg.d_model})")
    return model, decoder_cfg


def decode_svg_from_cmd_args(commands, args, viewbox_size=256, pad_val=-1) -> SVG:
    """
    Decode commands and args tensors back to an SVG object.

    Filters out SOS/EOS/PAD tokens.

    Supports both single-group (S,) and multi-group (G, S) tensors.
    For multi-group input, all groups are decoded and combined into one SVG.
    """
    commands = commands.detach().cpu().long()
    args = args.detach().cpu().float()

    eos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
    sos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("SOS")

    # Handle both (S,) and (G, S) shapes
    if commands.ndim == 1:
        commands_list = [commands]
        args_list = [args]
    else:
        # Multi-group case (G, S) and (G, S, n_args)
        commands_list = list(commands)
        args_list = list(args)

    # Decode each group
    svg_path_groups = []
    for cmd, arg in zip(commands_list, args_list, strict=False):
        # Filter out SOS/EOS/PAD tokens
        valid_mask = (cmd != pad_val) & (cmd != eos_idx) & (cmd != sos_idx)

        if not valid_mask.any():
            # Empty group - skip
            continue

        cmd_valid = cmd[valid_mask]
        arg_valid = arg[valid_mask]

        try:
            svg_tensor = SVGTensor.from_cmd_args(cmd_valid, arg_valid, PAD_VAL=pad_val)
            tensor_data = svg_tensor.data
            # SVGPath.from_tensor returns an SVGPathGroup
            svg_path_group = SVGPath.from_tensor(tensor_data, allow_empty=True)
            svg_path_groups.append(svg_path_group)
        except Exception as e:
            logger.warning(f"Failed to decode group: {e}")
            continue

    if not svg_path_groups:
        return SVG([], viewbox=Bbox(viewbox_size))

    return SVG(svg_path_groups, viewbox=Bbox(viewbox_size))


# =============================================================================
# Encoding and Decoding
# =============================================================================


@torch.no_grad()
def encode_sample(model: DecoderFromMultiMAE, item: dict, device: torch.device) -> torch.Tensor:
    """
    Encode a single sample to latent z using frozen MultiMAE encoder.

    Args:
        model: DecoderFromMultiMAE model
        item: Sample dict with commands, args, and image/dino_patches
        device: torch device

    Returns:
        z: Latent tensor [1, 1, 1, d_model] ready for svg_transformer.greedy_sample()
    """
    # Build batch from single item
    batch = {
        "commands": item["commands"].unsqueeze(0),
        "args": item["args"].unsqueeze(0),
    }

    if "dino_patches" in item:
        batch["dino_patches"] = item["dino_patches"].unsqueeze(0)
    elif "image" in item:
        batch["image"] = item["image"].unsqueeze(0)

    batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Get CLS embedding from frozen MultiMAE
    joint = model.encoder.encode_joint(batch_device)
    cls_embed = joint["svg"]  # [1, d_model]

    # Reshape to match decoder expected input: (1, d_model) -> (1, 1, 1, d_model)
    z = cls_embed.unsqueeze(1).unsqueeze(1)

    return z


@torch.no_grad()
def decode_z(
    model: DecoderFromMultiMAE, z: torch.Tensor, temperature: float = 0.0001
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode latent z to commands and args using SVGTransformer decoder.

    Args:
        model: DecoderFromMultiMAE model
        z: Latent tensor [N, 1, 1, d_model]
        temperature: Sampling temperature

    Returns:
        tuple: (commands [N, seq_len], args [N, seq_len, n_args])
    """
    commands_y, args_y = model.svg_transformer.greedy_sample(
        commands_enc=None,
        args_enc=None,
        z=z,
        concat_groups=True,
        temperature=temperature,
    )
    return commands_y, args_y


def interpolate_z(z_a: torch.Tensor, z_b: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation between two latent tensors."""
    return (1 - t) * z_a + t * z_b


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="DecoderFromMultiMAE Latent Interpolation")

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
        help="Path to image for sample A (optional)",
    )
    parser.add_argument(
        "--img-b",
        type=str,
        default=None,
        help="Path to image for sample B (optional)",
    )

    # Model args
    parser.add_argument(
        "--ckpt", type=str, required=True, help="DecoderFromMultiMAE checkpoint path"
    )

    # Interpolation args (for dataset mode)
    parser.add_argument("--idx-a", type=int, default=None, help="Index of first SVG (dataset mode)")
    parser.add_argument(
        "--idx-b", type=int, default=None, help="Index of second SVG (dataset mode)"
    )
    parser.add_argument("--num-steps", type=int, default=10, help="Number of interpolation steps")
    parser.add_argument("--out-dir", type=str, default="interp_multimae", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.0001, help="Sampling temperature")
    parser.add_argument("--viewbox-size", type=int, default=256, help="SVG viewbox size")

    # Runtime args
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--save-png", action="store_true", help="Also save PNG versions")

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("DecoderFromMultiMAE Latent Interpolation")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, cfg = load_decoder_from_multimae(Path(args.ckpt), device)

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

        # Load images if provided
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

    # Encode both samples
    z_a = encode_sample(model, item_a, device)
    z_b = encode_sample(model, item_b, device)

    logger.info(f"Encoded z_a shape: {z_a.shape}")
    logger.info(f"Encoded z_b shape: {z_b.shape}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save ground truth endpoints
    for tag, item in [("a_gt", item_a), ("b_gt", item_b)]:
        svg = decode_svg_from_cmd_args(
            item["commands"], item["args"], viewbox_size=args.viewbox_size
        )
        out_path = out_dir / f"interp_{tag}.svg"
        svg.save_svg(str(out_path))
        logger.info(f"Saved {out_path}")
        if args.save_png:
            svg_to_png(str(out_path), str(out_path.with_suffix(".png")))

    # Save reconstructions of endpoints (to verify encoder-decoder roundtrip)
    for tag, z in [("a_recon", z_a), ("b_recon", z_b)]:
        commands_y, args_y = decode_z(model, z, args.temperature)
        svg = decode_svg_from_cmd_args(commands_y[0], args_y[0], viewbox_size=args.viewbox_size)
        out_path = out_dir / f"interp_{tag}.svg"
        svg.save_svg(str(out_path))
        logger.info(f"Saved {out_path}")
        if args.save_png:
            svg_to_png(str(out_path), str(out_path.with_suffix(".png")))

    # Linear interpolation
    logger.info(f"Starting interpolation with {args.num_steps} steps...")

    for i, t in enumerate(torch.linspace(0.0, 1.0, args.num_steps)):
        t_val = t.item()

        # Interpolate in latent space
        z_t = interpolate_z(z_a, z_b, t_val)

        # Decode
        commands_t, args_t = decode_z(model, z_t, args.temperature)

        # Save
        svg_t = decode_svg_from_cmd_args(commands_t[0], args_t[0], viewbox_size=args.viewbox_size)
        out_path = out_dir / f"interp_{i:02d}_t{t_val:.2f}.svg"
        svg_t.save_svg(str(out_path))

        if args.save_png:
            svg_to_png(str(out_path), str(out_path.with_suffix(".png")))

    logger.info(f"Saved {args.num_steps} interpolation steps to {out_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
