"""
Latent Interpolation Script

Visualize latent space by interpolating between two SVGs and decoding the results.

Usage:
    python scripts/eval_interpolation.py \
        --ae-ckpt checkpoints/decoder/best_model.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv \
        --idx-a 0 --idx-b 100 --num-steps 10 --out-dir interp_output
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

logger = logging.getLogger(__name__)


def load_autoencoder(checkpoint_path: Path, device: torch.device):
    """Load autoencoder from checkpoint."""
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

    model = SVGTransformer(cfg)

    # Handle wrapped model
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

    # Find actual sequence length (before EOS/padding)
    eos_idx = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
    valid_mask = (commands != pad_val) & (commands != eos_idx)

    if valid_mask.any():
        seq_len = valid_mask.long().sum().item()
    else:
        seq_len = 0

    if seq_len == 0:
        return SVG([], viewbox=Bbox(viewbox_size))

    commands = commands[:seq_len]
    args = args[:seq_len]

    try:
        svg_tensor = SVGTensor.from_cmd_args(commands, args, PAD_VAL=pad_val)
        tensor_data = svg_tensor.data
        svg = SVG.from_tensor(tensor_data, viewbox=Bbox(viewbox_size), allow_empty=True)
    except Exception as e:
        logger.warning(f"Failed to decode SVG: {e}")
        svg = SVG([], viewbox=Bbox(viewbox_size))

    return svg


@torch.no_grad()
def encode_sample(model, item, device):
    """
    Encode a single sample to latent z.

    Returns z in the shape expected by greedy_sample.
    """
    commands = item["commands"].unsqueeze(0).to(device)  # [1, G, S]
    args = item["args"].unsqueeze(0).to(device)  # [1, G, S, n_args]

    # Get z via encode_mode
    z = model.forward(
        commands_enc=commands,
        args_enc=args,
        commands_dec=None,
        args_dec=None,
        encode_mode=True,
    )

    # z is in seq-first format [1, B, dim_z], keep as-is for greedy_sample
    return z


@torch.no_grad()
def decode_z(model, z):
    """
    Decode latent z to commands and args.

    Args:
        z: Latent tensor in format expected by greedy_sample

    Returns:
        tuple: (commands, args) tensors
    """
    commands_y, args_y = model.greedy_sample(
        commands_enc=None,
        args_enc=None,
        z=z,
        concat_groups=True,
        temperature=0.0001,
    )
    return commands_y, args_y


def main():
    parser = argparse.ArgumentParser(description="Latent interpolation visualization")

    # Dataset args
    parser.add_argument("--svg-dir", type=str, required=True, help="SVG directory")
    parser.add_argument("--img-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--meta", type=str, required=True, help="Metadata CSV")
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max path groups")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")

    # Model args
    parser.add_argument("--ae-ckpt", type=str, required=True, help="Autoencoder checkpoint")

    # Interpolation args
    parser.add_argument("--idx-a", type=int, required=True, help="Index of first SVG")
    parser.add_argument("--idx-b", type=int, required=True, help="Index of second SVG")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of interpolation steps")
    parser.add_argument("--out-dir", type=str, default="interp_svgs", help="Output directory")

    # Runtime args
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("Latent Interpolation")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, cfg = load_autoencoder(Path(args.ae_ckpt), device)

    # Load dataset using test split for proper evaluation
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

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Interpolating between idx {args.idx_a} and {args.idx_b}")

    # Get samples
    item_a = dataset[args.idx_a]
    item_b = dataset[args.idx_b]

    logger.info(f"Sample A: {item_a.get('name', item_a['uuid'])}")
    logger.info(f"Sample B: {item_b.get('name', item_b['uuid'])}")

    # Encode
    z_a = encode_sample(model, item_a, device)
    z_b = encode_sample(model, item_b, device)

    logger.info(f"Encoded z_a shape: {z_a.shape}")
    logger.info(f"Encoded z_b shape: {z_b.shape}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save endpoints
    for tag, z, item in [("a", z_a, item_a), ("b", z_b, item_b)]:
        commands_y, args_y = decode_z(model, z)
        svg = decode_svg_from_cmd_args(commands_y[0], args_y[0])
        out_path = out_dir / f"interp_{tag}.svg"
        svg.save_svg(str(out_path))
        logger.info(f"Saved {out_path}")

    # Linear interpolation
    for i, t in enumerate(torch.linspace(0.0, 1.0, args.num_steps)):
        z_t = (1 - t) * z_a + t * z_b
        commands_t, args_t = decode_z(model, z_t)
        svg_t = decode_svg_from_cmd_args(commands_t[0], args_t[0])

        out_path = out_dir / f"interp_{i:02d}_t{t:.2f}.svg"
        svg_t.save_svg(str(out_path))

    logger.info(f"Saved {args.num_steps} interpolation steps to {out_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
