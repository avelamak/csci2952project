"""
Latent Interpolation Script (Universal)

Visualize latent space by interpolating between two SVGs and decoding the results.
Supports: Autoencoder (AE), JEPA, Contrastive.

Usage:
    python scripts/eval_interpolation_universal.py \
        --ckpt checkpoints/best_model.pt \
        --model-type ae \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv \
        --idx-a 0 --idx-b 100 --num-steps 10 --out-dir interp_output
"""

import argparse
import logging
from pathlib import Path
import numpy as np

import torch

from vecssl.data.dataset import SVGXDataset
from vecssl.data.geom import Bbox
from vecssl.data.svg import SVG
from vecssl.data.svg_tensor import SVGTensor
from vecssl.util import setup_logging

# 1. Import all model classes
from vecssl.models.model import SVGTransformer
from vecssl.models.config import _DefaultConfig

# Try importing JEPA and Contrastive (prevent missing environment errors)
try:
    from vecssl.models.jepa import Jepa
    from vecssl.models.config import JepaConfig
except ImportError:
    Jepa = None
    JepaConfig = None

try:
    from vecssl.models.contrastive import ContrastiveModel
    from vecssl.models.config import ContrastiveConfig
except ImportError:
    ContrastiveModel = None
    ContrastiveConfig = None

logger = logging.getLogger(__name__)


def load_config_to_obj(cfg_cls, cfg_source):
    """Helper to convert dictionary/object config to specific Config class."""
    cfg = cfg_cls()
    
    # If source is an object, convert to dict
    if hasattr(cfg_source, "__dict__"):
        source_dict = cfg_source.__dict__
    elif isinstance(cfg_source, dict):
        source_dict = cfg_source
    else:
        return cfg # Empty default

    for k, v in source_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_model(checkpoint_path: Path, model_type: str, device: torch.device):
    """
    Load model based on type (ae, jepa, contrastive).
    """
    logger.info(f"Loading {model_type} model from {checkpoint_path}")
    # weights_only=False allows loading arbitrary objects like configs
    ckpt = torch.load(checkpoint_path, map_location=device) 
    
    cfg_obj = ckpt.get("cfg", {})

    # === Instantiate Model ===
    if model_type == "ae":
        cfg = load_config_to_obj(_DefaultConfig, cfg_obj)
        # Ensure necessary parameters for AE exist
        if not hasattr(cfg, "encode_stages"): cfg.encode_stages = 2
        if not hasattr(cfg, "decode_stages"): cfg.decode_stages = 2
        if not hasattr(cfg, "use_vae"): cfg.use_vae = True
        model = SVGTransformer(cfg)

    elif model_type == "jepa":
        if Jepa is None: raise ImportError("Jepa code not found in vecssl.models")
        cfg = load_config_to_obj(JepaConfig, cfg_obj)
        model = Jepa(cfg)

    elif model_type == "contrastive":
        if ContrastiveModel is None: raise ImportError("Contrastive code not found in vecssl.models")
        cfg = load_config_to_obj(ContrastiveConfig, cfg_obj)
        model = ContrastiveModel(cfg)

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choices: ae, jepa, contrastive")

    # === Load Weights ===
    state_dict = ckpt["model"]
    # Handle 'model.' prefix from DDP training
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
        }
    
    # strict=False because some models might have unused keys (e.g., loss params)
    keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Model loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")

    model.to(device)
    model.eval()

    return model, cfg


def decode_svg_from_cmd_args(commands, args, viewbox_size=256, pad_val=-1) -> SVG:
    """Decode commands and args tensors back to an SVG object."""
    if commands.ndim == 2:
        commands = commands[0]
        args = args[0]

    commands = commands.detach().cpu()
    args = args.detach().cpu().float()

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
def encode_sample(model, item, device, model_type="ae"):
    """
    Encode a single sample to latent z.
    Adapts to different model APIs.
    """
    commands = item["commands"].unsqueeze(0).to(device) # [1, G, S]
    args = item["args"].unsqueeze(0).to(device) # [1, G, S, n_args]

    # === Use forward for AE ===
    if model_type == "ae":
        z = model.forward(
            commands_enc=commands,
            args_enc=args,
            commands_dec=None,
            args_dec=None,
            encode_mode=True,
        )
    
    # === Use encode_joint for JEPA / Contrastive ===
    elif model_type in ["jepa", "contrastive"]:
        # Construct batch, because encode_joint requires a dict
        # Note: If dataset has no image, create a dummy image to prevent errors
        # (Even if we only use SVG embedding, the code might expect the key to exist)
        dummy_img = torch.zeros(1, 3, 224, 224).to(device)
        
        batch = {
            "commands": commands,
            "args": args,
            "image": item.get("image", dummy_img).unsqueeze(0).to(device) if "image" in item else dummy_img
        }
        
        # encode_joint returns {"svg": ..., "img": ...}
        out = model.encode_joint(batch)
        z = out["svg"] # [1, dim]

        # If z is [1, dim], unsqueeze to [1, 1, dim] for consistency with interpolation logic
        if z.ndim == 2:
            z = z.unsqueeze(1)
            
    else:
        raise ValueError(f"Encoding not implemented for {model_type}")

    return z


@torch.no_grad()
def decode_z(model, z):
    """
    Decode latent z to commands and args.
    """
    # Check if model has decoding capability
    if not hasattr(model, "greedy_sample"):
        # If JEPA/Contrastive (no decoder), return None to avoid errors
        return None, None

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

    # Model args (CHANGED)
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--model-type", type=str, default="ae", choices=["ae", "jepa", "contrastive"], help="Type of model to load")

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
    logger.info(f"Latent Interpolation ({args.model_type.upper()})")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, cfg = load_model(Path(args.ckpt), args.model_type, device)

    # Load dataset
    dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        seed=42,
        already_preprocessed=True,
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Interpolating between idx {args.idx_a} and {args.idx_b}")

    item_a = dataset[args.idx_a]
    item_b = dataset[args.idx_b]

    # Encode
    z_a = encode_sample(model, item_a, device, args.model_type)
    z_b = encode_sample(model, item_b, device, args.model_type)

    logger.info(f"Encoded z_a shape: {z_a.shape}")
    logger.info(f"Encoded z_b shape: {z_b.shape}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Interpolation loop
    for i, t in enumerate(torch.linspace(0.0, 1.0, args.num_steps)):
        z_t = (1 - t) * z_a + t * z_b
        
        commands_t, args_t = decode_z(model, z_t)
        
        if commands_t is None:
            if i == 0:
                logger.warning(f"Model type '{args.model_type}' does not have a decoder. Cannot visualize SVGs.")
                logger.warning("Interpolation calculated in latent space, but no SVG output will be generated.")
            break

        svg_t = decode_svg_from_cmd_args(commands_t[0], args_t[0])
        out_path = out_dir / f"interp_{i:02d}_t{t:.2f}.svg"
        svg_t.save_svg(str(out_path))

    if args.model_type == "ae":
        logger.info(f"Saved {args.num_steps} interpolation steps to {out_dir}")
    else:
        logger.info("Done (No SVGs saved for encoder-only models).")

if __name__ == "__main__":
    main()
