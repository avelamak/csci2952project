"""
DecoderFromMultiMAE Reconstruction Evaluation Script

Evaluate reconstruction quality from DecoderFromMultiMAE checkpoints
(trained via train_decoder_from_multimae.py).

Flow:
    1. Encode SVG+image via frozen MultiMAE encoder -> CLS embedding
    2. Decode via SVGTransformer -> reconstructed SVG
    3. Compare against ground truth

Usage (single sample):
    python scripts/eval_reconstruction_multimae.py \
        --ckpt checkpoints/multimae_decoder/checkpoint.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv \
        --idx 0 --out-dir recon_multimae --save-png

Usage (batch mode with metrics):
    python scripts/eval_reconstruction_multimae.py \
        --ckpt checkpoints/multimae_decoder/checkpoint.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv \
        --out-dir recon_multimae --batch-size 32

Usage (.pt mode):
    python scripts/eval_reconstruction_multimae.py \
        --ckpt checkpoints/multimae_decoder/checkpoint.pt \
        --pt sample.pt --img sample.png \
        --out-dir recon_multimae --save-png
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
from eval_reconstruction import decode_svg_from_cmd_args
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
        decoder_cfg.encode_stages = 0
        decoder_cfg.decode_stages = 2
        decoder_cfg.use_vae = False
        decoder_cfg.pred_mode = "one_shot"
        decoder_cfg.max_seq_len = 40  # !

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


def _decode_svg_from_cmd_args(commands, args, viewbox_size=256, pad_val=-1) -> SVG:
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


def custom_collate(batch):
    """Standard collate for SVGXDataset."""
    collated = {
        "commands": torch.stack([item["commands"] for item in batch]),
        "args": torch.stack([item["args"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "tensors": [item["tensors"] for item in batch],
        "uuid": [item["uuid"] for item in batch],
        "name": [item["name"] for item in batch],
        "source": [item["source"] for item in batch],
    }

    if "dino_embedding" in batch[0]:
        collated["dino_embedding"] = torch.stack([item["dino_embedding"] for item in batch])
    if "dino_patches" in batch[0]:
        collated["dino_patches"] = torch.stack([item["dino_patches"] for item in batch])

    return collated


# =============================================================================
# Encoding and Decoding
# =============================================================================


@torch.no_grad()
def encode_batch(model: DecoderFromMultiMAE, batch: dict, device: torch.device) -> torch.Tensor:
    """
    Encode batch to latent z using frozen MultiMAE encoder.

    Args:
        model: DecoderFromMultiMAE model
        batch: Batch dict with commands, args, and image/dino_patches
        device: torch device

    Returns:
        z: Latent tensor [N, 1, 1, d_model] ready for svg_transformer.greedy_sample()
    """
    batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Get CLS embedding from frozen MultiMAE
    joint = model.encoder.encode_joint(batch_device)
    cls_embed = joint["svg"]  # [N, d_model]

    # Reshape to match decoder expected input: (N, d_model) -> (N, 1, 1, d_model)
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
        concat_groups=False,
        temperature=temperature,
    )
    return commands_y, args_y


@torch.no_grad()
def reconstruct_batch(
    model: DecoderFromMultiMAE, batch: dict, device: torch.device, temperature: float = 0.0001
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode and decode a batch, return predictions and targets.

    Returns:
        tuple: (pred_cmds, pred_args, tgt_cmds, tgt_args)
    """
    z = encode_batch(model, batch, device)
    commands_y, args_y = decode_z(model, z, temperature)

    return commands_y.cpu(), args_y.cpu(), batch["commands"].cpu(), batch["args"].cpu()


# =============================================================================
# Metrics
# =============================================================================


def command_accuracy(pred_cmds: torch.Tensor, tgt_cmds: torch.Tensor) -> float:
    """
    Compute command token accuracy over real drawing commands (m, l, c, a).

    COMMANDS_SIMPLIFIED = ["m", "l", "c", "a", "EOS", "SOS", "z"]
    Real commands have indices 0-3 (< EOS index).

    Args:
        pred_cmds: [B, S] or [B, G, S] predicted commands
        tgt_cmds: [B, S] or [B, G, S] target commands

    Returns:
        float: Accuracy
    """
    # Flatten if needed
    if pred_cmds.ndim == 3:
        pred_cmds = pred_cmds.reshape(pred_cmds.size(0), -1)
    if tgt_cmds.ndim == 3:
        tgt_cmds = tgt_cmds.reshape(tgt_cmds.size(0), -1)

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


def arg_errors(pred_args: torch.Tensor, tgt_args: torch.Tensor, pad_val: int = -1) -> tuple:
    """
    Compute argument L1 and L2 errors.

    Args:
        pred_args: predicted arguments
        tgt_args: target arguments
        pad_val: Padding value to ignore

    Returns:
        tuple: (L1 error, L2 error)
    """
    # Flatten if needed
    if pred_args.ndim == 4:
        pred_args = pred_args.reshape(pred_args.size(0), -1, pred_args.size(-1))
    if tgt_args.ndim == 4:
        tgt_args = tgt_args.reshape(tgt_args.size(0), -1, tgt_args.size(-1))

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


def save_reconstruction_results(
    save_dir: Path,
    uuids: list,
    pred_cmds: torch.Tensor,
    pred_args: torch.Tensor,
    tgt_cmds: torch.Tensor,
    tgt_args: torch.Tensor,
    viewbox_size: int = 256,
    save_png: bool = False,
):
    """Save ground truth and reconstructed SVGs."""
    save_dir.mkdir(parents=True, exist_ok=True)

    B = pred_cmds.size(0)
    for i in range(B):
        uuid = uuids[i]
        out_dir = save_dir / uuid
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            gt_svg = decode_svg_from_cmd_args(tgt_cmds[i], tgt_args[i], viewbox_size)
            gt_path = out_dir / "gt.svg"
            gt_svg.save_svg(str(gt_path))
            if save_png:
                svg_to_png(str(gt_path), str(gt_path.with_suffix(".png")))
        except Exception as e:
            logger.warning(f"Failed to save GT SVG for {uuid}: {e}")

        try:
            recon_svg = decode_svg_from_cmd_args(pred_cmds[i], pred_args[i], viewbox_size)
            recon_path = out_dir / "recon.svg"
            recon_svg.save_svg(str(recon_path))
            if save_png:
                svg_to_png(str(recon_path), str(recon_path.with_suffix(".png")))
        except Exception as e:
            logger.warning(f"Failed to save recon SVG for {uuid}: {e}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="DecoderFromMultiMAE Reconstruction Evaluation")

    # Dataset mode
    parser.add_argument("--svg-dir", type=str, default=None, help="SVG directory")
    parser.add_argument("--img-dir", type=str, default=None, help="Image directory")
    parser.add_argument("--meta", type=str, default=None, help="Metadata CSV")
    parser.add_argument("--max-num-groups", type=int, default=8, help="Max path groups")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Max sequence length")
    parser.add_argument(
        "--idx", type=int, default=None, help="Dataset index (single sample mode, omit for batch)"
    )

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
    parser.add_argument(
        "--ckpt", type=str, required=True, help="DecoderFromMultiMAE checkpoint path"
    )
    parser.add_argument("--out-dir", type=str, default="recon_multimae", help="Output directory")
    parser.add_argument("--viewbox-size", type=int, default=256, help="SVG viewbox size")
    parser.add_argument(
        "--save-reconstruction-dir",
        type=str,
        default=None,
        help="Directory to save reconstructed SVGs (batch mode)",
    )

    # Batch mode args
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (batch mode)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--save-png", action="store_true", help="Also save PNG versions")
    parser.add_argument("--temperature", type=float, default=0.0001, help="Sampling temperature")

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("DecoderFromMultiMAE Reconstruction Evaluation")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, cfg = load_decoder_from_multimae(Path(args.ckpt), device)

    # Determine mode
    use_pt = args.pt is not None
    use_dataset = args.svg_dir is not None and args.meta is not None
    single_sample_mode = args.idx is not None

    if not (use_pt ^ use_dataset):
        raise ValueError(
            "Specify exactly one of: "
            "(--pt [and optionally --img]) or "
            "(--svg-dir/--meta for dataset mode)."
        )

    # =========================================================================
    # .pt file mode (single sample)
    # =========================================================================
    if use_pt:
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
            logger.warning(
                "No --img provided for .pt sample. This will fail if model expects an image."
            )

        name = item.get("name", Path(args.pt).stem)
        logger.info(f"Loaded .pt sample: {name}")

        # Build batch
        batch = {
            "commands": item["commands"].unsqueeze(0),
            "args": item["args"].unsqueeze(0),
        }
        if "image" in item:
            batch["image"] = item["image"].unsqueeze(0)

        # Reconstruct
        pred_cmds, pred_args, tgt_cmds, tgt_args = reconstruct_batch(
            model, batch, device, args.temperature
        )

        # Save results
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Ground truth SVG
        svg_gt = decode_svg_from_cmd_args(tgt_cmds[0], tgt_args[0], args.viewbox_size)
        gt_path = out_dir / f"{name}_gt.svg"
        svg_gt.save_svg(str(gt_path))
        logger.info(f"Saved ground truth SVG to {gt_path}")
        if args.save_png:
            svg_to_png(str(gt_path), str(gt_path.with_suffix(".png")))

        # Reconstructed SVG
        svg_recon = decode_svg_from_cmd_args(pred_cmds[0], pred_args[0], args.viewbox_size)
        recon_path = out_dir / f"{name}_recon.svg"
        svg_recon.save_svg(str(recon_path))
        logger.info(f"Saved reconstructed SVG to {recon_path}")
        if args.save_png:
            svg_to_png(str(recon_path), str(recon_path.with_suffix(".png")))

        logger.info("Done.")
        return

    # =========================================================================
    # Dataset mode
    # =========================================================================
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

    # -------------------------------------------------------------------------
    # Single sample mode (--idx provided)
    # -------------------------------------------------------------------------
    if single_sample_mode:
        item = dataset[args.idx]
        name = item.get("name", item.get("uuid", f"idx{args.idx}"))
        logger.info(f"Loaded dataset sample: {name} (idx {args.idx})")

        # Build batch
        batch = {
            "commands": item["commands"].unsqueeze(0),
            "args": item["args"].unsqueeze(0),
        }
        if "dino_patches" in item:
            batch["dino_patches"] = item["dino_patches"].unsqueeze(0)
        elif "image" in item:
            batch["image"] = item["image"].unsqueeze(0)
        else:
            raise ValueError("Sample has no image or dino_patches")

        # Reconstruct
        pred_cmds, pred_args, tgt_cmds, tgt_args = reconstruct_batch(
            model, batch, device, args.temperature
        )

        # Compute single-sample metrics
        ca = command_accuracy(pred_cmds, tgt_cmds)
        l1, l2 = arg_errors(pred_args, tgt_args)
        logger.info(f"Command accuracy: {ca:.4f}")
        logger.info(f"Args L1 error: {l1:.4f}")
        logger.info(f"Args L2 error: {l2:.4f}")

        # Save results
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        svg_gt = decode_svg_from_cmd_args(tgt_cmds[0], tgt_args[0], args.viewbox_size)
        gt_path = out_dir / f"{name}_gt.svg"
        svg_gt.save_svg(str(gt_path))
        logger.info(f"Saved ground truth SVG to {gt_path}")
        if args.save_png:
            svg_to_png(str(gt_path), str(gt_path.with_suffix(".png")))

        svg_recon = decode_svg_from_cmd_args(pred_cmds[0], pred_args[0], args.viewbox_size)
        recon_path = out_dir / f"{name}_recon.svg"
        svg_recon.save_svg(str(recon_path))
        logger.info(f"Saved reconstructed SVG to {recon_path}")
        if args.save_png:
            svg_to_png(str(recon_path), str(recon_path.with_suffix(".png")))

        logger.info("Done.")
        return

    # -------------------------------------------------------------------------
    # Batch mode (no --idx, process full test set)
    # -------------------------------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        drop_last=False,
    )

    logger.info(f"Evaluating on {len(dataset)} samples (batch_size={args.batch_size})...")

    # Accumulate metrics
    total_cmd_acc = 0.0
    total_l1 = 0.0
    total_l2 = 0.0
    n_batches = 0

    should_save = args.save_reconstruction_dir is not None

    for batch in loader:
        pred_cmds, pred_args, tgt_cmds, tgt_args = reconstruct_batch(
            model, batch, device, args.temperature
        )

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
                viewbox_size=args.viewbox_size,
                save_png=args.save_png,
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
    logger.info("Done.")


if __name__ == "__main__":
    main()
