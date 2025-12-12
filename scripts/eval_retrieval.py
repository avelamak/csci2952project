"""
Retrieval Evaluation Script

Compute Precision@k metrics for latent space retrieval.
Supports same-modal (SVG→SVG) and cross-modal (SVG↔image) retrieval.
Supports JEPA, Contrastive, and Autoencoder encoders.

Precision@k = (# correct in top-k) / k, averaged over all queries.

Usage:
    python scripts/eval_retrieval.py \
        --encoder-type jepa \
        --checkpoint checkpoints/jepa/best_model.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv

    # Or with autoencoder:
    python scripts/eval_retrieval.py \
        --encoder-type autoencoder \
        --checkpoint checkpoints/ae/checkpoint.pt \
        --svg-dir data/fonts_svg --img-dir data/fonts_img --meta data/fonts_meta.csv
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from vecssl.util import setup_logging
from vecssl.data.svg import SVG

from eval_utils import (
    add_common_args,
    create_eval_dataloader,
    load_encoder,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_all_embeddings(
    model,
    loader,
    device: torch.device,
    max_samples: int | None = None,
    cache_clear_interval: int = 50,
) -> tuple[torch.Tensor, torch.Tensor | None, list, list]:
    """
    Compute SVG embeddings in a single pass through the dataset.

    Args:
        model: Encoder model with encode_joint() method
        loader: DataLoader
        device: torch device
        max_samples: if set, stop after roughly this many samples
        cache_clear_interval: clear GPU cache every N batches to prevent fragmentation

    Returns:
        tuple: (z_svg [N, d], z_img (None), labels [N], uuids [N])
    """
    model.eval()
    model.to(device)

    svg_embeddings = []
    # img_embeddings = []
    labels = []
    uuids = []

    try:
        total_samples = len(loader.dataset)
    except Exception:
        total_samples = None

    logger.info(
        f"Computing embeddings: {total_samples if total_samples else 'unknown'} samples, "
        f"batch_size={loader.batch_size}"
    )

    seen = 0
    for batch_idx, batch in enumerate(loader):
        # Move tensors to device
        batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass - get both modalities at once
        joint = model.encode_joint(batch_device)

        # Normalize and move to CPU immediately
        z_svg = F.normalize(joint["svg"], dim=-1).cpu()
        # z_img = F.normalize(joint["img"], dim=-1).cpu()

        svg_embeddings.append(z_svg)
        # img_embeddings.append(z_img)
        labels.extend(batch["label"].tolist())
        uuids.extend(batch["uuid"])

        bsz = z_svg.size(0)
        seen += bsz

        # Progress logging
        if total_samples is not None:
            pct = min(100.0 * seen / total_samples, 100.0)
            if (batch_idx + 1) % 10 == 0 or seen >= total_samples:
                logger.info(f"Embedded {seen}/{total_samples} ({pct:.1f}%) samples")

        # Clear GPU cache periodically to prevent memory fragmentation
        if (batch_idx + 1) % cache_clear_interval == 0:
            torch.cuda.empty_cache()

        # Early stop if max_samples reached
        if max_samples is not None and seen >= max_samples:
            logger.info(f"Reached max_samples={max_samples}, stopping early.")
            break

    z_svg = torch.cat(svg_embeddings, dim=0)
    # z_img = torch.cat(img_embeddings, dim=0)

    logger.info(f"Computed {z_svg.size(0)} embeddings (dim={z_svg.size(1)})")
    return z_svg, None, labels, uuids


def precision_at_k(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: list,
    gallery_labels: list,
    ks: tuple = (1, 5, 10),
    exclude_self: bool = False,
    chunk_size: int = 256,
    return_topk_indices: bool = False,
) -> dict | tuple[dict, list]:
    """
    Compute Precision@k for retrieval in a memory-efficient way by batching queries.

    Precision@k = (# correct in top-k) / k, averaged over all queries.

    Args:
        query_embeddings: [N_q, d] normalized query embeddings
        gallery_embeddings: [N_g, d] normalized gallery embeddings
        query_labels: list[int] of length N_q
        gallery_labels: list[int] of length N_g
        ks: tuple of k values to compute
        exclude_self: if True and N_q == N_g, exclude self-matches for same-modal retrieval
        chunk_size: number of queries per similarity batch
        return_topk_indices: if True, also return top-k gallery indices per query

    Returns:
        dict: {k: precision} for each k
        If return_topk_indices=True, also returns list of (query_idx, [gallery_indices])
    """
    assert query_embeddings.dim() == 2 and gallery_embeddings.dim() == 2
    assert query_embeddings.size(1) == gallery_embeddings.size(1)

    device = query_embeddings.device
    ks = tuple(sorted(ks))
    k_max = max(ks)

    N_q = query_embeddings.size(0)
    N_g = gallery_embeddings.size(0)

    # Make sure labels are indexable lists
    query_labels = list(query_labels)
    gallery_labels = list(gallery_labels)

    # Accumulators for precision
    precision_accum = {k: [] for k in ks}

    # Collect top-k indices per query if requested
    topk_results: list[tuple[int, list[int]]] = []

    # Precompute gallery^T once on the same device as queries
    gallery_embeddings_t = gallery_embeddings.t()  # [d, N_g]

    for start in range(0, N_q, chunk_size):
        end = min(start + chunk_size, N_q)
        bsz = end - start

        # [B, d]
        q_chunk = query_embeddings[start:end]

        # [B, N_g] cosine similarity
        sim_chunk = q_chunk @ gallery_embeddings_t

        # For same-modal retrieval, exclude diagonal self-matches
        if exclude_self and N_q == N_g:
            row_idx = torch.arange(bsz, device=device)
            col_idx = torch.arange(start, end, device=device)
            sim_chunk[row_idx, col_idx] = -1e9  # large negative so never in top-k

        # Top-k over the gallery for each query in the chunk
        topk_idx = sim_chunk.topk(k_max, dim=1).indices.cpu().tolist()

        # Update precision counts
        for row_offset, idxs in enumerate(topk_idx):
            q_idx = start + row_offset
            q_label = query_labels[q_idx]
            if q_label < 0:
                continue  # skip invalid labels

            # Store top-k indices for this query if requested
            if return_topk_indices:
                topk_results.append((q_idx, idxs))

            # Convert indices → labels for this query
            retrieved_labels = [gallery_labels[j] for j in idxs]

            for k in ks:
                # Precision = fraction of top-k with correct label
                correct = sum(1 for lbl in retrieved_labels[:k] if lbl == q_label)
                precision_accum[k].append(correct / k)

    # Average precision over all queries
    precisions = {}
    for k in ks:
        if precision_accum[k]:
            precisions[k] = sum(precision_accum[k]) / len(precision_accum[k])
        else:
            precisions[k] = 0.0

    if return_topk_indices:
        return precisions, topk_results
    return precisions


def tensor_to_svg(pt_path: Path) -> SVG:
    """Load .pt tensor file and reconstruct SVG.

    Uses SVGTensor.from_data() to properly convert raw tensor format
    to the format expected by SVG.from_tensor().
    """
    from vecssl.data.svg_tensor import SVGTensor
    from vecssl.data.geom import Bbox

    res = torch.load(pt_path, weights_only=False)
    t_sep, _ = res["t_sep"], res["fillings"]

    # Convert each path group's raw tensor through SVGTensor
    valid_tensors = []
    for t in t_sep:
        if t.numel() == 0:
            continue
        # SVGTensor.from_data() parses the 14-column format
        # .data property returns tensor in format expected by SVG.from_tensor()
        svg_tensor = SVGTensor.from_data(t, PAD_VAL=-1)
        valid_tensors.append(svg_tensor.data)

    if not valid_tensors:
        return SVG([], viewbox=Bbox(24))

    return SVG.from_tensors(valid_tensors, viewbox=Bbox(24), allow_empty=True)


def save_retrieval_results(
    save_dir: Path,
    svg_dir: Path,
    query_uuids: list,
    gallery_uuids: list,
    topk_per_query: list[tuple[int, list[int]]],
    max_rank: int = 10,
):
    """
    Save retrieved SVGs for visualization.

    For each query, creates a folder with:
        gt.svg - the query SVG
        rank1.svg, rank2.svg, ... - top-k retrieved SVGs
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(topk_per_query)} query results to {save_dir}")

    for i, (query_idx, gallery_indices) in enumerate(topk_per_query):
        query_uuid = query_uuids[query_idx]
        out_dir = save_dir / query_uuid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save query SVG as gt.svg
        query_svg = tensor_to_svg(svg_dir / f"{query_uuid}.pt")
        query_svg.save_svg(out_dir / "gt.svg")

        # Save top-k retrieved SVGs
        for rank, gal_idx in enumerate(gallery_indices[:max_rank], start=1):
            gal_uuid = gallery_uuids[gal_idx]
            gal_svg = tensor_to_svg(svg_dir / f"{gal_uuid}.pt")
            gal_svg.save_svg(out_dir / f"rank{rank}.svg")

        # Progress logging
        if (i + 1) % 100 == 0 or (i + 1) == len(topk_per_query):
            logger.info(f"Saved {i + 1}/{len(topk_per_query)} query results")


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation (Precision@k)")
    parser = add_common_args(parser)

    # Encoder args
    parser.add_argument(
        "--encoder-type",
        type=str,
        required=True,
        choices=["jepa", "contrastive", "autoencoder", "multimae"],
        help="Type of encoder",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to encoder checkpoint",
    )

    # Retrieval args
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="k values for Recall@k",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (for debugging or memory-constrained runs)",
    )
    parser.add_argument(
        "--cache-clear-interval",
        type=int,
        default=50,
        help="Clear GPU cache every N batches to prevent memory fragmentation",
    )
    parser.add_argument(
        "--save-retrieval-dir",
        type=str,
        default=None,
        help="Directory to save retrieved SVGs (reconstructed from .pt tensors). Saves top-10 per query.",
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("Retrieval Evaluation (Precision@k)")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load frozen encoder
    encoder, cfg = load_encoder(Path(args.checkpoint), args.encoder_type)

    # Create dataloader
    loader = create_eval_dataloader(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Compute embeddings for both modalities in a single pass
    logger.info("Computing embeddings (single-pass for both modalities)...")
    z_svg, z_img, labels, uuids = compute_all_embeddings(
        encoder,
        loader,
        device,
        max_samples=args.max_eval_samples,
        cache_clear_interval=args.cache_clear_interval,
    )

    # Note for autoencoder
    # if args.encoder_type == "autoencoder":
    #     logger.info("Note: Autoencoder uses same latent for SVG and image.")
    #     logger.info("      Cross-modal retrieval will be trivial (identical embeddings).")

    ks = tuple(args.ks)
    should_save = args.save_retrieval_dir is not None

    # Same-modal retrieval: SVG → SVG
    logger.info("Computing SVG→SVG retrieval...")
    topk_indices: list[tuple[int, list[int]]] = []
    if should_save:
        precisions_svg2svg, topk_indices = precision_at_k(
            z_svg, z_svg, labels, labels, ks=ks, exclude_self=True, return_topk_indices=True
        )
    else:
        precisions_svg2svg = precision_at_k(
            z_svg, z_svg, labels, labels, ks=ks, exclude_self=True, return_topk_indices=False
        )
    assert isinstance(precisions_svg2svg, dict)  # type guard for Pylance

    logger.info("SVG → SVG Retrieval:")
    for k, v in precisions_svg2svg.items():
        logger.info(f"  Precision@{k}: {v * 100:.2f}%")

    # Save retrieved SVGs if requested
    if should_save:
        save_retrieval_results(
            save_dir=Path(args.save_retrieval_dir),
            svg_dir=Path(args.svg_dir),
            query_uuids=uuids,
            gallery_uuids=uuids,  # same for SVG→SVG
            topk_per_query=topk_indices,
            max_rank=10,
        )

    # Cross-modal retrieval: SVG → Image
    # logger.info("Computing SVG→Image retrieval...")
    # recalls_svg2img = recall_at_k(z_svg, z_img, labels, labels, ks=ks, exclude_self=False)

    # logger.info("SVG → Image Retrieval:")
    # for k, v in recalls_svg2img.items():
    # logger.info(f"  Recall@{k}: {v * 100:.2f}%")

    # Cross-modal retrieval: Image → SVG
    # logger.info("Computing Image→SVG retrieval...")
    # recalls_img2svg = recall_at_k(z_img, z_svg, labels, labels, ks=ks, exclude_self=False)

    # logger.info("Image → SVG Retrieval:")
    # for k, v in recalls_img2svg.items():
    # logger.info(f"  Recall@{k}: {v * 100:.2f}%")

    # Summary
    # logger.info("=" * 60)
    # logger.info("Summary:")
    # logger.info(f"  SVG→SVG   Recall@1: {recalls_svg2svg[1] * 100:.2f}%")
    # logger.info(f"  SVG→Image Recall@1: {recalls_svg2img[1] * 100:.2f}%")
    # logger.info(f"  Image→SVG Recall@1: {recalls_img2svg[1] * 100:.2f}%")
    logger.info("Done.")


if __name__ == "__main__":
    main()
