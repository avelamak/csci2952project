"""
Retrieval Evaluation Script

Compute Recall@k metrics for latent space retrieval.
Supports same-modal (SVG→SVG) and cross-modal (SVG↔image) retrieval.
Supports JEPA, Contrastive, and Autoencoder encoders.

Note: For autoencoder, cross-modal retrieval is trivial (100%) since
the same latent z is used for both SVG and image embeddings.

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

from vecssl.util import setup_logging

from eval_utils import (
    add_common_args,
    compute_embeddings,
    create_eval_dataloader,
    load_encoder,
)

logger = logging.getLogger(__name__)


def recall_at_k(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: list,
    gallery_labels: list,
    ks: tuple = (1, 5, 10),
    exclude_self: bool = False,
) -> dict:
    """
    Compute Recall@k for retrieval.

    Args:
        query_embeddings: [N_q, d] normalized query embeddings
        gallery_embeddings: [N_g, d] normalized gallery embeddings
        query_labels: Labels for queries
        gallery_labels: Labels for gallery
        ks: Tuple of k values to compute
        exclude_self: If True, exclude self-matches (for same-modal)

    Returns:
        dict: {k: recall} for each k
    """
    # Compute cosine similarity
    sim = query_embeddings @ gallery_embeddings.t()  # [N_q, N_g]

    if exclude_self and query_embeddings.shape[0] == gallery_embeddings.shape[0]:
        # Exclude diagonal (self-matches)
        sim.fill_diagonal_(-1e9)

    N = sim.size(0)
    recalls = dict.fromkeys(ks, 0.0)

    for i in range(N):
        query_label = query_labels[i]
        if query_label < 0:
            continue  # Skip invalid labels

        # Get top-k indices
        topk_idx = sim[i].topk(max(ks), dim=0).indices.tolist()

        for k in ks:
            # Check if any of top-k has same label
            retrieved_labels = [gallery_labels[j] for j in topk_idx[:k]]
            hits = sum(1 for lbl in retrieved_labels if lbl == query_label)
            recalls[k] += 1.0 if hits > 0 else 0.0

    # Normalize
    valid_count = sum(1 for lbl in query_labels if lbl >= 0)
    for k in ks:
        recalls[k] /= max(1, valid_count)

    return recalls


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation (Recall@k)")
    parser = add_common_args(parser)

    # Encoder args
    parser.add_argument(
        "--encoder-type",
        type=str,
        required=True,
        choices=["jepa", "contrastive", "autoencoder"],
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

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("Retrieval Evaluation (Recall@k)")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load frozen encoder
    encoder, cfg = load_encoder(Path(args.checkpoint), args.encoder_type, device)

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

    # Compute embeddings for both modalities
    logger.info("Computing SVG embeddings...")
    z_svg, labels_svg, uuids_svg = compute_embeddings(encoder, loader, device, modality="svg")

    logger.info("Computing image embeddings...")
    z_img, labels_img, uuids_img = compute_embeddings(encoder, loader, device, modality="img")

    # Note for autoencoder
    if args.encoder_type == "autoencoder":
        logger.info("Note: Autoencoder uses same latent for SVG and image.")
        logger.info("      Cross-modal retrieval will be trivial (identical embeddings).")

    ks = tuple(args.ks)

    # Same-modal retrieval: SVG → SVG
    logger.info("Computing SVG→SVG retrieval...")
    recalls_svg2svg = recall_at_k(z_svg, z_svg, labels_svg, labels_svg, ks=ks, exclude_self=True)

    logger.info("SVG → SVG Retrieval:")
    for k, v in recalls_svg2svg.items():
        logger.info(f"  Recall@{k}: {v * 100:.2f}%")

    # Cross-modal retrieval: SVG → Image
    logger.info("Computing SVG→Image retrieval...")
    recalls_svg2img = recall_at_k(z_svg, z_img, labels_svg, labels_img, ks=ks, exclude_self=False)

    logger.info("SVG → Image Retrieval:")
    for k, v in recalls_svg2img.items():
        logger.info(f"  Recall@{k}: {v * 100:.2f}%")

    # Cross-modal retrieval: Image → SVG
    logger.info("Computing Image→SVG retrieval...")
    recalls_img2svg = recall_at_k(z_img, z_svg, labels_img, labels_svg, ks=ks, exclude_self=False)

    logger.info("Image → SVG Retrieval:")
    for k, v in recalls_img2svg.items():
        logger.info(f"  Recall@{k}: {v * 100:.2f}%")

    # Summary
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  SVG→SVG   Recall@1: {recalls_svg2svg[1] * 100:.2f}%")
    logger.info(f"  SVG→Image Recall@1: {recalls_svg2img[1] * 100:.2f}%")
    logger.info(f"  Image→SVG Recall@1: {recalls_img2svg[1] * 100:.2f}%")
    logger.info("Done.")


if __name__ == "__main__":
    main()
