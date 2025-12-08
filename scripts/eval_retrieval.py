"""
Retrieval Evaluation Script

Compute Recall@k metrics for latent space retrieval.
Supports same-modal (SVG→SVG) and cross-modal (SVG↔image) retrieval.
Supports JEPA, Contrastive, and Autoencoder encoders.

Note: For autoencoder, cross-modal retrieval is trivial (100%) since
the same latent z is used for both SVG and image embeddings.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from vecssl.util import setup_logging

from eval_utils import (
    add_common_args,
    compute_embeddings,  # kept for compatibility, not used here
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
    chunk_size: int = 256,
) -> dict:
    """
    Compute Recall@k for retrieval in a memory-efficient way by batching queries.
    """
    assert query_embeddings.dim() == 2 and gallery_embeddings.dim() == 2
    assert query_embeddings.size(1) == gallery_embeddings.size(1)

    device = query_embeddings.device
    ks = tuple(sorted(ks))
    k_max = max(ks)

    N_q = query_embeddings.size(0)
    N_g = gallery_embeddings.size(0)

    query_labels = list(query_labels)
    gallery_labels = list(gallery_labels)

    hit_counts = dict.fromkeys(ks, 0.0)
    valid_count = sum(1 for lbl in query_labels if lbl >= 0)

    gallery_embeddings_t = gallery_embeddings.t()  # [d, N_g]

    for start in range(0, N_q, chunk_size):
        end = min(start + chunk_size, N_q)
        bsz = end - start

        q_chunk = query_embeddings[start:end]  # [B, d]
        sim_chunk = q_chunk @ gallery_embeddings_t  # [B, N_g]

        if exclude_self and N_q == N_g:
            row_idx = torch.arange(bsz, device=device)
            col_idx = torch.arange(start, end, device=device)
            sim_chunk[row_idx, col_idx] = -1e9

        topk_idx = sim_chunk.topk(k_max, dim=1).indices.cpu().tolist()

        for row_offset, idxs in enumerate(topk_idx):
            q_idx = start + row_offset
            q_label = query_labels[q_idx]
            if q_label < 0:
                continue

            retrieved_labels = [gallery_labels[j] for j in idxs]

            for k in ks:
                if any(lbl == q_label for lbl in retrieved_labels[:k]):
                    hit_counts[k] += 1.0

    recalls = {}
    denom = max(1, valid_count)
    for k in ks:
        recalls[k] = hit_counts[k] / denom

    return recalls


@torch.no_grad()
def compute_embeddings_with_progress(
    model,
    loader,
    device: torch.device,
    modality: str = "svg",
    max_eval_samples: int | None = None,
    log_every: int = 1,
) -> tuple[torch.Tensor, list, list]:
    """
    Compute embeddings for entire dataset, with progress logging
    and optional early stop to avoid OOM.

    Args:
        model: Encoder model with encode_joint() method
        loader: DataLoader
        device: torch device
        modality: "svg" or "img" - which embedding to return
        max_eval_samples: if set, stop after roughly this many samples
        log_every: log progress every N batches

    Returns:
        tuple: (embeddings [N, d], labels [N], uuids [N])
    """
    model.eval()
    embeddings = []
    labels = []
    uuids = []

    try:
        total_samples = len(loader.dataset)
    except Exception:
        total_samples = None

    logger.info(
        f"[{modality}] Starting embedding: "
        f"{total_samples if total_samples is not None else 'unknown'} samples, "
        f"batch_size={loader.batch_size}"
    )

    seen = 0

    for batch_idx, batch in enumerate(loader):
        # Move tensors to device
        batch_device = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        # Forward
        joint = model.encode_joint(batch_device)
        z = joint[modality]  # [B, d_joint]
        z = F.normalize(z, dim=-1)

        embeddings.append(z.cpu())
        labels.extend(batch["label"].tolist())
        uuids.extend(batch["uuid"])

        bsz = z.size(0)
        seen += bsz

        # Progress logging
        if total_samples is not None:
            pct = min(100.0 * seen / total_samples, 100.0)
            if (batch_idx + 1) % log_every == 0 or seen >= total_samples:
                logger.info(
                    f"[{modality}] Embedded {seen}/{total_samples} "
                    f"({pct:.1f}%) samples"
                )

        # Early stop to control memory usage
        if max_eval_samples is not None and seen >= max_eval_samples:
            logger.info(
                f"[{modality}] Reached max_eval_samples={max_eval_samples}, "
                "stopping early."
            )
            break

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, labels, uuids


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

    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help=(
            "Optionally cap number of eval samples to avoid OOM. "
            "If set, stop embedding after roughly this many samples."
        ),
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level, rich_tracebacks=True)
    logger.info("=" * 60)
    logger.info("Retrieval Evaluation (Recall@k)")
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

    # Compute embeddings for both modalities (with progress + max-eval-samples)
    logger.info("Computing SVG embeddings...")
    z_svg, labels_svg, uuids_svg = compute_embeddings_with_progress(
        encoder,
        loader,
        device,
        modality="svg",
        max_eval_samples=args.max_eval_samples,
        log_every=1,  # 每个 batch 打一次 log
    )

    logger.info("Computing image embeddings...")
    z_img, labels_img, uuids_img = compute_embeddings_with_progress(
        encoder,
        loader,
        device,
        modality="img",
        max_eval_samples=args.max_eval_samples,
        log_every=1,
    )

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
        logger.info(f"  SVG→Image Recall@{k}: {v * 100:.2f}%")

    # Cross-modal retrieval: Image → SVG
    logger.info("Computing Image→SVG retrieval...")
    recalls_img2svg = recall_at_k(z_img, z_svg, labels_img, labels_svg, ks=ks, exclude_self=False)

    logger.info("Image → SVG Retrieval:")
    for k, v in recalls_img2svg.items():
        logger.info(f"  Image→SVG Recall@{k}: {v * 100:.2f}%")

    # Summary
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  SVG→SVG   Recall@1: {recalls_svg2svg[1] * 100:.2f}%")
    logger.info(f"  SVG→Image Recall@1: {recalls_svg2img[1] * 100:.2f}%")
    logger.info(f"  Image→SVG Recall@1: {recalls_img2svg[1] * 100:.2f}%")
    logger.info("Done.")


if __name__ == "__main__":
    main()
