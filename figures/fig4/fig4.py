import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse
import numpy as np

from torch.utils.data import DataLoader
from vecssl.data.dataset import SVGXDataset
from vecssl.util import setup_logging

import logging

logger = logging.getLogger(__name__)


# ===============================================================
#   EMBEDDINGS
# ===============================================================
@torch.no_grad()
def get_embeddings(model, dataloader):
    model.eval()
    all_svg, all_img = [], []

    for i, batch in enumerate(dataloader):
        out = model.encode_joint(batch)

        if "svg" in out:
            all_svg.append(out["svg"].cpu())
        if "img" in out:
            all_img.append(out["img"].cpu())

        if i == 0:  # Quick test
            break

    return (
        torch.cat(all_svg, dim=0) if all_svg else None,
        torch.cat(all_img, dim=0) if all_img else None,
    )


# ===============================================================
#   PCA + Plot
# ===============================================================
def run_pca(embeddings, tag):
    x = embeddings.numpy()

    pca = PCA(n_components=min(32, x.shape[1]))
    pca.fit(x)

    explained = pca.explained_variance_ratio_
    logging.info(f"\n====== PCA RESULTS ({tag}) ======")
    logging.info("Top 10 explained variance ratios:")
    logging.info(explained[:10])

    collapse_score = explained[0] / explained.sum()
    logging.info(f"Collapse Score = {collapse_score:.4f} (≈1.0 means collapsed)")

    # # Individual plot
    # plt.figure(figsize=(6, 4))
    # plt.plot(np.cumsum(explained))
    # plt.xlabel("Principal Component")
    # plt.ylabel("Cumulative Variance Explained")
    # plt.title(f"PCA: {tag}")
    # plt.grid()

    # plot_path = f"pca_{tag}.png"
    # plt.savefig(plot_path)
    # logging.info(f"[saved] {plot_path}")

    return explained


# ===============================================================
#   COLLATE
# ===============================================================
def custom_collate(batch):
    """Custom collate function that handles SVGTensor objects."""
    collated = {}

    collated["commands"] = torch.stack([item["commands"] for item in batch])
    collated["args"] = torch.stack([item["args"] for item in batch])
    collated["image"] = torch.stack([item["image"] for item in batch])

    collated["tensors"] = [item["tensors"] for item in batch]
    collated["uuid"] = [item["uuid"] for item in batch]
    collated["name"] = [item["name"] for item in batch]
    collated["source"] = [item["source"] for item in batch]

    return collated


# ===============================================================
#   DATA
# ===============================================================
def create_dataloaders(args, max_samples=None):
    logger.info("Creating datasets...")

    dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=8,
        max_seq_len=30,
        train_ratio=1.0,
        already_preprocessed=True,
    )

    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))

    logger.info(f"  Val samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate,
        drop_last=False,
    )

    return loader


# ===============================================================
#   MODEL LOADER
# ===============================================================
def load_model_by_type(model_type, device):
    model_type = model_type.lower()

    if model_type == "jepa":
        from vecssl.models.jepa import Jepa
        from vecssl.models.config import JepaConfig

        cfg = JepaConfig()
        return Jepa(cfg).to(device)

    elif model_type == "contrastive":
        from vecssl.models.contrastive import ContrastiveModel
        from vecssl.models.config import ContrastiveConfig

        cfg = ContrastiveConfig()
        return ContrastiveModel(cfg).to(device)

    elif model_type == "multimae":
        from vecssl.models.multimae import MultiMAE
        from vecssl.models.config import MultiMAEConfig

        cfg = MultiMAEConfig()
        return MultiMAE(cfg).to(device)

    elif model_type == "svgae":
        import sys
        import os

        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
        from scripts.test_svg_autoencoder import SimpleSVGAutoencoder
        from vecssl.models.config import _DefaultConfig

        cfg = _DefaultConfig()
        return SimpleSVGAutoencoder(cfg).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ===============================================================
#   MAIN
# ===============================================================
def main():
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="data/fonts/svg")
    parser.add_argument("--img-dir", type=str, default="data/fonts/img")
    parser.add_argument("--meta", type=str, default="data/fonts/metadata.csv")

    # Manual model specification
    parser.add_argument("--model-checkpoints", nargs="+", required=True)
    parser.add_argument("--model-types", nargs="+", required=True)
    parser.add_argument("--model-labels", nargs="+", required=True)

    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--title", type=str)

    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Validate lengths
    assert len(args.model_checkpoints) == len(args.model_types) == len(args.model_labels), (
        "model-checkpoints, model-types, and model-labels must be the same length!"
    )

    setup_logging(level="INFO", log_file=None, rich_tracebacks=True, show_level=True)
    device = torch.device(args.device)

    # Load dataset
    # dataloader = create_dataloaders(args, max_samples=100)
    dataloader = create_dataloaders(args)

    # Results
    all_results_svg = {}
    all_results_img = {}

    # ---------------------------------------------------------
    # Evaluate each model manually
    # ---------------------------------------------------------
    for ckpt_path, model_type, label in zip(
        args.model_checkpoints, args.model_types, args.model_labels, strict=True
    ):
        logging.info(f"\n===== Evaluating {label} ({model_type}) =====")

        # Load proper architecture
        model = load_model_by_type(model_type, device)

        # Load weights
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)

        # Extract embeddings
        svg_embeds, img_embeds = get_embeddings(model, dataloader)

        # load DINO embeddings using contrastive model
        contrastive = load_model_by_type("contrastive", device)
        _, img_embeds = get_embeddings(contrastive, dataloader)

        # PCA only if embeddings exist
        if svg_embeds is not None:
            explained_svg = run_pca(svg_embeds, tag=f"{label}_svg")
            all_results_svg[label] = explained_svg
        else:
            logging.warning(f"No SVG embeddings for {label}")

        if img_embeds is not None:
            explained_img = run_pca(img_embeds, tag=f"{label}_img")
            all_results_img[label] = explained_img
        else:
            logging.warning(f"No IMG embeddings for {label}")

    # ---------------------------------------------------------
    # Combined Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))

    fontdict = {
        "fontsize": 10,
        "fontweight": "normal",
        "family": "serif",
        "color": "black",
    }

    legend_font = FontProperties(family="serif", size=10, weight="normal")

    for label, curve in all_results_svg.items():
        plt.plot(np.cumsum(curve), label=f"{label}")

    for label, curve in all_results_img.items():
        plt.plot(np.cumsum(curve), linestyle="--", color="purple", label="DINOv2 Image")
        break

    plt.xlabel("Principal Component", fontdict=fontdict)
    plt.ylabel("Cumulative Variance Explained", fontdict=fontdict)
    plt.title(args.title, fontdict=fontdict)
    plt.grid()

    # Legend outside the plot
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop=legend_font)

    plt.tight_layout()  # Adjust layout so everything fits
    outpath = args.out_dir
    plt.savefig(outpath, bbox_inches="tight")  # Include the legend in saved figure
    logging.info(f"[saved] {outpath}")


if __name__ == "__main__":
    main()
