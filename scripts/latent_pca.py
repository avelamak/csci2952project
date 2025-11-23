import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import numpy as np

from vecssl.models.jepa import Jepa
from vecssl.models.config import JepaConfig
from torch.utils.data import DataLoader
from vecssl.data.dataset import SVGXDataset

import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def get_embeddings(model, dataloader):
    model.eval()
    all_svg = []
    all_img = []

    for batch in dataloader:
        out = model.encode_joint(batch)
        all_svg.append(out["svg"].cpu())
        all_img.append(out["img"].cpu())

    all_svg = torch.cat(all_svg, dim=0)
    all_img = torch.cat(all_img, dim=0)
    return all_svg, all_img


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

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(explained))
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Variance Explained")
    plt.title(f"PCA: {tag}")
    plt.grid()

    plot_path = f"pca_{tag}.png"
    plt.savefig(plot_path)
    logging.info(f"[saved] {plot_path}")

    return explained


def custom_collate(batch):
    """Custom collate function that handles SVGTensor objects"""
    collated = {}

    # Stack tensors normally
    collated["commands"] = torch.stack([item["commands"] for item in batch])
    collated["args"] = torch.stack([item["args"] for item in batch])
    collated["image"] = torch.stack([item["image"] for item in batch])

    # Keep SVGTensor objects and strings as lists
    collated["tensors"] = [item["tensors"] for item in batch]
    collated["uuid"] = [item["uuid"] for item in batch]
    collated["name"] = [item["name"] for item in batch]
    collated["source"] = [item["source"] for item in batch]

    return collated


def create_dataloaders(args):
    """Create train and val dataloaders"""
    logger.info("Creating datasets...")

    dataset = SVGXDataset(
        svg_dir=args.svg_dir,
        img_dir=args.img_dir,
        meta_filepath=args.meta,
        max_num_groups=8,
        max_seq_len=40,
        train_ratio=1.0,
        already_preprocessed=True,
    )

    logger.info(f"  Val samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate,
        drop_last=False,
    )

    return loader


def main():
    parser = argparse.ArgumentParser()
    # Dataset args
    parser.add_argument("--svg-dir", type=str, default="svgx_svgs", help="SVG directory")
    parser.add_argument("--img-dir", type=str, default="svgx_imgs", help="Image directory")
    parser.add_argument("--meta", type=str, default="svgx_meta.csv", help="Metadata CSV")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    cfg = JepaConfig()
    model = Jepa(cfg).to(device)

    logging.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)

    dataloader = create_dataloaders(args)

    svg_embeds, img_embeds = get_embeddings(model, dataloader)

    run_pca(svg_embeds, tag="svg")
    run_pca(img_embeds, tag="img")


if __name__ == "__main__":
    main()
