import matplotlib.pyplot as plt
from typing import Optional, List
from torch.utils.data import DataLoader
import numpy as np
import sys
from pathlib import Path
import torch

from vecssl.data.dataset import SVGXDataset
from vecssl.data.svg_tensor import SVGTensor
from vecssl.data.svg import SVG
from vecssl.data.geom import Bbox


def custom_collate(batch):
    """Custom collate function that handles SVGTensor objects"""
    # Separate SVGTensor objects from tensors
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


def visualize_samples(
    dataset,
    num_samples: int = 4,
    indices: Optional[List[int]] = None,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
):
    if indices is None:
        indices = list(range(min(num_samples, len(dataset))))
    else:
        # Only use valid indices
        indices = [idx for idx in indices if idx < len(dataset)]
        num_samples = len(indices)

    # Create subplot grid for original images
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols

    # Figure 1: Original images
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        sample = dataset[idx]
        ax = axes[i]

        if "image" in sample:
            img = sample["image"]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"#{idx}", fontsize=8)
        ax.axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Original images saved to {save_path}")
    else:
        plt.show()

    plt.close()

    # Figure 2: Reconstructed SVGs
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        sample = dataset[idx]
        ax = axes[i]

        nempty_tensors: list[torch.Tensor] = []
        for j in range(len(sample["tensors"])):
            tensor = sample["tensors"][j].copy().drop_sos().unpad()
            if tensor.seq_len > 0:
                nempty_tensors.append(tensor.data)
        full_tensor = torch.cat(nempty_tensors)
        svg_reconstructed = SVG.from_tensor(
            full_tensor.data, viewbox=Bbox(256), allow_empty=True
            ).normalize().split_paths().set_color("random")

        img_recon = svg_reconstructed.draw(do_display=False, return_png=True)
        ax.imshow(np.array(img_recon))
        ax.set_title(f"Reconstructed #{idx}", fontsize=8)
        # except Exception as e:
        #     ax.text(0.5, 0.5, f"Error: {str(e)[:20]}", ha="center", va="center", fontsize=8)
        #     print(f"Failed to reconstruct {idx}: {e}")

        ax.axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        recon_path = Path(save_path).parent / (Path(save_path).stem + "_reconstructed.png")
        plt.savefig(recon_path, dpi=150, bbox_inches="tight")
        print(f"Reconstructed SVGs saved to {recon_path}")
    else:
        plt.show()

    plt.close()


def print_sample_info(dataset, idx: int = 0):
    sample = dataset[idx]

    print(f"\n{'='*60}")
    print(f"Sample {idx} Information")
    print(f"{'='*60}")

    print(f"\nAvailable keys: {list(sample.keys())}")

    for key, value in sample.items():
        print(f"\n{key}:")
        if key == "image":
            print(f"  Type: {type(value)}")
            if isinstance(value, torch.Tensor):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Range: [{value.min():.3f}, {value.max():.3f}]")
            elif hasattr(value, "size"):
                print(f"  Size: {value.size}")
        elif key in ["svg", "svg_code", "svg_path"]:
            # Truncate long SVG code
            val_str = str(value)
            if len(val_str) > 200:
                print(f"  {val_str[:200]}...")
            else:
                print(f"  {val_str}")
        else:
            val_str = str(value)
            if len(val_str) > 300:
                print(f"  {val_str[:300]}...")
            else:
                print(f"  {val_str}")

    print(f"\n{'='*60}\n")


def main():
    """Run basic tests on the SVGX dataset."""
    print("=" * 60)
    print("SVGX-Core-250k Dataset Test")
    print("=" * 60)

    # Test 1: Load dataset
    print("\n[1] Loading dataset...")
    try:
        dataset = SVGXDataset(svg_dir="/Users/jz/work/csci2952project/svgx_svgs", img_dir="/Users/jz/work/csci2952project/svgx_imgs", meta_filepath="/Users/jz/work/csci2952project/svgx_meta.csv", max_num_groups=10, max_seq_len=80)
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Test 2: Inspect a single sample
    print("\n[2] Inspecting first sample...")
    print_sample_info(dataset, idx=0)
    print("✓ Sample inspection successful!")

    # Test 3: Create DataLoader
    print("\n[3] Creating DataLoader...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=1,
            collate_fn=custom_collate,
        )
        print("✓ DataLoader created successfully!")
    except Exception as e:
        print(f"✗ Failed to create DataLoader: {e}")
        return
    # Test 4: Iterate through a batch
    print("\n[4] Testing batch iteration...")
    iterator = iter(dataloader)
    for _ in range(2):
        batch = next(iterator)
        print(f"✓ Successfully loaded a batch!")
        print(f"  Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if hasattr(value, "shape"):
                print(f"  {key} shape: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key} length: {len(value)}")
            else:
                print(f"  {key} type: {type(value)}")

    # Test 5: Visualize samples
    print("\n[5] Visualizing samples...")
    try:
        output_path = Path(__file__).parent / "sample_visualization.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        visualize_samples(
            dataset,
            num_samples=8,
            indices=[0, 1, 2, 3, 4, 5, 6, 7],
            figsize=(16, 8),
            save_path=str(output_path),
        )
        print(f"✓ Visualization saved to {output_path}")
    except Exception as e:
        print(f"✗ Failed to visualize samples: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
