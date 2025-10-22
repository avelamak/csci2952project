import matplotlib.pyplot as plt
from typing import Optional, List
from torch.utils.data import DataLoader
import numpy as np
import sys
from pathlib import Path
import torch

from vecssl.data.dataset import SVGXDataset


def visualize_samples(
    dataset,
    num_samples: int = 4,
    indices: Optional[List[int]] = None,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
):
    if indices is None:
        indices = list(range(num_samples))
    else:
        num_samples = len(indices)

    # Create subplot grid
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols

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

        # Display image if available
        if "image" in sample:
            img = sample["image"]

            # Convert tensor to numpy array for visualization
            if isinstance(img, torch.Tensor):
                # Convert from (C, H, W) to (H, W, C) and to numpy
                img = img.permute(1, 2, 0).numpy()

            ax.imshow(img)
            ax.axis("off")

            # Add caption as title if available
            caption = None
            if "caption" in sample:
                caption = sample["caption"]
            elif "blip_caption" in sample:
                caption = sample["blip_caption"]

            if caption:
                # Truncate long captions
                if len(caption) > 60:
                    caption = caption[:57] + "..."
                ax.set_title(caption, fontsize=8, wrap=True)
        else:
            ax.text(
                0.5,
                0.5,
                "No image available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


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
        dataset = SVGXDataset(split="train")
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Test 2: Inspect a single sample
    print("\n[2] Inspecting first sample...")
    try:
        print_sample_info(dataset, idx=0)
        print("✓ Sample inspection successful!")
    except Exception as e:
        print(f"✗ Failed to inspect sample: {e}")
        return

    # Test 3: Create DataLoader
    print("\n[3] Creating DataLoader...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=1,
        )
        print("✓ DataLoader created successfully!")
    except Exception as e:
        print(f"✗ Failed to create DataLoader: {e}")
        return

    # Test 4: Iterate through a batch
    print("\n[4] Testing batch iteration...")
    try:
        batch = next(iter(dataloader))
        print(f"✓ Successfully loaded a batch!")
        print(f"  Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if hasattr(value, "shape"):
                print(f"  {key} shape: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key} length: {len(value)}")
            else:
                print(f"  {key} type: {type(value)}")
    except Exception as e:
        print(f"✗ Failed to iterate batch: {e}")
        return

    # Test 5: Visualize samples
    print("\n[5] Visualizing samples...")
    try:
        output_path = Path(__file__).parent / "sample_visualization.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        visualize_samples(
            dataset,
            num_samples=8,
            indices=[0, 1, 2, 3, 10, 20, 30, 40],
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
