"""
Sample a subset of n svg images from a dataset

Usage:
    python scripts/extract_svg_sample.py \
        --seed 42 \
        --svg-dir /oscar/scratch/zzhan215/google_fonts_processed_reduced/svg/ \
        --img-dir /oscar/scratch/zzhan215/google_fonts_processed_reduced/img/ \
        --meta-dir /oscar/scratch/zzhan215/google_fonts_processed_reduced/metadata.csv \
        --svg-save-dir /oscar/scratch/avelama1/google_fonts_svg_sample_dataset/svg/ \
        --img-save-dir /oscar/scratch/avelama1/google_fonts_svg_sample_dataset/img/ \
        --meta-save-dir /oscar/scratch/avelama1/google_fonts_svg_sample_dataset/metadata.csv \
        --num-samples 100
"""

import os
import random
import argparse
import shutil
import pandas as pd

def sample_svg_paths(dir_path, n=100, seed=42):
    svgs = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(".pt")
    ]

    if len(svgs) < n:
        raise ValueError(f"Requested {n} samples but only found {len(svgs)} SVG files.")

    random.seed(seed)
    return random.sample(svgs, n)

def get_corresponding_image_paths(svg_paths, img_dir):
    """
    For each SVG path, return the corresponding image path from img_dir.
    Image extensions checked: .png, .jpg, .jpeg
    Returns:
        List of (svg_path, img_path_or_None)
    """
    possible_exts = ["png"]
    results = []

    for svg_path in svg_paths:
        svg_name = os.path.basename(svg_path)
        base = os.path.splitext(svg_name)[0]

        img_path = None
        for ext in possible_exts:
            candidate = os.path.join(img_dir, base + "." + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        results.append((svg_path, img_path))

    return results

def filter_metadata_by_svgs(metadata_csv_path, svg_paths):
    """
    Filter metadata CSV rows where uuid matches the SVG filenames.
    
    """
    # Extract base names without extension
    svg_uuids = {os.path.splitext(os.path.basename(p))[0] for p in svg_paths}

    # Load metadata
    df = pd.read_csv(metadata_csv_path)

    # Filter
    filtered = df[df["uuid"].isin(svg_uuids)]

    return filtered

def main():
    parser = argparse.ArgumentParser(description="Sample random SVG files and copy them.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--svg-dir", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--meta-dir", type=str, required=True)
    parser.add_argument("--svg-save-dir", type=str, required=True)
    parser.add_argument("--img-save-dir", type=str, required=True)
    parser.add_argument("--meta-save-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100)

    args = parser.parse_args()

    # Sample file paths
    sampled_svgs = sample_svg_paths(
        dir_path=args.svg_dir,
        n=args.num_samples,
        seed=args.seed
    )

    # Corresponding images
    paired_paths = get_corresponding_image_paths(
        svg_paths=sampled_svgs,
        img_dir=args.img_dir
    )

    # Correspoding metada
    metadata = filter_metadata_by_svgs(args.meta_dir, sampled_svgs)
    metadata.to_csv(args.meta_save_dir, index=False)

    # Create output directory
    os.makedirs(args.svg_save_dir, exist_ok=True)
    os.makedirs(args.img_save_dir, exist_ok=True)

    for svg_path, img_path in paired_paths:
        # Copy SVG
        shutil.copy(svg_path, os.path.join(args.svg_save_dir, os.path.basename(svg_path)))

        # Copy IMG (if found)
        if img_path is not None:
            shutil.copy(img_path, os.path.join(args.img_save_dir, os.path.basename(img_path)))
        else:
            print(f"WARNING: No matching image for {os.path.basename(svg_path)}")

    print(f"Samples {len(paired_paths)} SVG/IMG files")

if __name__ == "__main__":
    main()
