"""
Compare SVG vs Tensor dataloading performance
Runs both tests and generates a comparison report
"""

import os
import time
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset


def custom_collate(batch):
    """Custom collate function that handles SVGTensor objects"""
    collated = {}
    collated["commands"] = torch.stack([item["commands"] for item in batch])
    collated["args"] = torch.stack([item["args"] for item in batch])
    collated["image"] = torch.stack([item["image"] for item in batch])
    collated["tensors"] = [item["tensors"] for item in batch]
    collated["uuid"] = [item["uuid"] for item in batch]
    collated["name"] = [item["name"] for item in batch]
    collated["source"] = [item["source"] for item in batch]
    return collated


def benchmark_single_sample(svg_dir, tensor_dir, img_dir, meta_file, num_samples=50):
    """Compare single sample loading times"""
    print("\n" + "="*80)
    print("SINGLE SAMPLE LOADING COMPARISON")
    print("="*80)

    # Create datasets
    dataset_svg = SVGXDataset(
        svg_dir=svg_dir,
        img_dir=img_dir,
        meta_filepath=meta_file,
        max_num_groups=8,
        max_seq_len=40,
        train_ratio=1.0,
        already_preprocessed=True,
        already_tensor=False
    )

    dataset_tensor = SVGXDataset(
        svg_dir=tensor_dir,
        img_dir=img_dir,
        meta_filepath=meta_file,
        max_num_groups=8,
        max_seq_len=40,
        train_ratio=1.0,
        already_preprocessed=True,
        already_tensor=True
    )

    total_samples = min(len(dataset_svg), len(dataset_tensor))
    indices = random.sample(range(total_samples), min(num_samples, total_samples))

    print(f"Testing {len(indices)} random samples from {total_samples} total samples\n")

    # Benchmark SVG loading
    print("Loading from SVG files...")
    svg_times = []
    for idx in indices:
        start = time.perf_counter()
        _ = dataset_svg[idx]
        end = time.perf_counter()
        svg_times.append((end - start) * 1000)

    # Benchmark tensor loading
    print("Loading from tensor files...")
    tensor_times = []
    for idx in indices:
        start = time.perf_counter()
        _ = dataset_tensor[idx]
        end = time.perf_counter()
        tensor_times.append((end - start) * 1000)

    # Calculate statistics
    svg_mean = sum(svg_times) / len(svg_times)
    tensor_mean = sum(tensor_times) / len(tensor_times)
    speedup = svg_mean / tensor_mean if tensor_mean > 0 else 0

    # Print comparison
    print(f"\n{'Metric':<20} {'SVG Loading':<20} {'Tensor Loading':<20} {'Speedup':<15}")
    print("-" * 80)
    print(f"{'Mean (ms)':<20} {svg_mean:>18.2f}  {tensor_mean:>18.2f}  {speedup:>13.2f}x")
    print(f"{'Median (ms)':<20} {sorted(svg_times)[len(svg_times)//2]:>18.2f}  {sorted(tensor_times)[len(tensor_times)//2]:>18.2f}")
    print(f"{'Min (ms)':<20} {min(svg_times):>18.2f}  {min(tensor_times):>18.2f}")
    print(f"{'Max (ms)':<20} {max(svg_times):>18.2f}  {max(tensor_times):>18.2f}")

    return {
        "svg_mean_ms": svg_mean,
        "tensor_mean_ms": tensor_mean,
        "speedup": speedup
    }


def benchmark_dataloader(svg_dir, tensor_dir, img_dir, meta_file, batch_size=16, num_workers=4, num_iterations=25):
    """Compare DataLoader throughput"""
    print("\n" + "="*80)
    print(f"DATALOADER THROUGHPUT COMPARISON (batch={batch_size}, workers={num_workers})")
    print("="*80)

    # Test SVG loading
    dataset_svg = SVGXDataset(
        svg_dir=svg_dir,
        img_dir=img_dir,
        meta_filepath=meta_file,
        max_num_groups=8,
        max_seq_len=40,
        train_ratio=1.0,
        already_preprocessed=True,
        already_tensor=False
    )

    dataloader_svg = DataLoader(
        dataset_svg,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate,
        shuffle=False
    )

    # Warmup
    for i, _ in enumerate(dataloader_svg):
        if i >= 2:
            break

    # Measure
    print("\nTesting SVG loading...")
    start = time.perf_counter()
    svg_samples = 0
    for i, batch in enumerate(dataloader_svg):
        svg_samples += batch["commands"].shape[0]
        if i >= num_iterations:
            break
    svg_time = time.perf_counter() - start
    svg_throughput = svg_samples / svg_time if svg_time > 0 else 0

    # Test tensor loading
    dataset_tensor = SVGXDataset(
        svg_dir=tensor_dir,
        img_dir=img_dir,
        meta_filepath=meta_file,
        max_num_groups=8,
        max_seq_len=40,
        train_ratio=1.0,
        already_preprocessed=True,
        already_tensor=True
    )

    dataloader_tensor = DataLoader(
        dataset_tensor,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate,
        shuffle=False
    )

    # Warmup
    for i, _ in enumerate(dataloader_tensor):
        if i >= 2:
            break

    # Measure
    print("Testing tensor loading...")
    start = time.perf_counter()
    tensor_samples = 0
    for i, batch in enumerate(dataloader_tensor):
        tensor_samples += batch["commands"].shape[0]
        if i >= num_iterations:
            break
    tensor_time = time.perf_counter() - start
    tensor_throughput = tensor_samples / tensor_time if tensor_time > 0 else 0

    speedup = tensor_throughput / svg_throughput if svg_throughput > 0 else 0

    # Print comparison
    print(f"\n{'Method':<20} {'Throughput (s/sec)':<20} {'Time (sec)':<15} {'Samples':<10}")
    print("-" * 70)
    print(f"{'SVG Loading':<20} {svg_throughput:>18.2f}  {svg_time:>13.2f}  {svg_samples:>8}")
    print(f"{'Tensor Loading':<20} {tensor_throughput:>18.2f}  {tensor_time:>13.2f}  {tensor_samples:>8}")
    print("-" * 70)
    print(f"{'Speedup':<20} {speedup:>18.2f}x")

    return {
        "svg_throughput": svg_throughput,
        "tensor_throughput": tensor_throughput,
        "speedup": speedup
    }


def main():
    # Set paths
    svg_dir = "/Users/jz/work/csci2952project/svgx_svgs"
    tensor_dir = "/Users/jz/work/csci2952project/svgx_svgs"  # Same dir, different file types
    img_dir = "/Users/jz/work/csci2952project/svgx_imgs"
    meta_file = "/Users/jz/work/csci2952project/svgx_meta.csv"

    print("="*80)
    print("SVG vs TENSOR DATALOADING COMPARISON")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  SVG Directory:   {svg_dir}")
    print(f"  Tensor Directory: {tensor_dir}")
    print(f"  Image Directory:  {img_dir}")
    print(f"  Metadata File:    {meta_file}")

    # Check for files
    svg_files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]
    pt_files = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]

    print(f"\nFound:")
    print(f"  {len(svg_files)} .svg files")
    print(f"  {len(pt_files)} .pt files")

    if not svg_files or not pt_files:
        print("\nERROR: Need both .svg and .pt files to compare!")
        return

    # Run benchmarks
    try:
        single_results = benchmark_single_sample(svg_dir, tensor_dir, img_dir, meta_file, num_samples=50)
        dataloader_results = benchmark_dataloader(svg_dir, tensor_dir, img_dir, meta_file, batch_size=16, num_workers=4)

        # Print final summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\nSingle Sample Loading:")
        print(f"  SVG:    {single_results['svg_mean_ms']:.2f} ms/sample")
        print(f"  Tensor: {single_results['tensor_mean_ms']:.2f} ms/sample")
        print(f"  Speedup: {single_results['speedup']:.2f}x faster")
        print(f"\nDataLoader Throughput (batch=16, workers=4):")
        print(f"  SVG:    {dataloader_results['svg_throughput']:.2f} samples/sec")
        print(f"  Tensor: {dataloader_results['tensor_throughput']:.2f} samples/sec")
        print(f"  Speedup: {dataloader_results['speedup']:.2f}x faster")

        # Save results
        results_df = pd.DataFrame([{
            "single_svg_ms": single_results['svg_mean_ms'],
            "single_tensor_ms": single_results['tensor_mean_ms'],
            "single_speedup": single_results['speedup'],
            "dataloader_svg_throughput": dataloader_results['svg_throughput'],
            "dataloader_tensor_throughput": dataloader_results['tensor_throughput'],
            "dataloader_speedup": dataloader_results['speedup']
        }])
        output_file = "/Users/jz/work/csci2952project/jz_test/20251119_test_tensor_dataset/comparison_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        print("\n" + "="*80)
        print("COMPARISON COMPLETE!")
        print("="*80)

    except Exception as e:
        print(f"\nERROR during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
