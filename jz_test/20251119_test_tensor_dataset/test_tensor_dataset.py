"""
Test tensor dataloading performance with already_tensor=True
Tests loading .pt files from preprocessing with --to_tensor flag
"""

import os
import time
import tracemalloc
import psutil
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset


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


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def test_single_sample_loading(tensor_dir, img_dir, meta_file, num_samples=50):
    """Test single sample loading time"""
    print("\n" + "="*80)
    print("1. SINGLE SAMPLE LOADING TEST")
    print("="*80)

    # Create dataset
    dataset = SVGXDataset(
        svg_dir=tensor_dir,
        img_dir=img_dir,
        meta_filepath=meta_file,
        max_num_groups=8,
        max_seq_len=40,
        train_ratio=1.0,
        already_preprocessed=True,
        already_tensor=True
    )

    # Get random sample indices
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))

    print(f"Testing {len(indices)} random samples from {total_samples} total samples")

    # Benchmark tensor loading
    print("\nLoading tensor samples...")
    loading_times = []
    for idx in indices:
        start = time.perf_counter()
        sample = dataset[idx]
        end = time.perf_counter()
        loading_times.append((end - start) * 1000)  # Convert to ms

    # Calculate statistics
    mean_time = sum(loading_times) / len(loading_times)
    median_time = sorted(loading_times)[len(loading_times) // 2]
    std_time = (sum((t - mean_time) ** 2 for t in loading_times) / len(loading_times)) ** 0.5
    min_time = min(loading_times)
    max_time = max(loading_times)

    # Print results
    print(f"\n{'Metric':<20} {'Time (ms)':<20}")
    print("-" * 40)
    print(f"{'Mean':<20} {mean_time:>18.2f}")
    print(f"{'Median':<20} {median_time:>18.2f}")
    print(f"{'Std Dev':<20} {std_time:>18.2f}")
    print(f"{'Min':<20} {min_time:>18.2f}")
    print(f"{'Max':<20} {max_time:>18.2f}")

    # Verify sample structure
    print(f"\nSample data structure:")
    print(f"  Commands shape: {sample['commands'].shape}")
    print(f"  Args shape:     {sample['args'].shape}")
    print(f"  Image shape:    {sample['image'].shape}")
    print(f"  Num tensors:    {len(sample['tensors'])}")
    print(f"  UUID:           {sample['uuid']}")

    return {
        "mean_ms": mean_time,
        "median_ms": median_time,
        "std_ms": std_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "num_samples": len(indices)
    }


def test_dataloader_throughput(tensor_dir, img_dir, meta_file, num_iterations=25):
    """Test DataLoader throughput with various configurations"""
    print("\n" + "="*80)
    print("2. DATALOADER THROUGHPUT TEST")
    print("="*80)

    batch_sizes = [4, 8, 16, 32]
    num_workers_list = [0, 2, 4]

    results = []

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            print(f"\nTesting batch_size={batch_size}, num_workers={num_workers}")

            dataset = SVGXDataset(
                svg_dir=tensor_dir,
                img_dir=img_dir,
                meta_filepath=meta_file,
                max_num_groups=8,
                max_seq_len=40,
                train_ratio=1.0,
                already_preprocessed=True,
                already_tensor=True
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=custom_collate,
                shuffle=False
            )

            # Warmup
            for i, batch in enumerate(dataloader):
                if i >= 2:
                    break

            # Measure throughput
            start = time.perf_counter()
            total_samples = 0
            for i, batch in enumerate(dataloader):
                total_samples += batch["commands"].shape[0]
                if i >= num_iterations:
                    break
            elapsed = time.perf_counter() - start
            throughput = total_samples / elapsed if elapsed > 0 else 0

            print(f"  Throughput: {throughput:6.2f} samples/sec ({elapsed:.2f}s for {total_samples} samples)")

            results.append({
                "batch_size": batch_size,
                "num_workers": num_workers,
                "throughput": throughput,
                "elapsed_sec": elapsed,
                "total_samples": total_samples
            })

    # Print summary table
    print("\n" + "-"*70)
    print("SUMMARY TABLE")
    print("-"*70)
    print(f"{'Batch':<10} {'Workers':<10} {'Samples/sec':<15} {'Total Samples':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['batch_size']:<10} {r['num_workers']:<10} {r['throughput']:>13.2f}  {r['total_samples']:>13}")

    # Find best configuration
    best = max(results, key=lambda x: x['throughput'])
    print(f"\nBest configuration:")
    print(f"  batch_size={best['batch_size']}, num_workers={best['num_workers']}")
    print(f"  Throughput: {best['throughput']:.2f} samples/sec")

    return results


def test_memory_usage(tensor_dir, img_dir, meta_file, num_samples=50):
    """Test memory usage during loading"""
    print("\n" + "="*80)
    print("3. MEMORY USAGE TEST")
    print("="*80)

    print("\nMeasuring memory usage...")
    tracemalloc.start()
    mem_before = get_memory_usage()

    dataset = SVGXDataset(
        svg_dir=tensor_dir,
        img_dir=img_dir,
        meta_filepath=meta_file,
        max_num_groups=8,
        max_seq_len=40,
        train_ratio=1.0,
        already_preprocessed=True,
        already_tensor=True
    )

    # Load samples
    samples = []
    for i in range(min(num_samples, len(dataset))):
        samples.append(dataset[i])

    current, peak = tracemalloc.get_traced_memory()
    mem_after = get_memory_usage()
    tracemalloc.stop()

    peak_mb = peak / 1024 / 1024
    delta_mb = mem_after - mem_before
    per_sample_mb = delta_mb / num_samples if num_samples > 0 else 0

    # Print results
    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'Peak Memory (MB)':<30} {peak_mb:>18.2f}")
    print(f"{'Memory Delta (MB)':<30} {delta_mb:>18.2f}")
    print(f"{'Samples Loaded':<30} {num_samples:>18}")
    print(f"{'Memory per Sample (MB)':<30} {per_sample_mb:>18.2f}")

    return {
        "peak_mb": peak_mb,
        "delta_mb": delta_mb,
        "num_samples": num_samples,
        "per_sample_mb": per_sample_mb
    }


def save_results_to_csv(single_results, throughput_results, memory_results, output_file):
    """Save all results to CSV files"""
    print("\n" + "="*80)
    print("4. SAVING RESULTS")
    print("="*80)

    # Save single sample results
    single_df = pd.DataFrame([single_results])
    single_csv = output_file.replace(".csv", "_single.csv")
    single_df.to_csv(single_csv, index=False)
    print(f"Single sample results saved to: {single_csv}")

    # Save throughput results
    throughput_df = pd.DataFrame(throughput_results)
    throughput_csv = output_file.replace(".csv", "_throughput.csv")
    throughput_df.to_csv(throughput_csv, index=False)
    print(f"Throughput results saved to: {throughput_csv}")

    # Save memory results
    memory_df = pd.DataFrame([memory_results])
    memory_csv = output_file.replace(".csv", "_memory.csv")
    memory_df.to_csv(memory_csv, index=False)
    print(f"Memory results saved to: {memory_csv}")


def main():
    # Set paths - .pt files are in svgx_svgs directory
    tensor_dir = "/Users/jz/work/csci2952project/svgx_svgs"
    img_dir = "/Users/jz/work/csci2952project/svgx_imgs"
    meta_file = "/Users/jz/work/csci2952project/svgx_meta.csv"
    output_file = "/Users/jz/work/csci2952project/jz_test/20251119_test_tensor_dataset/benchmark_results.csv"

    print("="*80)
    print("TENSOR DATALOADING PERFORMANCE TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Tensor Directory (.pt files): {tensor_dir}")
    print(f"  Image Directory:              {img_dir}")
    print(f"  Metadata File:                {meta_file}")

    # Check if directories exist
    if not os.path.exists(tensor_dir):
        print(f"\nERROR: Tensor directory not found: {tensor_dir}")
        return
    if not os.path.exists(img_dir):
        print(f"\nERROR: Image directory not found: {img_dir}")
        return
    if not os.path.exists(meta_file):
        print(f"\nERROR: Metadata file not found: {meta_file}")
        return

    # Check for .pt files
    pt_files = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
    if not pt_files:
        print(f"\nERROR: No .pt files found in {tensor_dir}")
        print("\nTo create tensor files, run:")
        print("  python scripts/preprocess.py \\")
        print("    --output_meta_file svgx_meta.csv \\")
        print(f"    --output_svg_folder {tensor_dir} \\")
        print(f"    --output_img_folder {img_dir} \\")
        print("    --to_tensor \\")
        print("    --max_samples 500")
        return

    print(f"\nFound {len(pt_files)} .pt files in tensor directory")

    # Run tests
    try:
        single_results = test_single_sample_loading(tensor_dir, img_dir, meta_file, num_samples=50)
        throughput_results = test_dataloader_throughput(tensor_dir, img_dir, meta_file, num_iterations=25)
        memory_results = test_memory_usage(tensor_dir, img_dir, meta_file, num_samples=50)

        # Save results
        save_results_to_csv(single_results, throughput_results, memory_results, output_file)

        # Print final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"\nSingle Sample Loading:")
        print(f"  Average time: {single_results['mean_ms']:.2f} ms per sample")
        print(f"\nBest DataLoader Configuration:")
        best = max(throughput_results, key=lambda x: x['throughput'])
        print(f"  batch_size={best['batch_size']}, num_workers={best['num_workers']}")
        print(f"  Throughput: {best['throughput']:.2f} samples/sec")
        print(f"\nMemory Usage:")
        print(f"  Peak memory: {memory_results['peak_mb']:.2f} MB")
        print(f"  Per sample: {memory_results['per_sample_mb']:.2f} MB")

        print("\n" + "="*80)
        print("TEST COMPLETE!")
        print("="*80)

    except Exception as e:
        print(f"\nERROR during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    main()
