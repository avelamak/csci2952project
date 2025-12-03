"""
Test DataLoader efficiency by simulating realistic training workloads.

This script measures:
1. Data loading time vs GPU compute time
2. Whether data loading is a bottleneck
3. Throughput in samples/second

Usage:
    python scripts/test_dataloader.py --svg-dir svgx_svgs --img-dir svgx_imgs --meta svgx_meta.csv
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vecssl.data.dataset import SVGXDataset
from vecssl.util import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Timing statistics for dataloader benchmarking."""

    total_time: float = 0.0
    data_time: float = 0.0
    compute_time: float = 0.0
    transfer_time: float = 0.0
    num_batches: int = 0
    num_samples: int = 0

    @property
    def data_pct(self) -> float:
        return (self.data_time / self.total_time * 100) if self.total_time > 0 else 0

    @property
    def compute_pct(self) -> float:
        return (self.compute_time / self.total_time * 100) if self.total_time > 0 else 0

    @property
    def transfer_pct(self) -> float:
        return (self.transfer_time / self.total_time * 100) if self.total_time > 0 else 0

    @property
    def throughput(self) -> float:
        return self.num_samples / self.total_time if self.total_time > 0 else 0

    @property
    def avg_batch_time(self) -> float:
        return self.total_time / self.num_batches if self.num_batches > 0 else 0


def custom_collate(batch):
    """Custom collate function matching test_svg_autoencoder.py"""
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


class DummyModel(nn.Module):
    """
    Dummy model that simulates realistic GPU compute time.

    Uses matrix multiplications to create realistic GPU workload
    that matches approximate training step duration.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 6, simulate_backward: bool = True):
        super().__init__()
        self.simulate_backward = simulate_backward

        # Create layers that roughly match SVGTransformer compute
        self.cmd_embed = nn.Embedding(10, d_model)
        self.arg_proj = nn.Linear(11, d_model)

        # Transformer-like layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model, nhead=8, dim_feedforward=d_model * 4, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )

        # Output heads
        self.cmd_head = nn.Linear(d_model, 10)
        self.arg_head = nn.Linear(d_model, 11)

    def forward(self, commands: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
        """
        Simulate forward pass computation.

        Args:
            commands: [B, G, S] command indices
            args: [B, G, S, 11] argument values
        """
        B, G, S = commands.shape

        # Flatten groups into sequence
        commands_flat = commands.view(B, G * S)
        args_flat = args.view(B, G * S, -1)

        # Embed
        x = self.cmd_embed(commands_flat.clamp(0, 9)) + self.arg_proj(args_flat.float())

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        cmd_logits = self.cmd_head(x)
        arg_logits = self.arg_head(x)

        # Compute dummy loss
        loss = cmd_logits.mean() + arg_logits.mean()
        return loss


def run_benchmark(
    loader: DataLoader,
    device: torch.device,
    num_epochs: int = 1,
    max_batches: Optional[int] = None,
    simulate_training: bool = True,
    warmup_batches: int = 5,
) -> TimingStats:
    """
    Run dataloader benchmark with optional training simulation.

    Args:
        loader: DataLoader to benchmark
        device: Device to transfer data to
        num_epochs: Number of epochs to run
        max_batches: Maximum batches per epoch (None for all)
        simulate_training: Whether to simulate GPU compute
        warmup_batches: Number of warmup batches to skip in timing
    """
    stats = TimingStats()

    # Create dummy model if simulating training
    model = None
    optimizer = None
    if simulate_training and device.type == "cuda":
        model = DummyModel().to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_count = 0

    for _epoch in range(num_epochs):
        data_start = time.perf_counter()

        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break

            # --- Data loading time ---
            data_end = time.perf_counter()
            batch_data_time = data_end - data_start

            # --- Transfer to device ---
            transfer_start = time.perf_counter()
            commands = batch["commands"].to(device, non_blocking=True)
            args = batch["args"].to(device, non_blocking=True)
            images = batch["image"].to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            transfer_end = time.perf_counter()
            batch_transfer_time = transfer_end - transfer_start

            # --- Simulate compute ---
            compute_start = time.perf_counter()
            if model is not None:
                optimizer.zero_grad()
                loss = model(commands, args)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
            else:
                # CPU-only: just do some tensor ops
                _ = commands.float().mean()
                _ = args.float().mean()
                _ = images.float().mean()
            compute_end = time.perf_counter()
            batch_compute_time = compute_end - compute_start

            # Accumulate stats (skip warmup batches)
            batch_count += 1
            if batch_count > warmup_batches:
                stats.data_time += batch_data_time
                stats.transfer_time += batch_transfer_time
                stats.compute_time += batch_compute_time
                stats.num_batches += 1
                stats.num_samples += commands.shape[0]

            # Log progress
            if batch_count % 20 == 0:
                logger.info(
                    f"  Batch {batch_count}: data={batch_data_time * 1000:.1f}ms, "
                    f"transfer={batch_transfer_time * 1000:.1f}ms, "
                    f"compute={batch_compute_time * 1000:.1f}ms"
                )

            # Start timing next data load
            data_start = time.perf_counter()

    stats.total_time = stats.data_time + stats.transfer_time + stats.compute_time
    return stats


def log_results(stats: TimingStats, title: str = "Results"):
    """Log benchmark results in a formatted table."""
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    logger.info(f"Total batches:     {stats.num_batches:,}")
    logger.info(f"Total samples:     {stats.num_samples:,}")
    logger.info(f"Total time:        {stats.total_time:.2f}s")
    logger.info("-" * 60)
    logger.info(f"Data loading:      {stats.data_time:.2f}s ({stats.data_pct:.1f}%)")
    logger.info(f"GPU transfer:      {stats.transfer_time:.2f}s ({stats.transfer_pct:.1f}%)")
    logger.info(f"Compute:           {stats.compute_time:.2f}s ({stats.compute_pct:.1f}%)")
    logger.info("-" * 60)
    logger.info(f"Throughput:        {stats.throughput:.1f} samples/sec")
    logger.info(f"Avg batch time:    {stats.avg_batch_time * 1000:.1f}ms")
    logger.info("=" * 60)

    # Bottleneck analysis
    logger.info("Bottleneck Analysis:")
    if stats.data_pct > 50:
        logger.warning(f"DATA LOADING IS THE BOTTLENECK ({stats.data_pct:.1f}%)")
        logger.info("  Suggestions:")
        logger.info("  - Increase num_workers")
        logger.info("  - Enable pin_memory=True")
        logger.info("  - Use prefetch_factor > 2")
        logger.info("  - Pre-process/cache data to disk")
    elif stats.transfer_pct > 30:
        logger.warning(f"GPU TRANSFER IS SLOW ({stats.transfer_pct:.1f}%)")
        logger.info("  Suggestions:")
        logger.info("  - Enable pin_memory=True")
        logger.info("  - Use non_blocking=True (already enabled)")
    else:
        logger.info(f"Compute-bound ({stats.compute_pct:.1f}% compute) - DataLoader is efficient!")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DataLoader efficiency for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset args
    parser.add_argument("--svg-dir", type=Path, default="svgx_svgs")
    parser.add_argument("--img-dir", type=Path, default="svgx_imgs")
    parser.add_argument("--meta", type=Path, default="svgx_meta.csv")
    parser.add_argument("--split", type=str, default="train")

    # DataLoader args
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true", default=False)

    # Dataset config (match training)
    parser.add_argument("--max-num-groups", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=40)

    # Benchmark args
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Max batches per epoch (None for all)"
    )
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--no-simulate", action="store_true", help="Skip GPU compute simulation")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )

    # Comparison mode
    parser.add_argument(
        "--sweep-workers", action="store_true", help="Sweep num_workers from 0 to 8"
    )

    # Logging args
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # Create dataset
    logger.info("Creating dataset...")
    ds = SVGXDataset(
        svg_dir=str(args.svg_dir),
        img_dir=str(args.img_dir),
        meta_filepath=str(args.meta),
        split=args.split,
        max_num_groups=args.max_num_groups,
        max_seq_len=args.max_seq_len,
        already_preprocessed=True,
    )
    logger.info(f"Dataset size: {len(ds)}")

    if args.sweep_workers:
        # Sweep through different num_workers values
        logger.info("=" * 60)
        logger.info("NUM_WORKERS SWEEP")
        logger.info("=" * 60)

        results = []
        for num_workers in [0, 1, 2, 4, 6, 8]:
            logger.info(f">>> Testing num_workers={num_workers}")

            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                num_workers=num_workers,
                shuffle=True,
                collate_fn=custom_collate,
                pin_memory=args.pin_memory and device.type == "cuda",
                prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
                persistent_workers=args.persistent_workers and num_workers > 0,
            )

            stats = run_benchmark(
                loader,
                device,
                num_epochs=1,
                max_batches=args.max_batches or 50,
                simulate_training=not args.no_simulate,
                warmup_batches=args.warmup_batches,
            )
            results.append((num_workers, stats))

        # Log comparison table
        logger.info("=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)
        logger.info(
            f" {'Workers':<10} {'Throughput':<15} {'Data %':<10} {'Compute %':<10} {'Batch (ms)':<12}"
        )
        logger.info("-" * 70)
        for nw, s in results:
            logger.info(
                f" {nw:<10} {s.throughput:>10.1f}/s   {s.data_pct:>6.1f}%    {s.compute_pct:>6.1f}%     {s.avg_batch_time * 1000:>8.1f}"
            )
        logger.info("=" * 70)

    else:
        # Single benchmark run
        logger.info("DataLoader config:")
        logger.info(f"  batch_size:         {args.batch_size}")
        logger.info(f"  num_workers:        {args.num_workers}")
        logger.info(f"  pin_memory:         {args.pin_memory}")
        logger.info(f"  prefetch_factor:    {args.prefetch_factor}")
        logger.info(f"  persistent_workers: {args.persistent_workers}")

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=custom_collate,
            pin_memory=args.pin_memory and device.type == "cuda",
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
        )

        logger.info(f"Running benchmark ({args.num_epochs} epoch(s))...")
        stats = run_benchmark(
            loader,
            device,
            num_epochs=args.num_epochs,
            max_batches=args.max_batches,
            simulate_training=not args.no_simulate,
            warmup_batches=args.warmup_batches,
        )

        log_results(stats, f"DataLoader Benchmark (workers={args.num_workers})")


if __name__ == "__main__":
    main()
