"""
Preprocess SVGX-Core-250k dataset: simplify SVGs, save metadata,
optionally save images/SVGs by UUID
"""

from vecssl.util import setup_logging, make_progress
from argparse import ArgumentParser
import logging
from pathlib import Path
import datasets
from datasets import load_dataset
import pandas as pd
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from vecssl.data.svg import SVG

logger = logging.getLogger(__name__)

# --- minimal worker state ---
_DATASET = None
_OUT_SVG = None
_OUT_IMG = None


def _init_worker(output_svg_folder, output_img_folder):
    """Run once per process: load cached HF dataset and remember output dirs."""
    global _DATASET, _OUT_SVG, _OUT_IMG
    _DATASET = load_dataset("xingxm/SVGX-Core-250k", split="train")
    _OUT_SVG = output_svg_folder
    _OUT_IMG = output_img_folder


def preprocess_svg_idx(idx):
    """Use your existing function, but fetch the sample inside the worker."""
    sample = _DATASET[idx]
    return preprocess_svg_sample(sample, _OUT_SVG, _OUT_IMG)


def preprocess_svg_sample(sample, output_svg_folder=None, output_img_folder=None):
    """Process a single HF dataset sample and return metadata"""
    try:
        uuid = sample["uuid"]
        svg_code = sample["svg_code"]

        # Load and simplify SVG
        svg = SVG.load_svg(svg_code).to_path()
        svg.fill_(False)
        svg.normalize()
        svg.zoom(0.9)
        svg.canonicalize()
        svg = svg.simplify_heuristic()

        # Calculate metadata
        len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]

        metadata = {
            "uuid": uuid,
            "name": sample.get("name", ""),
            "source": sample.get("source", ""),
            "total_len": sum(len_groups),
            "nb_groups": len(len_groups),
            "len_groups": len_groups,
            "max_len_group": max(len_groups) if len_groups else 0,
        }

        # Optionally save simplified SVG
        if output_svg_folder:
            svg_path = Path(output_svg_folder) / f"{uuid}.svg"
            svg.save_svg(str(svg_path))

        # Optionally save image as PNG (raw PIL Image from HF dataset)
        if output_img_folder and "image" in sample:
            img = sample["image"]  # PIL Image before ToTensor conversion
            img_path = Path(output_img_folder) / f"{uuid}.png"
            img.save(str(img_path))
        return metadata

    except Exception as e:
        logger.warning(f"[red]Failed to process sample {sample.get('uuid', 'unknown')}: {e}[/red]")
        logger.error(traceback.format_exc())
        return None


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--output_meta_file", required=True, type=str, help="Path to save metadata CSV"
    )
    parser.add_argument(
        "--output_svg_folder",
        default=None,
        type=str,
        help="Optional: folder to save simplified SVGs (by UUID)",
    )
    parser.add_argument(
        "--output_img_folder",
        default=None,
        type=str,
        help="Optional: folder to save images as PNG (by UUID)",
    )
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument(
        "--max_samples", default=None, type=int, help="Optional: limit number of samples to process"
    )
    args = parser.parse_args()

    # Setup logging (can be overridden with LOG_LEVEL env var)
    console = setup_logging(level="INFO", reset=True)
    logger.info("[bold cyan]Starting SVGX dataset preprocessing...[/bold cyan]")

    # Create output folders if needed
    if args.output_svg_folder:
        Path(args.output_svg_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"[dim]Creating SVG output folder: {args.output_svg_folder}[/dim]")
    if args.output_img_folder:
        Path(args.output_img_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"[dim]Creating image output folder: {args.output_img_folder}[/dim]")

    # Load dataset
    logger.info("[yellow]Loading SVGX-Core-250k dataset...[/yellow]")
    dataset: datasets.Dataset = load_dataset(  # type: ignore
        "xingxm/SVGX-Core-250k",
        split="train",
    )

    # Determine number of samples
    num_samples = args.max_samples if args.max_samples else len(dataset)
    logger.info(
        f"[blue]Processing [bold]{num_samples}[/bold] samples "
        f"with [bold]{args.workers}[/bold] workers[/blue]"
    )

    # Process samples in batches to avoid memory issues
    meta_data = []
    progress = make_progress(console)
    batch_size = args.workers * 32  # no RAM blowups

    with progress:
        task = progress.add_task("Processing SVGs", total=num_samples)

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.output_svg_folder, args.output_img_folder),
        ) as executor:
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)

                future_to_idx = {
                    executor.submit(preprocess_svg_idx, idx): idx
                    for idx in range(batch_start, batch_end)
                }

                for future in as_completed(future_to_idx):
                    result = future.result()
                    if result is not None:
                        meta_data.append(result)
                    progress.advance(task)

    # Save metadata
    df = pd.DataFrame(meta_data)
    df.to_csv(args.output_meta_file, index=False)

    logger.info("[bold green]✓ Preprocessing complete![/bold green]")
    logger.info(
        f"  [green]Successfully processed: [bold]{len(meta_data)}/{num_samples}[/bold] "
        f"samples[/green]"
    )
    logger.info(f"  [cyan]Metadata saved to:[/cyan] {args.output_meta_file}")
    if args.output_svg_folder:
        logger.info(f"  [cyan]SVGs saved to:[/cyan] {args.output_svg_folder}")
    if args.output_img_folder:
        logger.info(f"  [cyan]Images saved to:[/cyan] {args.output_img_folder}")


if __name__ == "__main__":
    main()
