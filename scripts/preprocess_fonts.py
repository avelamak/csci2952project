"""
Preprocess Google Fonts glyphs: simplify SVGs, render PNGs, save metadata with labels.
Input: raw SVGs from extract_glyphs.py
Output: preprocessed SVGs/tensors, 512x512 PNGs, metadata CSV
"""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import logging
import pandas as pd
import torch
import cairosvg

from vecssl.util import setup_logging, make_progress
from vecssl.data.svg import SVG
from vecssl.data.dataset import SVGXDataset

logger = logging.getLogger(__name__)

# Worker state
_OUT_SVG = None
_OUT_IMG = None
_TO_TENSOR = False


def char_to_label(char: str) -> int:
    """Convert character to label (DeepSVG convention)."""
    if char.isdigit():
        return ord(char) - 48  # 0-9 → 0-9
    elif char.isupper():
        return ord(char) - 65 + 10  # A-Z → 10-35
    else:
        return ord(char) - 97 + 36  # a-z → 36-61


def filename_to_char(filename: str) -> str:
    """Extract character from filename (upper_A.svg → A, lower_a.svg → a, 0.svg → 0)."""
    stem = Path(filename).stem
    if stem.startswith("upper_"):
        return stem.replace("upper_", "")
    elif stem.startswith("lower_"):
        return stem.replace("lower_", "")
    return stem  # digits


def _init_worker(output_svg_folder, output_img_folder, to_tensor):
    global _OUT_SVG, _OUT_IMG, _TO_TENSOR
    _OUT_SVG = output_svg_folder
    _OUT_IMG = output_img_folder
    _TO_TENSOR = to_tensor


def preprocess_glyph(svg_path: Path) -> dict | None:
    """Process a single glyph SVG."""
    try:
        font_name = svg_path.parent.name
        family_name = font_name.split("-")[0]
        char = filename_to_char(svg_path.name)
        label = char_to_label(char)
        uuid = f"{font_name}_{svg_path.stem}"

        # Load raw SVG
        with open(svg_path) as f:
            svg_code = f.read()

        # Render PNG from ORIGINAL raw SVG first (before vecssl processing)
        if _OUT_IMG:
            png_path = Path(_OUT_IMG) / f"{uuid}.png"
            cairosvg.svg2png(
                bytestring=svg_code.encode(),
                write_to=str(png_path),
                output_width=512,
                output_height=512,
                background_color="white",
            )

        # Then do vecssl processing for SVG/tensor output
        svg = SVG.load_svg(svg_code).to_path()
        svg.fill_(False)
        svg.normalize()
        svg.zoom(0.9)
        svg.canonicalize()
        # svg = svg.simplify_heuristic()

        # Compute metrics
        len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]

        metadata = {
            "uuid": uuid,
            "font_name": font_name,
            "family_name": family_name,
            "char": char,
            "label": label,
            "total_len": sum(len_groups),
            "nb_groups": len(len_groups),
            "max_len_group": max(len_groups) if len_groups else 0,
        }

        # Save preprocessed SVG or tensor
        if _OUT_SVG:
            if not _TO_TENSOR:
                out_path = Path(_OUT_SVG) / f"{uuid}.svg"
                svg.save_svg(str(out_path))
            else:
                SVGXDataset.preprocess(svg, augment=False)
                t_sep, fillings = svg.to_tensor(concat_groups=False), svg.to_fillings()
                out_path = Path(_OUT_SVG) / f"{uuid}.pt"
                torch.save({"t_sep": t_sep, "fillings": fillings}, out_path)

        return metadata

    except Exception as e:
        logger.warning(f"[red]Failed to process {svg_path}: {e}[/red]")
        return None


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input directory with raw SVGs from extract_glyphs",
    )
    parser.add_argument(
        "--output_meta_file", required=True, type=str, help="Path to save metadata CSV"
    )
    parser.add_argument(
        "--output_svg_folder", default=None, type=str, help="Folder to save preprocessed SVGs"
    )
    parser.add_argument("--output_img_folder", default=None, type=str, help="Folder to save PNGs")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--max_samples", default=None, type=int, help="Limit number of samples")
    parser.add_argument(
        "--to_tensor", action="store_true", help="Save as .pt tensors instead of .svg"
    )
    args = parser.parse_args()

    console = setup_logging(level="INFO", reset=True)
    logger.info("[bold cyan]Starting font glyph preprocessing...[/bold cyan]")

    # Create output folders
    if args.output_svg_folder:
        Path(args.output_svg_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"[dim]SVG output folder: {args.output_svg_folder}[/dim]")
    if args.output_img_folder:
        Path(args.output_img_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"[dim]Image output folder: {args.output_img_folder}[/dim]")

    # Find all SVG files
    svg_files = list(args.input.rglob("*.svg"))
    if args.max_samples:
        svg_files = svg_files[: args.max_samples]

    logger.info(
        f"[blue]Processing [bold]{len(svg_files)}[/bold] glyphs with [bold]{args.workers}[/bold] workers[/blue]"
    )

    # Process in parallel
    meta_data = []
    progress = make_progress(console)
    batch_size = args.workers * 32

    with progress:
        task = progress.add_task("Processing glyphs", total=len(svg_files))

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.output_svg_folder, args.output_img_folder, args.to_tensor),
        ) as executor:
            for batch_start in range(0, len(svg_files), batch_size):
                batch_end = min(batch_start + batch_size, len(svg_files))
                batch = svg_files[batch_start:batch_end]

                futures = {executor.submit(preprocess_glyph, p): p for p in batch}

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        meta_data.append(result)
                    progress.advance(task)

    # Save metadata
    df = pd.DataFrame(meta_data)
    Path(args.output_meta_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_meta_file, index=False)

    logger.info("[bold green]Done![/bold green]")
    logger.info(f"  [green]Processed:[/green] {len(meta_data):,}/{len(svg_files):,} glyphs")
    logger.info(f"  [cyan]Metadata:[/cyan] {args.output_meta_file}")
    if args.output_svg_folder:
        logger.info(f"  [cyan]SVGs:[/cyan] {args.output_svg_folder}")
    if args.output_img_folder:
        logger.info(f"  [cyan]Images:[/cyan] {args.output_img_folder}")


if __name__ == "__main__":
    main()
