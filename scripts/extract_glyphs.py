"""
Extract alphanumeric glyphs (A-Z, a-z, 0-9) from TTF fonts to SVG files.
"""

import logging
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from vecssl.util import setup_logging, get_console, make_progress

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

logger = logging.getLogger(__name__)
setup_logging()
console = get_console()


def extract_glyphs(ttf_path: Path, output_dir: Path) -> dict:
    """Extract glyphs from a single TTF file."""
    result = {"path": str(ttf_path), "extracted": 0, "missing": []}

    try:
        font = TTFont(ttf_path)
        glyph_set = font.getGlyphSet()
        cmap = font.getBestCmap()
        units_per_em = font["head"].unitsPerEm
        # ascender = font["hhea"].ascender

        font_name = ttf_path.stem
        font_output_dir = output_dir / font_name
        font_output_dir.mkdir(parents=True, exist_ok=True)

        for char in CHARS:
            code_point = ord(char)
            if code_point not in cmap:
                result["missing"].append(char)
                continue

            glyph_name = cmap[code_point]
            glyph = glyph_set[glyph_name]

            # Get glyph width
            width = font["hmtx"][glyph_name][0]

            # Draw glyph to SVG path
            from fontTools.pens.transformPen import TransformPen

            # Draw glyph to SVG path, but bake in the Y-flip + translate
            pen = SVGPathPen(glyph_set)

            # Matrix: (xx, xy, yx, yy, dx, dy)
            # x' = 1 * x + 0 * y + 0
            # y' = 0 * x + (-1) * y + units_per_em  => y' = units_per_em - y
            tpen = TransformPen(pen, (1, 0, 0, -1, 0, units_per_em))

            glyph.draw(tpen)
            path_data = pen.getCommands()

            if not path_data:
                result["missing"].append(char)
                continue

            # Write SVG with Y-axis flip and fill
            svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {units_per_em}">
  <path d="{path_data}" fill="black"/>
</svg>'''

            # Handle case-insensitive filesystems (macOS)
            if char.isupper():
                filename = f"upper_{char}.svg"
            elif char.islower():
                filename = f"lower_{char}.svg"
            else:
                filename = f"{char}.svg"
            svg_path = font_output_dir / filename
            svg_path.write_text(svg_content)
            result["extracted"] += 1

        font.close()

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = ArgumentParser(description="Extract glyphs from TTF fonts to SVG")
    parser.add_argument("--input", required=True, type=Path, help="Google Fonts directory")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for SVGs")
    parser.add_argument("--workers", default=8, type=int, help="Number of parallel workers")
    args = parser.parse_args()

    logger.info("[bold cyan]Starting glyph extraction...[/bold cyan]")

    # Find all TTF files
    ttf_files = []
    for subdir in ["ofl", "apache", "ufl"]:
        ttf_files.extend((args.input / subdir).rglob("*.ttf"))

    logger.info(f"[blue]Found [bold]{len(ttf_files)}[/bold] TTF files[/blue]")

    args.output.mkdir(parents=True, exist_ok=True)

    # Process fonts in parallel
    total_extracted = 0
    total_missing = 0
    errors = []

    progress = make_progress()

    with progress:
        task = progress.add_task("Extracting glyphs", total=len(ttf_files))

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(extract_glyphs, ttf, args.output): ttf for ttf in ttf_files}

            for future in as_completed(futures):
                result = future.result()
                total_extracted += result["extracted"]
                total_missing += len(result.get("missing", []))
                if "error" in result:
                    errors.append(result)
                progress.advance(task)

    # Summary
    logger.info("[bold green]Done![/bold green]")
    logger.info(f"  [green]Extracted:[/green] {total_extracted:,} SVGs")
    logger.info(f"  [yellow]Missing glyphs:[/yellow] {total_missing:,}")
    if errors:
        logger.info(f"  [red]Errors:[/red] {len(errors)} fonts failed")


if __name__ == "__main__":
    main()
