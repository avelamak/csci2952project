"""
Check discrepancy between metadata max_len_group and actual tensor length.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from vecssl.data.dataset import SVGXDataset
from vecssl.data.svg import SVG
from vecssl.data.svg_tensor import SVGTensor
from vecssl.util import setup_logging

logger = logging.getLogger(__name__)


def check_svg(svg_path: Path, verbose: bool = False):
    """Compare preprocessing length vs actual tensor length."""

    # Check if it's a .pt file (already preprocessed tensor)
    if svg_path.suffix == ".pt":
        data = torch.load(svg_path, weights_only=False)
        t_sep = data["t_sep"]
        fillings = data["fillings"]

        # Compute lengths from tensor
        len_groups_preprocess = [len(t) for t in t_sep if len(t) > 0]
        max_len_preprocess = max(len_groups_preprocess) if len_groups_preprocess else 0
    else:
        # Method 1: How preprocess_fonts.py computes it from SVG
        with open(svg_path) as f:
            svg_code = f.read()

        svg = SVG.load_svg(svg_code).to_path()
        len_groups_preprocess = [pg.total_len() for pg in svg.svg_path_groups]
        max_len_preprocess = max(len_groups_preprocess) if len_groups_preprocess else 0

        # Method 2: How dataset.py actually loads it
        SVGXDataset.preprocess(svg, augment=False)
        t_sep, fillings = svg.to_tensor(concat_groups=False), svg.to_fillings()

    # After add_sos + add_eos + pad (what dataset.py does)
    # Match dataset.py exactly: pad t_sep to MAX_NUM_GROUPS, process ALL including empty
    MAX_SEQ_LEN = 40  # default from dataset
    MAX_NUM_GROUPS = 8  # default from dataset

    # Pad to MAX_NUM_GROUPS like dataset.py does
    pad_len = max(MAX_NUM_GROUPS - len(t_sep), 0)
    t_sep.extend([torch.empty(0, 14)] * pad_len)
    fillings.extend([0] * pad_len)

    actual_lens = []
    cmds_shapes = []
    for t, f in zip(t_sep, fillings, strict=False):
        # Don't skip empty - process ALL groups like dataset.py does
        st = SVGTensor.from_data(t, PAD_VAL=-1, filling=f)
        st.add_eos().add_sos().pad(seq_len=MAX_SEQ_LEN + 2)
        actual_lens.append(len(st.commands))
        cmds_shapes.append(st.cmds().shape[0])  # flattened size

    max_len_actual = max(actual_lens) if actual_lens else 0

    # For .pt files, max_len_preprocess is already raw tensor len, so +2 for SOS/EOS
    # For .svg files, same logic applies
    expected_after_sos_eos = max_len_preprocess + 2
    discrepancy = max_len_actual - expected_after_sos_eos

    # Check if all cmds() have same flattened size (required for torch.stack)
    cmds_mismatch = len(set(cmds_shapes)) > 1 if cmds_shapes else False

    if verbose or discrepancy != 0 or cmds_mismatch:
        logger.info(f"{svg_path.name}:")
        logger.info(f"  Preprocess max_len_group: {max_len_preprocess}")
        logger.info(f"  Expected after SOS/EOS:   {max_len_preprocess + 2}")
        logger.info(f"  Actual tensor length:     {max_len_actual}")
        logger.info(f"  Group lengths (preprocess): {len_groups_preprocess}")
        logger.info(f"  Group lengths (actual):     {actual_lens}")
        logger.info(f"  cmds() flattened sizes:     {cmds_shapes}")
        if discrepancy != 0:
            logger.warning(f"  DISCREPANCY: {discrepancy}")
        if cmds_mismatch:
            logger.warning(f"  CMDS MISMATCH: {cmds_shapes}")

    return {
        "file": svg_path.name,
        "preprocess_max": max_len_preprocess,
        "expected": max_len_preprocess + 2,
        "actual": max_len_actual,
        "discrepancy": discrepancy,
        "cmds_mismatch": cmds_mismatch,
        "cmds_shapes": cmds_shapes,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--svg", type=Path, help="Single SVG file to check")
    parser.add_argument("--svg-dir", type=Path, help="Directory of SVGs to check")
    parser.add_argument("--meta", type=Path, help="Metadata CSV to cross-reference")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO", reset=True)

    if args.svg:
        check_svg(args.svg, verbose=True)
    elif args.svg_dir:
        # Find both .svg and .pt files
        svg_files = list(args.svg_dir.glob("*.svg")) + list(args.svg_dir.glob("*.pt"))
        svg_files = svg_files[: args.max_samples]
        logger.info(f"Checking {len(svg_files)} files...")

        results = []
        for svg_path in svg_files:
            try:
                result = check_svg(svg_path, verbose=args.verbose)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {svg_path}: {e}")

        # Summary
        discrepancies = [r for r in results if r["discrepancy"] != 0]
        cmds_mismatches = [r for r in results if r["cmds_mismatch"]]
        logger.info("=" * 50)
        logger.info(f"Total checked: {len(results)}")
        logger.info(f"Discrepancies: {len(discrepancies)}")
        logger.info(f"Cmds mismatches: {len(cmds_mismatches)}")

        if discrepancies:
            logger.warning("Files with discrepancies:")
            for d in discrepancies:
                logger.warning(
                    f"  {d['file']}: expected {d['expected']}, got {d['actual']} (diff: {d['discrepancy']})"
                )

        if cmds_mismatches:
            logger.warning("Files with cmds() size mismatch between groups:")
            for d in cmds_mismatches:
                logger.warning(f"  {d['file']}: {d['cmds_shapes']}")


if __name__ == "__main__":
    main()
