import torch
import logging
from vecssl.data.dataset import SVGXDataset
from vecssl.util import setup_logging

# Set print options to show full tensors
torch.set_printoptions(threshold=10000, linewidth=200, edgeitems=30)

# Setup logging
console = setup_logging(level="INFO", reset=True)
logger = logging.getLogger(__name__)

# Load dataset
logger.info("[cyan]Loading dataset...[/cyan]")
dataset = SVGXDataset(
    svg_dir="/Users/jz/work/csci2952project/svgx_svgs",
    img_dir="/Users/jz/work/csci2952project/svgx_imgs",
    meta_filepath="/Users/jz/work/csci2952project/svgx_meta.csv",
    max_num_groups=10,
    max_seq_len=80
)

logger.info(f"[green]Dataset size: {len(dataset)}[/green]")

# Get first sample
sample = dataset[0]

logger.info("\n[bold]Sample 0:[/bold]")
logger.info(f"UUID: {sample['uuid']}")
logger.info(f"Name: {sample['name']}")
logger.info(f"Commands shape: {sample['commands'].shape}")
logger.info(f"Args shape: {sample['args'].shape}")
logger.info(f"Number of tensors: {len(sample['tensors'])}")

# Check each tensor group
logger.info("\n[bold cyan]Tensor groups:[/bold cyan]")
for i, svg_tensor in enumerate(sample['tensors']):
    logger.info(f"\n[yellow]Group {i}:[/yellow]")
    logger.info(f"  seq_len: {svg_tensor.seq_len}")
    logger.info(f"  commands shape: {svg_tensor.commands.shape}")
    logger.info(f"  commands: {svg_tensor.commands.flatten()}")
    logger.info(f"  end_pos shape: {svg_tensor.end_pos.shape}")

    # Check after unpad
    cleaned = svg_tensor.copy().drop_sos().unpad()
    logger.info("  [dim]After drop_sos().unpad():[/dim]")
    logger.info(f"    data shape: {cleaned.data.shape}")
    if cleaned.data.size(0) > 0:
        logger.info(f"    First 3 rows:\n{cleaned.data[:3]}")
    else:
        logger.info("    [red]Empty tensor (padding group)[/red]")
