# Data Pipeline Guide

This document explains how SVG files are processed, represented, and loaded for training in VecSSL.

---

## Table of Contents

- [SVGTensor Format](#svgtensor-format)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Dataset: SVGXDataset](#dataset-svgxdataset)
- [Batch Format and Collation](#batch-format-and-collation)

---

## SVGTensor Format

**Location**: `src/vecssl/data/svg_tensor.py`

SVGs are represented as sequences of drawing commands using a **14-column tensor format**.

### Tensor Shape

```
[num_paths, seq_len, 14]
```

- **num_paths**: Number of paths in the SVG (padded to `max_num_groups`)
- **seq_len**: Number of commands per path (padded to `max_seq_len`)
- **14 columns**: Command index (1 column) + Arguments (13 columns)

### Column Structure

| Column | Name | Type | Values | Description |
|--------|------|------|--------|-------------|
| 0 | Command | int | 0-6 | Command index (m=0, l=1, c=2, a=3, EOS=4, SOS=5, z=6) |
| 1 | radius_x | float | 0-255 | Arc x-radius (quantized) |
| 2 | radius_y | float | 0-255 | Arc y-radius (quantized) |
| 3 | x_axis_rot | float | 0-255 | Arc rotation angle (quantized) |
| 4 | large_arc_flag | float | 0 or 1 | Arc large-arc-flag |
| 5 | sweep_flag | float | 0 or 1 | Arc sweep-flag |
| 6 | start_x | float | 0-255 | Start point x-coordinate (quantized) |
| 7 | start_y | float | 0-255 | Start point y-coordinate (quantized) |
| 8 | control1_x | float | 0-255 | First control point x (quantized) |
| 9 | control1_y | float | 0-255 | First control point y (quantized) |
| 10 | control2_x | float | 0-255 | Second control point x (quantized) |
| 11 | control2_y | float | 0-255 | Second control point y (quantized) |
| 12 | end_x | float | 0-255 | End point x-coordinate (quantized) |
| 13 | end_y | float | 0-255 | End point y-coordinate (quantized) |

### Command Types

| Command | Index | SVG Path Command | Arguments Used |
|---------|-------|------------------|----------------|
| moveto | 0 | `M` or `m` | `[start_x, start_y]` (2 args) |
| lineto | 1 | `L` or `l` | `[start_x, start_y, end_x, end_y]` (4 args) |
| cubic Bézier | 2 | `C` or `c` | `[start_x, start_y, control1_x, control1_y, control2_x, control2_y, end_x, end_y]` (8 args) |
| arc | 3 | `A` or `a` | All 13 arguments |
| EOS (End-of-Sequence) | 4 | N/A | No arguments (marks path end) |
| SOS (Start-of-Sequence) | 5 | N/A | No arguments (marks path start) |
| closepath | 6 | `Z` or `z` | No arguments |

### Special Tokens

- **SOS (Start-of-Sequence)**: Added at the beginning of each path, serves as aggregation token for path embeddings
- **EOS (End-of-Sequence)**: Added at the end of each path, marks path termination
- **Padding**: Invalid positions filled with command=EOS and args=-1

### Command-Argument Masking

**CMD_ARGS_MASK** (`src/vecssl/data/svg_tensor.py:14`) defines which arguments are valid per command:

```python
CMD_ARGS_MASK = {
    0: [0,0,0,0,0,1,1,0,0,0,0,0,0],  # moveto: only start_x, start_y
    1: [0,0,0,0,0,1,1,0,0,0,0,1,1],  # lineto: start + end
    2: [0,0,0,0,0,1,1,1,1,1,1,1,1],  # cubic: start + control1 + control2 + end
    3: [1,1,1,1,1,1,1,0,0,0,0,1,1],  # arc: all 13 args
    4: [0,0,0,0,0,0,0,0,0,0,0,0,0],  # EOS: no args
    5: [0,0,0,0,0,0,0,0,0,0,0,0,0],  # SOS: no args
    6: [0,0,0,0,0,0,0,0,0,0,0,0,0],  # closepath: no args
}
```

**Purpose**: Loss is only computed on valid (non-zero) arguments for each command type, preventing noise from unused arguments.

### Example: Lineto Command

```python
# Lineto command: Move from (50, 50) to (100, 150)
tensor = [
    1,    # Command: lineto (index 1)
    -1,   # radius_x (unused)
    -1,   # radius_y (unused)
    -1,   # x_axis_rot (unused)
    -1,   # large_arc_flag (unused)
    -1,   # sweep_flag (unused)
    50,   # start_x (quantized to 0-255 range)
    50,   # start_y
    -1,   # control1_x (unused)
    -1,   # control1_y (unused)
    -1,   # control2_x (unused)
    -1,   # control2_y (unused)
    100,  # end_x
    150   # end_y
]
```

---

## Preprocessing Pipeline

**Location**: `src/vecssl/data/svg.py`

Raw SVG files undergo a 7-step preprocessing pipeline to convert them into neural network-friendly tensors.

### Pipeline Steps

#### 1. **Parse** (`parse()`)

Convert XML string to internal `SVG` representation.

**Input**: Raw SVG file (XML string)
```xml
<svg viewBox="0 0 24 24">
  <rect x="5" y="5" width="10" height="10" fill="blue"/>
  <circle cx="12" cy="12" r="5" fill="red"/>
</svg>
```

**Output**: `SVG` object with parsed elements (paths, primitives, metadata)

**Code**: `src/vecssl/data/svg.py:80`

#### 2. **To Path** (`to_path()`)

Convert primitives (rect, circle, ellipse, polygon, polyline) to path commands.

**Example**:
- `<rect x="5" y="5" width="10" height="10"/>` → `M 5 5 L 15 5 L 15 15 L 5 15 Z`
- `<circle cx="12" cy="12" r="5"/>` → Approximated with cubic Bézier curves

**Code**: `src/vecssl/data/svg_primitive.py`

#### 3. **Normalize** (`normalize()`)

Scale coordinates to viewbox (typically 0-24 range).

**Purpose**: Standardize coordinate ranges across SVGs with different viewbox sizes.

**Formula**:
```python
# If viewbox is (0, 0, 100, 100) and we want (0, 0, 24, 24):
x_normalized = (x - viewbox_x) * (24 / viewbox_width)
y_normalized = (y - viewbox_y) * (24 / viewbox_height)
```

**Code**: `src/vecssl/data/svg.py:280`

#### 4. **Zoom** (`zoom()`)

Apply zoom factor (default: 0.9) to add margin around graphics.

**Purpose**: Prevent clipping at image edges during rendering.

**Formula**:
```python
center = 12  # Center of 24x24 canvas
x_zoomed = center + (x - center) * zoom_factor
```

**Code**: `src/vecssl/data/svg.py:316`

#### 5. **Canonicalize** (`canonicalize()`)

Convert relative coordinates to absolute coordinates.

**Example**:
- `m 10 10 l 5 5` (relative) → `M 10 10 L 15 15` (absolute)

**Purpose**: Simplify representation (all commands use absolute coordinates).

**Code**: `src/vecssl/data/svg.py:228`

#### 6. **Simplify Heuristic** (`simplify_heuristic()`)

Reduce path complexity using RDP-like (Ramer-Douglas-Peucker) algorithm.

**What it does**:
- Merge close points (distance < threshold)
- Remove redundant commands (e.g., consecutive lineto to same point)
- Simplify Bézier curves with low curvature to lineto

**Purpose**: Reduce sequence length, making training more efficient.

**Code**: `src/vecssl/data/svg.py:349`

#### 7. **Numericalize** (`numericalize()`)

Quantize coordinates to 8-bit integers (0-255).

**Formula**:
```python
# Coordinates in [0, 24] range
quantized = int(coordinate * 256 / 24)  # Map to [0, 255]
```

**Purpose**: Convert continuous coordinates to discrete classification problem (256-way for each argument).

**Code**: `src/vecssl/data/svg.py:453`

### Running Preprocessing

**Script**: `scripts/preprocess.py`

```bash
uv run python scripts/preprocess.py \
    --output_meta_file svgx_meta.csv \
    --output_svg_folder svgx_svgs \
    --output_img_folder svgx_imgs \
    --max_samples 1000  # Optional: limit samples
```

**What it does**:
1. Downloads SVGX-Core-250k from HuggingFace (`xingxm/SVGX-Core-250k`)
2. Applies 7-step preprocessing pipeline to each SVG
3. Renders PNG images using CairoSVG (64x64 by default)
4. Saves preprocessed SVGs, images, and metadata to disk

**Output Files**:
- `svgx_svgs/{uuid}.svg`: Preprocessed SVG files
- `svgx_imgs/{uuid}.png`: Rendered PNG images (64x64)
- `svgx_meta.csv`: Metadata (uuid, nb_groups, max_len_group, etc.)

---

## Dataset: SVGXDataset

**Location**: `src/vecssl/data/dataset.py`

PyTorch dataset for loading preprocessed SVG-image pairs.
```python
SVGXDataset(
    svg_dir: str,              # Path to SVG folder
    img_dir: str,              # Path to PNG folder
    meta_filepath: str,        # Path to metadata CSV
    max_num_groups: int = 8,   # Max paths per SVG
    max_seq_len: int = 40,     # Max commands per path
    pad_val: int = -1          # Value used for padding
    train_ratio: float = 1.0,         # Train/test split ratio
    already_preprocessed: bool = True  # i.e. we already ran preprocess.py
)
```

### Usage Example

```python
from vecssl.data.dataset import SVGXDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = SVGXDataset(
    svg_dir="svgx_svgs",
    img_dir="svgx_imgs",
    meta_filepath="svgx_meta.csv",
    max_num_groups=8,
    max_seq_len=30,
    mode="train"
)

# Create dataloader with custom collation
from vecssl.data.dataset import pad_collate
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=pad_collate,
    num_workers=4
)

# Iterate
for batch in dataloader:
    commands = batch["commands"]      # [batch, num_paths, seq_len]
    args = batch["args"]              # [batch, num_paths, seq_len, 11]
    images = batch["image"]           # [batch, 3, H, W]
    # ...
```

## Batch Format and Collation

### Batch Dictionary

The dataset returns a dictionary with these keys:

```python
{
    # SVG data (neural network format)
    "commands": Tensor[batch, num_paths, seq_len],      # Command indices (0-6)
    "args": Tensor[batch, num_paths, seq_len, 11],      # Argument values (0-255, quantized)

    # SVG data (original format for visualization/debugging)
    "tensors": List[SVGTensor],                          # SVGTensor objects with metadata

    # Image data
    "image": Tensor[batch, 3, H, W],                     # Rendered PNG (normalized to [0, 1])

    # Metadata
    "uuid": List[str],                                   # Unique identifiers
    "name": List[str],                                   # Human-readable names
    "source": List[str]                                  # Source dataset/platform
}
```

### Custom Collation: `custom_collate()`

**Location**: `scripts/test_svg_autoencoder.py`

**Purpose**: Handle collation of different data types in batch, just stacking manually
