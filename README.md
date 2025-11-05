# VecSSL: Self-Supervised Learning for Joint SVG-Image Embeddings

We apply SSL techniques to learn joint embedding spaces between vector graphics (SVG) and rasterized images. We explore contrastive learning, MAE, JEPA, and other SSL adjacent methods for multi-modal representation learning.

## Project Overview

**Goal**: Learn aligned representations of SVG vector graphics and their rendered images through self-supervised learning, enabling:
- Cross-modal retrieval (SVG↔image search)
- Vector graphics generation from images
- Semantic understanding of vector primitives
- Interpolation and editing in joint latent space

**Key Innovation**: Combining DeepSVG's hierarchical transformer architecture for SVG encoding with modern SSL techniques and image encoders to create a unified multi-modal representation space.

## Repository Structure

```
csci2952project/
├── src/vecssl/                      # Main package
│   ├── data/                        # Data loading and SVG processing
│   │   ├── dataset.py              # SVGXDataset (HuggingFace wrapper)
│   │   ├── svg.py                  # SVG class (parsing, transforms)
│   │   ├── svg_path.py             # Path operations (arcs, beziers)
│   │   ├── svg_tensor.py           # SVGTensor (14-column representation)
│   │   └── svg_primitive.py        # Geometric primitives (rect, circle, etc.)
│   │
│   ├── models/                      # Model architectures
│   │   ├── base.py                 # JointModel interface, TrainStep
│   │   ├── model.py                # DeepSVG components (Encoder, Decoder, VAE)
│   │   ├── config.py               # Configuration dataclass, presets
│   │   ├── loss.py                 # SVGLoss (KL, visibility, command, args)
│   │   ├── contrastive.py          # Contrastive SSL model (skeleton)
│   │   ├── basic_blocks.py         # FCN, HierarchFCN, ResNet
│   │   └── layers/                 # Transformer layers
│   │       ├── improved_transformer.py  # Pre-norm encoder/decoder
│   │       ├── positional_encoding.py   # Learned position embeddings
│   │       └── attention.py             # Multi-head attention
│   │
│   ├── trainer.py                   # Generic Trainer class
│   └── util.py                      # Logging setup (Rich, TensorBoard)
│
├── scripts/                         # Training and preprocessing
│   ├── preprocess.py               # Download and preprocess SVGX dataset
│   ├── test_svg_autoencoder.py     # Test DeepSVG autoencoder
│   └── run_train.py                # Main training entry point (TODO)
│
├── tests/                           # Unit tests
│   ├── test_models_base.py
│   ├── test_logging_setup.py
│   └── test_trainer.py
│
├── svgx_svgs/                       # Local SVG samples (100 files)
├── svgx_imgs/                       # Rendered PNG images (100 files)
├── svgx_meta.csv                    # Metadata (uuid, nb_groups, etc.)
├── runs/                            # TensorBoard logs
├── pyproject.toml                   # Dependencies (uv package manager)
└── .pre-commit-config.yaml          # Code quality hooks (Ruff)
```

## Setup

### Environment Setup

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for package management

```bash
# Install dependencies and activate environment
uv sync
uv pip install -e .

# Activate virtual environment
source .venv/bin/activate

# Install pre-commit hooks (please do this otherwise your commits will fail CI)
pre-commit install
```

### Data Setup

Download and preprocess the SVGX-Core-250k dataset from HuggingFace:

```bash
uv run python scripts/preprocess.py
```

This will:
1. Download ~250k SVG-image pairs from `xingxm/SVGX-Core-250k`
2. Parse SVGs to internal representation
3. Apply normalization, simplification, and numericalization
4. Save preprocessed data to `svgx_svgs/`, `svgx_imgs/`, `svgx_meta.csv`

**Note**: You can test it on a smaller set of SVGs instead of the full dataset, check arguments in `scripts/preprocess.py`.

## Data Infrastructure

### SVG Representation: SVGTensor

SVGs are represented as sequences of drawing commands using a 14-column tensor format (src/vecssl/data/svg_tensor.py):

**Format**: `[num_paths, seq_len, 14]`
- **Column 0**: Command index
  - 0=moveto, 1=lineto, 2=cubic Bézier, 3=arc, 4=EOS, 5=SOS, 6=closepath
- **Columns 1-13**: Command arguments (unused args set to -1)
  - `[radius_x, radius_y, x_axis_rot, large_arc_flag, sweep_flag, start_x, start_y, control1_x, control1_y, control2_x, control2_y, end_x, end_y]`

**Command-Argument Masking**: `CMD_ARGS_MASK[cmd_idx]` defines valid arguments per command. For example:
- `moveto` uses only `[start_x, start_y]`
- `cubic` uses `[start_x, start_y, control1_x, control1_y, control2_x, control2_y, end_x, end_y]`
- `arc` uses all 13 arguments

**Special Tokens**:
- SOS (Start-of-Sequence): Added at beginning of each path
- EOS (End-of-Sequence): Added at end of each path
- Padding: -1 for invalid positions (when paths shorter than max_seq_len)

### Dataset: SVGXDataset

**Class**: `src/vecssl/data/dataset.py:SVGXDataset`

**Parameters**:
- `data_dir`: Path to preprocessed SVG/image files
- `meta_path`: Path to metadata CSV
- `max_num_groups`: Maximum number of paths per SVG (default: 8)
- `max_seq_len`: Maximum commands per path (default: 40)
- `filter_uni`: Whether to filter by unicode range
- `filter_platform`: Filter by source platform
- `filter_category`: Filter by category

**Batch Format** (returned by `__getitem__`):
```python
{
    "commands": Tensor[num_paths, seq_len],      # Command indices (0-6)
    "args": Tensor[num_paths, seq_len, 11],      # Argument values (0-255, quantized)
    "tensors": List[SVGTensor],                  # With metadata (viewbox, etc.)
    "image": Tensor[3, H, W],                    # Rendered PNG (normalized to [0, 1])
    "uuid": str,                                 # Unique identifier
    "name": str,                                 # Human-readable name
    "source": str                                # Source dataset/platform
}
```

**Collation**: Custom `pad_collate` function handles variable-length sequences:
- Pads paths to `max_num_groups`
- Pads commands to `max_seq_len`
- Creates masks for loss computation

### Preprocessing Pipeline

**Location**: `src/vecssl/data/svg.py:SVG`

**Steps** (applied in sequence):
1. **Parse**: XML string → internal SVG representation
2. **To Path**: Convert primitives (rect, circle, ellipse, polygon) to path commands
3. **Normalize**: Scale coordinates to viewbox (typically 0-24)
4. **Zoom**: Apply zoom factor (default: 0.9)
5. **Canonicalize**: Standardize path representation (absolute coordinates)
6. **Simplify Heuristic**: Reduce path complexity using RDP-like algorithm
7. **Numericalize**: Quantize coordinates to 8-bit integers (0-255)

**Coordinate System**: All coordinates normalized to [0, 255] after preprocessing for stable neural network training.

## Model Architecture

### JointModel Interface

**Location**: `src/vecssl/models/base.py:JointModel`

All SSL models must inherit from `JointModel` and implement this interface:

**Required Method**:
- `forward(batch: dict) -> TrainStep`: Main training forward pass
  - Input: Batch dictionary from dataset
  - Output: `TrainStep(loss, logs, extras)`
    - `loss`: Scalar tensor for backpropagation
    - `logs`: Dict of metrics for logging (e.g., `{"loss/total": ..., "loss/contrastive": ...}`)
    - `extras`: Optional dict for additional data (embeddings, predictions, etc.)

**Optional Methods** (for evaluation and downstream tasks):
- `encode_joint(batch: dict) -> dict[str, Tensor]`: Encode modalities to joint space
  - Returns: `{"img": z_img, "svg": z_svg}` with L2-normalized embeddings
  - Used for retrieval evaluation (computing similarities)

- `to_z_edit(batch: dict) -> Tensor`: Map to editing latent space
  - Returns: Latent vector for interpolation/manipulation
  - Different from `encode_joint` (may not be normalized)

- `decode_svg(z_edit: Tensor, N_max: int, mask: Tensor) -> tuple`: Decode latent to SVG
  - Returns: Command logits, argument logits, visibility logits
  - Used for generation and reconstruction

### DeepSVG Components

The core SVG encoder/decoder is a hierarchical transformer architecture migrated from DeepSVG.

#### SVGEmbedding

**Location**: `src/vecssl/models/model.py:SVGEmbedding`

Embeds SVG commands and arguments into `d_model`-dimensional space:
- **Command embedding**: Lookup table for 7 commands → `d_model`
- **Argument embedding**: 11 separate embeddings (64-dim each) concatenated/projected → `d_model`
- **Positional encoding**: Learned position embeddings (not sinusoidal)
- **Output**: `[seq_len, batch, d_model]`

#### Encoder: Hierarchical SVG Encoding

**Location**: `src/vecssl/models/model.py:Encoder`

**Architecture**: Two-stage hierarchical encoding (when `encode_stages=2`):

**Stage 1: Per-Path Encoding**
- Input: `[seq_len, num_paths * batch, d_model]` (all paths concatenated)
- Transformer encoder layers with self-attention
- Output: Extract only the SOS token embedding from each path → `[batch, num_paths, d_model]`

**Stage 2: Hierarchical Encoding**
- Input: `[num_paths, batch, d_model]` (path embeddings)
- Transformer encoder layers with self-attention
- Output: Global document embedding → `[batch, d_model]`

**Optional**: ResNet post-processing (4 residual FC blocks with BatchNorm + ReLU)

**Bottleneck**: VAE or deterministic
- **VAE** (`use_vae=True`): Predict `mu`, `log_sigma` → reparameterization trick → `z [batch, dim_z]`
- **Deterministic** (`use_vae=False`): Direct projection to `z [batch, dim_z]`

**Alternative**: 1-stage encoding (`encode_stages=1`) treats all commands as flat sequence (no hierarchy).

#### Decoder: Hierarchical SVG Decoding

**Location**: `src/vecssl/models/model.py:Decoder`

**Architecture**: Two-stage hierarchical decoding (when `decode_stages=2`):

**Stage 1: Hierarchical Decoding**
- Input: Global latent `z [batch, dim_z]`
- Queries: Learnable embeddings `[num_paths, batch, d_model]` (one per path slot)
- Transformer decoder with cross-attention to `z`
- Outputs:
  - Visibility logits: `[num_paths, batch, 2]` (does path exist?)
  - Per-path latents: `[num_paths, batch, dim_z]` (latent for each path)

**Stage 2: Per-Path Decoding**
- Input: Per-path latents `[num_paths, batch, dim_z]`
- Queries: Learnable embeddings `[seq_len, num_paths * batch, d_model]` (one per command slot)
- Transformer decoder with cross-attention to per-path latents
- Outputs:
  - Command logits: `[seq_len, num_paths * batch, 7]`
  - Argument logits: `[seq_len, num_paths * batch, 11, 256]` (11 args × 256-way classification)

**Prediction Modes**:
- **One-shot** (`pred_mode="one_shot"`): Decode all commands in parallel (non-autoregressive)
- **Autoregressive** (`pred_mode="autoregressive"`): Feed predicted commands back as input for next step

**Alternative**: 1-stage decoding (`decode_stages=1`) predicts all commands in flat sequence.

#### Loss Function: SVGLoss

**Location**: `src/vecssl/models/loss.py:SVGLoss`

**Components** (weighted sum):

1. **KL Divergence** (`loss_kl_weight=1.0`):
   - Formula: `-0.5 * mean(1 + log_sigma - mu^2 - exp(log_sigma))`
   - Clamped to `kl_tolerance=0.1` (no penalty if KL < tolerance)
   - Only computed if `use_vae=True`

2. **Visibility Loss** (`loss_visibility_weight=1.0`):
   - Binary cross-entropy on path existence predictions
   - Only computed for 2-stage decoder

3. **Command Loss** (`loss_cmd_weight=1.0`):
   - Cross-entropy on command predictions
   - Masked to valid (non-padding) positions using `mask [seq_len, batch]`

4. **Argument Loss** (`loss_args_weight=2.0`):
   - Cross-entropy on argument value predictions
   - Masked by `CMD_ARGS_MASK` (only penalize valid args per command)
   - Higher weight than command loss (arguments more important for reconstruction quality)

**Total Loss**: `loss_kl + loss_visibility + loss_cmd + loss_args`

### Configuration

**Location**: `src/vecssl/models/config.py:_DefaultConfig`

Key hyperparameters (dataclass-based):
```python
# Model architecture
d_model: 256                   # Transformer hidden dimension
n_heads: 8                     # Number of attention heads
n_layers: 4                    # Encoder layers
n_layers_decode: 4             # Decoder layers
dim_feedforward: 512           # FFN hidden dimension
dim_z: 256                     # Latent space dimension
dropout: 0.1

# Data format
max_num_groups: 8              # Max paths per SVG
max_seq_len: 30                # Max commands per path

# Architecture choices
encode_stages: 2               # 1 (flat) or 2 (hierarchical)
decode_stages: 2               # 1 (flat) or 2 (hierarchical)
use_vae: True                  # VAE vs deterministic bottleneck
pred_mode: "one_shot"          # "one_shot" or "autoregressive"
use_resnet: True               # Extra FC layers after encoder

# Loss weights
loss_kl_weight: 1.0
loss_cmd_weight: 1.0
loss_args_weight: 2.0
loss_visibility_weight: 1.0
kl_tolerance: 0.1
```

**Presets**: Predefined configurations (SketchRNN, Sketchformer, Hierarchical) in `config.py`.

## Training Infrastructure

### Trainer Class

**Location**: `src/vecssl/trainer.py:Trainer`

Generic trainer that works with any `JointModel` implementation.

**Features**:
- **Automatic Mixed Precision (AMP)**: Reduces memory usage, speeds up training on modern GPUs
- **Gradient Clipping**: Optional `max_grad_norm` to prevent exploding gradients
- **Learning Rate Scheduling**: Configurable LR scheduler (step, cosine, etc.)
- **TensorBoard Logging**: Real-time metrics tracking (`runs/` directory)
- **Rich Console Output**: Color-coded progress bars with live loss updates
- **Validation**: Automatic evaluation every epoch
- **Device Management**: Automatic GPU detection and model placement

**Constructor Parameters**:
```python
Trainer(
    model: JointModel,              # Any SSL model inheriting JointModel
    train_loader: DataLoader,       # Training data
    val_loader: DataLoader,         # Validation data
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str = "cuda",
    use_amp: bool = True,
    max_grad_norm: float = None,    # Gradient clipping threshold
    log_dir: str = "runs"           # TensorBoard output directory
)
```

**Training Loop** (simplified):
```python
trainer = Trainer(model, train_loader, val_loader, optimizer, num_epochs=100)
trainer.train()  # Runs full training loop with validation
```

**Logged Metrics** (to TensorBoard):
- `train/loss`: Total loss per batch
- `train/{key}`: Any metric in `logs` dict from `TrainStep`
- `val/loss`: Validation loss per epoch
- `val/{key}`: Validation metrics

### TrainStep Pattern

**Location**: `src/vecssl/models/base.py:TrainStep`

A structured way to return training information from `forward()`:

```python
@dataclass
class TrainStep:
    loss: Tensor              # Scalar loss for backpropagation
    logs: dict[str, float]    # Metrics to log (e.g., loss components)
    extras: dict = None       # Optional data (embeddings, predictions, etc.)
```

**Example Usage in SSL Models**:
```python
def forward(self, batch):
    # Compute loss components
    loss_contrastive = ...
    loss_regularization = ...

    # Total loss
    loss = loss_contrastive + 0.1 * loss_regularization

    # Metrics for logging
    logs = {
        "loss/total": loss.item(),
        "loss/contrastive": loss_contrastive.item(),
        "loss/reg": loss_regularization.item(),
        "temperature": self.logit_scale.exp().item()
    }

    return TrainStep(loss, logs)
```

### Logging Setup

**Location**: `src/vecssl/util.py:setup_logging`

- **Rich console logging**: Color-coded levels, timestamps, markup support
- **TensorBoard integration**: Automatic metric writing via `SummaryWriter`
- **Progress bars**: Live training progress with loss display

## Implementing SSL Methods

### What's Needed

To implement a new SSL method (Contrastive, MAE, JEPA, etc.), you need:

1. **Image Encoder**: A neural network to encode rasterized images
   - Suggested architectures: ResNet-18/50, ViT, ConvNeXt
   - Output: Image embedding `[batch, img_dim]`

2. **Projection Heads**: Map modality-specific embeddings to joint space
   - SVG projection: `dim_z → joint_dim` (typically 512)
   - Image projection: `img_dim → joint_dim`
   - Often a 2-3 layer MLP with BatchNorm/LayerNorm

3. **Joint Embedding Logic**: Implement `encode_joint()` method
   - Encode SVG: Use DeepSVG encoder → `z_svg [batch, dim_z]` → project → L2-normalize
   - Encode image: Image encoder → `z_img [batch, img_dim]` → project → L2-normalize
   - Return: `{"svg": z_svg_normalized, "img": z_img_normalized}`

4. **SSL-Specific Loss**: Implement in `forward()` method
   - Contrastive: InfoNCE loss with temperature scaling
   - MAE: Mask paths/tokens → reconstruct → MSE/L1 loss
   - JEPA: Predict SVG embedding from image embedding

### Interface Requirements

**Minimal Implementation**:
```python
from vecssl.models.base import JointModel, TrainStep

class MySSLModel(JointModel):
    def __init__(self, config):
        super().__init__()
        # Initialize image encoder, projection heads, etc.

    def forward(self, batch) -> TrainStep:
        # Compute SSL loss (contrastive, MAE, JEPA, etc.)
        loss = ...
        logs = {"loss/total": loss.item(), ...}
        return TrainStep(loss, logs)

    def encode_joint(self, batch) -> dict[str, Tensor]:
        # For retrieval evaluation
        z_svg = ...  # Encode SVG using DeepSVG encoder + projection
        z_img = ...  # Encode image using image encoder + projection
        return {
            "svg": F.normalize(z_svg, dim=-1),
            "img": F.normalize(z_img, dim=-1)
        }
```

**Optional Methods**: Implement `to_z_edit()` and `decode_svg()` if you want:
- Latent space interpolation
- SVG generation from learned representations
- Editing and manipulation tasks

### Design Patterns

**Contrastive Learning** (CLIP-style):
- Compute similarity matrix: `logits = z_img @ z_svg.T / temperature`
- InfoNCE loss: Symmetric cross-entropy (image→svg + svg→image)
- Learnable temperature: `logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))`
- See `src/vecssl/models/contrastive.py` for skeleton

**Masked Autoencoder (MAE)**:
- Mask random paths or tokens (e.g., 50-75% masking ratio)
- Encode visible tokens using DeepSVG encoder
- Decode all tokens (visible + masked) using DeepSVG decoder
- Reconstruction loss on masked tokens only

**Joint Embedding Predictive Architecture (JEPA)**:
- Encode image → target embedding (stop gradient)
- Encode SVG → context embedding
- Predict target from context (e.g., via predictor network)
- Loss: MSE between predicted and target embeddings

**Dual-VAE** (DeepSVG baseline for comparison):
- Separate VAE for SVG reconstruction (no image)
- Use DeepSVG encoder/decoder as-is
- See `scripts/test_svg_autoencoder.py:SimpleSVGAutoencoder` for example

## Testing and Evaluation

### Unit Tests

**Location**: `tests/`

Existing tests (minimal coverage, needs expansion):
- `test_models_base.py`: JointModel interface tests
- `test_logging_setup.py`: Logging configuration tests
- `test_trainer.py`: Trainer class tests

**Run Tests**:
```bash
pytest tests/
pytest tests/ --cov=src/vecssl  # With coverage
```

### Test Training Script

**Location**: `scripts/test_svg_autoencoder.py`

Minimal example of training a DeepSVG autoencoder (SVG-only, no images):
- Wraps DeepSVG components in `SimpleSVGAutoencoder`
- Uses `DebugTrainer` with gradient checking
- Tests forward/backward pass on small batch

**Usage**: Verify DeepSVG components work correctly before adding SSL complexity.

### Evaluation Metrics (TODO)

**Reconstruction Quality**:
- Command accuracy: Percentage of correctly predicted commands
- Argument MAE: Mean absolute error on argument values
- Chamfer distance: Between predicted and ground truth path points
- IoU: Intersection-over-union of rendered images

**Retrieval Performance** (for contrastive/JEPA):
- Recall@K: Percentage of queries where ground truth is in top-K results
- Mean reciprocal rank (MRR)
- Requires implementing `encode_joint()` for all SSL methods

**Generation Quality** (for MAE/VAE):
- FID: Fréchet Inception Distance on rendered images
- Qualitative visualization of reconstructions and interpolations

## Key Design Decisions

### Hierarchical Encoding

**Why**: SVGs are naturally hierarchical (document → paths → commands). The 2-stage architecture respects this structure:
- Stage 1 learns per-path representations (stroke semantics)
- Stage 2 learns document-level composition

**Alternative**: 1-stage (flat) treats all commands equally, losing compositionality.

### Command-Argument Masking

**Why**: Different commands use different subsets of 13 possible arguments. `CMD_ARGS_MASK` ensures loss is only computed on valid arguments, improving training stability and convergence.

**Example**: Lineto only uses `[start_x, start_y, end_x, end_y]`, so loss on `[radius_x, control1_x, ...]` is masked out.

### One-Shot vs Autoregressive Decoding

**One-shot** (`pred_mode="one_shot"`):
- Predicts all commands in parallel using learned queries
- Faster inference (single forward pass)
- Better for retrieval and representation learning (fixed computation graph)

**Autoregressive** (`pred_mode="autoregressive"`):
- Predicts commands sequentially, feeding predictions back as input
- More flexible for generation (can sample)
- Slower inference (sequential)

**Default**: One-shot for SSL training, autoregressive can be enabled for generation tasks.

### Coordinate Quantization

**Why**: Coordinates normalized to [0, 255] and quantized to integers (256-way classification per argument):
- Converts regression problem to classification (easier to train)
- Bounded prediction space (no outliers)
- Discretization provides implicit regularization

**Alternative**: Regression (predict continuous values) is more expressive but harder to optimize.

### Shape Conventions

Tensors use `(seq_len, batch, d_model)` convention (not `(batch, seq_len, d_model)`):
- **Reason**: PyTorch's TransformerEncoder expects this format for efficiency
- **Note**: Some layers (e.g., ResNet) use `(batch, d_model)` and require transposition

## References

### DeepSVG
- **Original Code**: https://github.com/alexandre01/deepsvg

### SSL Methods
...

### Dataset

- **SVGX-Core-250k**: HuggingFace dataset `xingxm/SVGX-Core-250k`
- Contains ~250k emoji icons from noto-emoji project
- Each sample: SVG file + rendered PNG + metadata

## Contributing

### Code Quality

- **Linting**: Ruff (100+ rules enabled)
- **Formatting**: Ruff auto-format on commit
- **Type Checking**: mypy (not yet enforced)
- **Testing**: pytest with coverage

**Pre-commit hooks**: Automatically run checks before commits:
```bash
pre-commit install
pre-commit run --all-files  # Manual run
```

### Development Workflow

1. Create feature branch from `main`
2. Implement SSL method (inherit `JointModel`, implement `forward()` and optionally `encode_joint()`)
3. Add unit tests for new components
4. Test training on small dataset (100 samples)
5. Document hyperparameters and design choices
6. Open pull request with results and visualizations
