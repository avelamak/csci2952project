# Architecture Guide

We discuss the various model types and their implementation details inside VecSSL.


## Table of Contents

- [Overview](#overview)
- [JointModel Interface](#jointmodel-interface)
- [DeepSVG Components](#deepsvg-components)
  - [SVGEmbedding](#svgembedding)
  - [Encoder (Hierarchical)](#encoder-hierarchical)
  - [Decoder (Hierarchical)](#decoder-hierarchical)
  - [SVGLoss](#svgloss)

---

## Overview

### Why Hierarchical Architecture?

SVGs have a natural hierarchical structure:
```
SVG
  └─> Paths (stroke, fill, visibility)
       └─> Commands (moveto, lineto, cubic, arc, ...)
            └─> Arguments (coordinates, radii, flags)
```

The **2-stage hierarchical architecture** respects this structure:

1. **Stage 1 (Per-Path Encoding)**: Learn path-level semantics (what does each stroke represent?)
2. **Stage 2 (Hierarchical Encoding)**: Learn document-level composition (how do strokes combine?)

### Why Command-Argument Masking?

Different SVG commands use different argument subsets:

| Command | Arguments Used | Unused Args |
|---------|---------------|-------------|
| `moveto (m)` | `[start_x, start_y]` | 11 args unused |
| `lineto (l)` | `[start_x, start_y, end_x, end_y]` | 9 args unused |
| `cubic (c)` | All 8 position args | 5 args unused |
| `arc (a)` | All 13 arguments | None |

Computing loss on unused arguments adds noise and slows convergence. **CMD_ARGS_MASK** (`src/vecssl/data/svg_tensor.py`) defines valid arguments per command, ensuring loss is only computed where meaningful.

### Why Coordinate Quantization?

Coordinates are normalized to `[0, 255]` and represented as integers (256-way classification per argument).

**Benefits**:
1. **Easier Optimization**: Classification is more stable than regression for neural networks
2. **Bounded Predictions**: No outliers or extreme values during generation
3. **Implicit Regularization**: Discretization prevents overfitting to precise coordinates
---

## JointModel Interface

**Location**: `src/vecssl/models/base.py`

All SSL models must inherit from `JointModel` and implement this interface:

### Required Method

#### `forward(batch: dict) -> TrainStep`

Main training forward pass that computes loss and metrics.

**Input**: Batch dictionary from `SVGXDataset`:
```python
{
    "commands": Tensor[num_paths, seq_len],      # Command indices (0-6)
    "args": Tensor[num_paths, seq_len, 11],      # Quantized args (0-255)
    "tensors": List[SVGTensor],                  # With metadata
    "image": Tensor[3, H, W],                    # Normalized PNG [0,1]
    "uuid": str,
    "name": str,
    "source": str
}
```

**Output**: `TrainStep` dataclass:
```python
@dataclass
class TrainStep:
    loss: Tensor              # Scalar tensor for backpropagation
    logs: dict[str, float]    # Metrics for TensorBoard (e.g., {"loss/total": 2.3})
    extras: dict = None       # Optional data (embeddings, predictions, etc.)
```

**Example Implementation**:
```python
def forward(self, batch):
    # Encode SVG
    z_svg = self.svg_encoder(batch["commands"], batch["args"])

    # Encode image
    z_img = self.image_encoder(batch["image"])

    # Compute contrastive loss
    loss_contrastive = self.contrastive_loss(z_svg, z_img)

    # Log metrics
    logs = {
        "loss/total": loss_contrastive.item(),
        "temperature": self.logit_scale.exp().item()
    }

    return TrainStep(loss=loss_contrastive, logs=logs)
```

### Optional Methods

#### `encode_joint(batch: dict) -> dict[str, Tensor]`

Encode both modalities to joint embedding space (L2-normalized). Used for retrieval evaluation.

**Output**:
```python
{
    "svg": Tensor[batch, joint_dim],
    "img": Tensor[batch, joint_dim]
}
```

**Example**:
```python
def encode_joint(self, batch):
    # Encode and project to joint space
    z_svg = self.svg_projection(self.svg_encoder(...))
    z_img = self.img_projection(self.image_encoder(...))

    # L2 normalize for cosine similarity
    return {
        "svg": z_svg,
        "img": z_img
    }
```

#### `to_z_edit(batch: dict) -> Tensor`

Map to editing latent space (not necessarily normalized). Used for interpolation and manipulation.

**Output**: `Tensor[batch, dim_z]` (editing latent, may differ from joint embedding)

#### `decode_svg(z_edit: Tensor, N_max: int, mask: Tensor) -> dict`

Decode latent vector to SVG (command logits, argument logits, visibility logits).

**Output**:
```python
{
    "command_logits": Tensor[seq_len, batch, 7],
    "args_logits": Tensor[seq_len, batch, 11, 256],
    "visibility_logits": Tensor[num_paths, batch, 2]  # 2-stage only
}
```

---

## DeepSVG Components

### SVGEmbedding

**Location**: `src/vecssl/models/model.py`

Embeds SVG commands and arguments into `d_model`-dimensional space.

**Architecture**:
```
Commands [seq_len, batch]
  └─> Embedding(7, d_model)  # 7 command types

Arguments [seq_len, batch, 11]
  └─> 11 separate Embedding(256, 64)  # 11 args, each 0-255
  └─> Concat [seq_len, batch, 11*64=704]
  └─> Linear(704, d_model)

Position Indices [seq_len]
  └─> Embedding(max_seq_len, d_model)  # Learned positional encoding

Output = Command_emb + Args_emb + Pos_emb  # [seq_len, batch, d_model]
```

**Key Points**:
- **Separate argument embeddings**: Each of the 11 arguments gets its own 64-dim embedding, then concatenated
- **Learned positional encoding**
- **Additive combination**: Command + args + position embeddings are summed

---

### Encoder (Hierarchical)

**Location**: `src/vecssl/models/model.py`

Two-stage hierarchical encoding when `encode_stages=2`:

#### Stage 1: Per-Path Encoding

**Purpose**: Learn stroke-level semantics for each path independently.

**Input**: `[seq_len, num_paths * batch, d_model]` (all paths concatenated)

**Architecture**:
```
1. SVGEmbedding: Commands + Args → [seq_len, num_paths*batch, d_model]
2. TransformerEncoder (n_layers layers): Self-attention within each path
```
#### Stage 2: Hierarchical Encoding

**Purpose**: Learn SVG-level composition (how paths combine).

**Input**: `[num_paths, batch, d_model]` (path embeddings from Stage 1)

**Architecture**:
```
1. Positional encoding: Add learned position embeddings
2. TransformerEncoder (n_layers layers): Self-attention across paths
3. Mean pooling: [num_paths, batch, d_model] → [batch, d_model]  # Document embedding
```

#### Post-Processing

**Optional ResNet** (`use_resnet=True`):
```
Document embedding [batch, d_model]
  └─> 4x ResidualFC(d_model)  # BatchNorm → ReLU → Linear → Residual connection
  └─> [batch, d_model]
```

**VAE Bottleneck** (`use_vae=True`):
```
Document embedding [batch, d_model]
  └─> Linear → mu [batch, dim_z]
  └─> Linear → log_sigma [batch, dim_z]
  └─> Reparameterization: z = mu + exp(log_sigma) * epsilon  # epsilon ~ N(0,1)
  └─> z [batch, dim_z]
```

**Deterministic** (`use_vae=False`):
```
Document embedding [batch, d_model]
  └─> Linear → z [batch, dim_z]
```

**Code Reference**: `src/vecssl/models/model.py`

---

### Decoder (Hierarchical)

**Location**: `src/vecssl/models/model.py`

Two-stage hierarchical decoding when `decode_stages=2`:

#### Stage 1: Hierarchical Decoding

**Purpose**: Decode per-path latents and visibility from global latent.

**Input**: Global latent `z [batch, dim_z]`

**Architecture**:
```
1. Learnable queries: [num_paths, batch, d_model]  # One query per path slot
2. TransformerDecoder (n_layers_decode layers):
   - Self-attention: Queries attend to each other
   - Cross-attention: Queries attend to global latent z
3. Outputs:
   - Visibility logits: Linear → [num_paths, batch, 2]  # Does path exist?
   - Per-path latents: Linear → [num_paths, batch, dim_z]
```

**Visibility Prediction**: Binary classification (path visible or invisible). Allows model to use fewer than `max_num_groups` paths when appropriate.

#### Stage 2: Per-Path Decoding

**Purpose**: Decode commands and arguments for each path.

**Input**: Per-path latents `[num_paths, batch, dim_z]`

**Architecture**:
```
1. Learnable queries: [seq_len, num_paths*batch, d_model]  # One query per command slot
2. TransformerDecoder (n_layers_decode layers):
   - Self-attention: Queries attend to each other
   - Cross-attention: Queries attend to per-path latents
3. Outputs:
   - Command logits: Linear → [seq_len, num_paths*batch, 7]
   - Argument logits: Linear → [seq_len, num_paths*batch, 11, 256]
```

**Prediction Modes**:

- **One-Shot** (`pred_mode="one_shot"`):
  - Decode all commands in parallel using learned queries
  - Faster inference (single forward pass)
  - Non-autoregressive (cannot sample during generation)

- **Autoregressive** (`pred_mode="autoregressive"`): (*we are not using this*)
  - Decode commands sequentially, feeding predictions back as input
  - Slower inference (requires seq_len forward passes)
  - Enables sampling during generation (random or top-k)

**Code Reference**: `src/vecssl/models/model.py`

---

### SVGLoss

**Location**: `src/vecssl/models/loss.py`

Weighted sum of four loss components:

#### 1. KL Divergence Loss

**When**: Only if `use_vae=True`

**Formula**:
```python
kl_loss = -0.5 * mean(1 + log_sigma - mu^2 - exp(log_sigma))
kl_loss = max(0, kl_loss - kl_tolerance)  # Free bits: no penalty if KL < tolerance
```

**Purpose**: Regularize latent space to be close to N(0,1), enabling interpolation and generation.

**Weight**: `loss_kl_weight` (default: 1.0)

**KL Tolerance**: `kl_tolerance` (default: 0.1) - Prevents posterior collapse by allowing small KL divergence without penalty.

#### 2. Visibility Loss

**When**: Only if `decode_stages=2`

**Formula**:
```python
visibility_loss = BCEWithLogitsLoss(visibility_logits, visibility_targets)
```

**Purpose**: Learn which paths should be visible (exist) vs invisible (padding).

**Weight**: `loss_visibility_weight` (default: 1.0)

#### 3. Command Loss

**Formula**:
```python
command_loss = CrossEntropyLoss(command_logits, command_targets)
command_loss = command_loss * mask  # Mask out padding positions
```

**Purpose**: Predict correct command type (moveto, lineto, cubic, arc, etc.).

**Weight**: `loss_cmd_weight` (default: 1.0)

#### 4. Argument Loss

**Formula**:
```python
args_loss = CrossEntropyLoss(args_logits, args_targets)  # For each of 11 arguments
args_loss = args_loss * CMD_ARGS_MASK[commands]  # Mask out invalid args per command
args_loss = args_loss * mask  # Mask out padding positions
```

**Purpose**: Predict correct argument values (coordinates, radii, flags).

**Weight**: `loss_args_weight` (default: 2.0) - Higher than command loss because argument accuracy is more important for reconstruction quality.

**Total Loss**:
```python
total_loss = (loss_kl_weight * kl_loss +
              loss_visibility_weight * visibility_loss +
              loss_cmd_weight * command_loss +
              loss_args_weight * args_loss)
```

**Code Reference**: `src/vecssl/models/loss.py`
