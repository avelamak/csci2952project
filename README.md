# VecSSL: Self-Supervised Learning for Joint SVG-Image Embeddings


## Quick Start

### Setup

```bash
# Clone and setup environment
git clone https://github.com/avelamak/csci2952project
cd csci2952project
uv sync && uv pip install -e .
source .venv/bin/activate

# Install pre-commit hooks (required for contributors)
pre-commit install
```
### Download and process data
We are using [SVGX-Core-250k](https://huggingface.co/datasets/xingxm/SVGX-Core-250k). Download and preprocess this dataset with (this takes approximately 60-90 minutes depending on netowrk and cpu):
```
# Download and preprocess SVGX-Core-250k dataset
python scripts/preprocess.py \
    --output_meta_file svgx_meta.csv \
    --output_svg_folder svgx_svgs \
    --output_img_folder svgx_imgs \
    --workers <n_workers> \
    --max_samples <max_samples>  # defaults to entire dataset
```

### Test run with autoencoder

To quickly test the model layers (encoder & decoder) and data from previous step, you can run a few epochs on the following:
```
# Test SVG autoencoder
python scripts/test_svg_autoencoder.py \
     --svg-dir svgx_svgs \
     --img-dir svgx_imgs \
     --meta svgx_meta.csv \
     --epochs 2 \
     --batch-size 4
```

**Logging with Weights & Biases**: To enable wandb logging (in addition to TensorBoard), add the `--wandb-project` flag:
```bash
# Set your wandb API key
export WANDB_API_KEY=your_api_key_here
wandb login

# Run with wandb enabled
python scripts/test_svg_autoencoder.py \
     --svg-dir svgx_svgs \
     --img-dir svgx_imgs \
     --meta svgx_meta.csv \
     --epochs 2 \
     --batch-size 4 \
     --wandb-project "your-project-name" \
     --wandb-name "optional-run-name"
```

## Project Overview

### Goals

**Learn aligned representations of vector graphics and rasterized images through self-supervised learning.**

We combine DeepSVG's hierarchical transformer architecture for SVG encoding with modern SSL techniques (contrastive learning, MAE, JEPA) to create unified multi-modal embeddings that enable cross-modal retrieval, vector graphics generation from images, and semantic understanding of vector primitives.

We will enable:

- **Cross-modal retrieval**: Search SVGs using images and vice versa
- **Vector graphics generation**: Generate SVGs from image queries
- **Semantic understanding**: Learn meaningful representations of vector primitives
- **Latent space manipulation**: Interpolate and edit in joint embedding space



## Repository Structure

```
csci2952project/
├── src/vecssl/
│   ├── data/                        # SVG data pipeline (parsing, preprocessing, dataset)
│   ├── models/                      # Model architectures (DeepSVG, SSL methods, interface)
│   ├── trainer.py                   # Generic Trainer (works with any JointModel)
│   └── util.py                      # Logging and utilities
│
├── scripts/
│   ├── preprocess.py               # Download & preprocess SVGX dataset
│   ├── test_svg_autoencoder.py     # Test DeepSVG autoencoder
│   └── <...>.py's                   # Other training scripts
│
├── tests/                           # Unit tests (pytest)
└── pyproject.toml
```

## Key Components

### SVG Data Pipeline

We convert raw SVG files into tensor format following DeepSVG conventions. Each SVG is parsed into path groups, where each group consists of some number of commands, and each commands has some number of arguments, i.e. `[num_paths, seq_len, 14]`, where we set a fixed length of `14` for the arguments.

**See**: [src/vecssl/data/README.md](src/vecssl/data/README.md)

### DeepSVG Architecture

We use the hierarchical transformer encoder-decoder from DeepSVG. The default config uses a **2-stage encoder-decoder**, where SVG commands and arguments are first encoded per path group, and then those group latents are encoded into a single latent. The decoder does the same thing but in reverse. Note that there are extensive uses of masking due to padding all SVGs into same-sized tensors.

**See**: [src/vecssl/models/README.md](src/vecssl/models/README.md)

### SSL Methods

We implement self-supervised learning frameworks for SVG/Image joint embedding.

**See**: [src/vecssl/models/README.md](src/vecssl/models/README.md)

### Training Infrastructure

There is a generic trainer class and config that works with any SSL model implementing the `JointModel` interface.

**See**: [src/vecssl/README.md](src/vecssl/README.md)


## Development Workflow

### Pre-commits and CI

We are enforcing consistent formatting through pre-commits and CI with [Ruff](https://docs.astral.sh/ruff/) and `pytest`. To implement a new feature (e.g. specific SSL implementations) you can commit to a feature branch and open a PR.
```bash
# Install pre-commit hooks
pre-commit install

# Manual run (checks all files)
pre-commit run --all-files

# Run tests with coverage
pytest tests/ --cov=src/vecssl
```


### Implementing a New SSL Method

**Interface**: All SSL models must inherit from `JointModel` (see `src/vecssl/models/base.py`):

```python
from vecssl.models.base import JointModel, TrainStep

class MySSLModel(JointModel):
    def forward(self, batch) -> TrainStep:
        """Required: Compute SSL loss and return TrainStep"""
        loss = ...  # Your SSL loss (contrastive, MAE, JEPA, etc.)
        logs = {"loss/total": loss.item(), ...}
        return TrainStep(loss, logs)

    def encode_joint(self, batch) -> dict[str, Tensor]:
        """Optional: For retrieval evaluation"""
        z_svg = ...  # Encode SVG
        z_img = ...  # Encode image
        return {"svg": z_svg,
                "img": z_img}
```

## Acknowledgments

- **DeepSVG**: https://github.com/alexandre01/deepsvg
- **SVGX Dataset**: HuggingFace dataset `xingxm/SVGX-Core-250k`
