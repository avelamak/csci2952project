# csci2952project

## Overview

### Code organization

```
.
├── scripts
│   └── run_train.py        # entry point for training
├── src
│   ├── vecssl
│   │   ├── data
│   │   │   └── dataset.py  # SVG dataset
│   │   ├── models
│   │   │   ├── base.py     # base SSL arch
│   │   │   └── <...>       # other SSL archs (MAE, JEPA, etc)
│   │   ├── trainer.py      # base trainer class
│   │   └── util.py         # utils
└── tests                   # unit tests

```
### Dataset `vecssl/data/`
Dataset for SVG, we follow "DeepSVG"'s data processing.

### Model `vecssl/models/`
The file `base.py` implements `JointModel`, a base class/interface for SSL joint embedding model of vector and image data:

```python
class TrainStep:
    loss: ...
    logs: ...
    extras: ...

class JointModel(nn.Module):
    def forward() -> TrainStep:
        ...
    def encode_joint():
        ...
    def decode_joint():
        ...
```

Specific implementations of SSL architectures should inherit this class.

### Training `vecssl/trainer.py`

We implement a base trainer class that work for the base `JointModel`, thus working for all the specific SSL architectures by extension.


## Setup

We are using `uv` as a package manager, see [here](https://docs.astral.sh/uv/getting-started/installation/) for installation.

Run the following to setup

```
uv sync
uv pip install -e .
```

To run a script use

```
uv run python <script.py>
# OR activate venv first
source .venv/bin/activate
python <script.py>

```

## Running

...


## Code Structure

...
