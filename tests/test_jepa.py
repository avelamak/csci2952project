import pytest
import torch
from unittest.mock import patch
from vecssl.models.jepa import SVGImageJepa, d_joint, N, D_row
from vecssl.models.base import TrainStep


@pytest.fixture
def dummy_batch():
    """Create a fake batch with random SVGs and images."""
    B = 2
    svg_batch = torch.randn(B, N, D_row)
    img_batch = torch.randn(B, 3, 224, 224)
    return {"svg": svg_batch, "img": img_batch}


class DummyImageEncoder(torch.nn.Module):
    """Mock image encoder to avoid downloading large pretrained models."""

    def __init__(self, output_dim=d_joint):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        B = x.size(0)
        return torch.randn(B, self.output_dim)


@pytest.fixture
def jepa_model():
    """Patch ImageEncoder with DummyImageEncoder before creating SVGImageJepa."""
    with patch("vecssl.models.jepa.ImageEncoder", new=DummyImageEncoder):
        model = SVGImageJepa()
        yield model


def test_model_init(jepa_model):
    """Check model initializes correctly."""
    assert isinstance(jepa_model, SVGImageJepa)
    assert hasattr(jepa_model, "svg_encoder")
    assert hasattr(jepa_model, "img_encoder")
    assert hasattr(jepa_model, "predictor")


def test_forward_pass_returns_trainstep(jepa_model, dummy_batch):
    """Forward pass returns a TrainStep with correct fields."""
    out = jepa_model(dummy_batch)
    assert isinstance(out, TrainStep)
    assert isinstance(out.loss, torch.Tensor)
    assert "mse_loss" in out.logs
    assert torch.isfinite(out.loss).all()


def test_latent_shapes(jepa_model, dummy_batch):
    """Check shapes of z_svg and z_img returned by encode_joint()."""
    outputs = jepa_model.encode_joint(dummy_batch)
    z_svg, z_img = outputs["svg"], outputs["img"]

    # Both should be [B, d_joint] now
    assert z_svg.shape == (dummy_batch["svg"].shape[0], d_joint)
    assert z_img.shape == (dummy_batch["img"].shape[0], d_joint)


def test_svg_encoder_output_shape(jepa_model):
    """Ensure SVG encoder produces correct shape for [B, N, D_row] input."""
    svg_input = torch.randn(2, N, D_row)
    z_svg = jepa_model.svg_encoder(svg_input)
    assert z_svg.shape == (2, d_joint)


def test_predictor_output_shape(jepa_model):
    """Check that predictor output matches the embedding dimension."""
    x = torch.randn(2, d_joint)
    z_pred = jepa_model.predictor(x)
    assert z_pred.shape == (2, d_joint)
