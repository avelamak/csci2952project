import pytest
import torch
from vecssl.models.base import TrainStep, JointModel


class TestTrainStep:
    def test_trainstep_creation_with_loss_only(self):
        loss = torch.tensor(1.5)
        step = TrainStep(loss=loss)
        assert step.loss is loss
        assert step.logs is None
        assert step.extras is None

    def test_trainstep_creation_with_logs(self):
        loss = torch.tensor(2.0)
        logs = {"accuracy": 0.95, "precision": 0.92}
        step = TrainStep(loss=loss, logs=logs)
        assert step.loss is loss
        assert step.logs == logs
        assert step.extras is None

    def test_trainstep_creation_with_all_fields(self):
        loss = torch.tensor(0.5)
        logs = {"metric": 0.8}
        extras = {"custom_data": [1, 2, 3], "flag": True}
        step = TrainStep(loss=loss, logs=logs, extras=extras)
        assert step.loss is loss
        assert step.logs == logs
        assert step.extras == extras

    def test_trainstep_loss_is_tensor(self):
        loss = torch.tensor(1.0)
        step = TrainStep(loss=loss)
        assert isinstance(step.loss, torch.Tensor)

    def test_trainstep_empty_logs(self):
        loss = torch.tensor(1.0)
        step = TrainStep(loss=loss, logs={})
        assert step.logs == {}


class DummyJointModel(JointModel):
    """Concrete implementation for testing"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, batch):
        # Simple implementation that uses the linear layer
        batch_size = batch.get("batch_size", 4)
        x = torch.randn(batch_size, 10)
        output = self.linear(x)
        loss = output.mean() ** 2
        logs = {"test_metric": float(loss.detach()) * 0.5}
        return TrainStep(loss=loss, logs=logs)

    def encode_joint(self, batch):
        # Simple implementation
        img_z = torch.randn(batch["batch_size"], 128)
        svg_z = torch.randn(batch["batch_size"], 128)
        # L2 normalize
        img_z = torch.nn.functional.normalize(img_z, p=2, dim=1)
        svg_z = torch.nn.functional.normalize(svg_z, p=2, dim=1)
        return {"img": img_z, "svg": svg_z}


class TestJointModel:
    def test_jointmodel_is_nn_module(self):
        model = DummyJointModel()
        assert isinstance(model, torch.nn.Module)

    def test_forward_returns_trainstep(self):
        model = DummyJointModel()
        batch = {"batch_size": 4}
        step = model(batch)
        assert isinstance(step, TrainStep)
        assert isinstance(step.loss, torch.Tensor)

    def test_forward_loss_has_grad(self):
        model = DummyJointModel()
        batch = {"batch_size": 4}
        step = model(batch)
        assert step.loss.requires_grad

    def test_encode_joint_returns_dict(self):
        model = DummyJointModel()
        batch = {"batch_size": 4}
        encodings = model.encode_joint(batch)
        assert isinstance(encodings, dict)
        assert "img" in encodings
        assert "svg" in encodings

    def test_encode_joint_output_shape(self):
        model = DummyJointModel()
        batch_size = 8
        batch = {"batch_size": batch_size}
        encodings = model.encode_joint(batch)
        assert encodings["img"].shape[0] == batch_size
        assert encodings["svg"].shape[0] == batch_size

    def test_encode_joint_normalized(self):
        model = DummyJointModel()
        batch = {"batch_size": 4}
        encodings = model.encode_joint(batch)
        # Check L2 normalization
        img_norms = torch.norm(encodings["img"], p=2, dim=1)
        svg_norms = torch.norm(encodings["svg"], p=2, dim=1)
        assert torch.allclose(img_norms, torch.ones_like(img_norms), atol=1e-6)
        assert torch.allclose(svg_norms, torch.ones_like(svg_norms), atol=1e-6)

    def test_base_jointmodel_forward_not_implemented(self):
        model = JointModel()
        batch = {}
        with pytest.raises(NotImplementedError):
            model(batch)

    def test_base_jointmodel_encode_joint_not_implemented(self):
        model = JointModel()
        batch = {}
        with pytest.raises(NotImplementedError):
            model.encode_joint(batch)

    def test_base_jointmodel_to_z_edit_not_implemented(self):
        model = JointModel()
        batch = {}
        with pytest.raises(NotImplementedError):
            model.to_z_edit(batch)

    def test_base_jointmodel_decode_svg_not_implemented(self):
        model = JointModel()
        z_edit = torch.randn(4, 128)
        mask = torch.ones(4, 100)
        with pytest.raises(NotImplementedError):
            model.decode_svg(z_edit, N_max=100, mask=mask)

    def test_model_can_be_moved_to_device(self):
        model = DummyJointModel()
        device = torch.device("cpu")
        model.to(device)
        # Check parameters are on correct device
        for param in model.parameters():
            assert param.device.type == device.type

    def test_model_gradients_work(self):
        model = DummyJointModel()
        batch = {"batch_size": 2}
        step = model(batch)
        step.loss.backward()
        # Check that gradients were computed
        assert model.linear.weight.grad is not None
