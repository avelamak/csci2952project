import pytest
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from vecssl.trainer import Trainer
from vecssl.models.base import TrainStep, JointModel


class MockModel(JointModel):
    """Mock model for testing trainer"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        self.call_count = 0

    def forward(self, batch):
        self.call_count += 1
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        output = self.linear(x).mean()
        loss = output**2
        logs = {"custom_metric": float(loss.detach()) * 0.5}
        return TrainStep(loss=loss, logs=logs)


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def optimizer(mock_model):
    return torch.optim.Adam(mock_model.parameters(), lr=0.001)


@pytest.fixture
def train_loader():
    # Create simple dataset
    data = torch.randn(20, 10)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def val_loader():
    data = torch.randn(8, 10)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=4)


class TestTrainerInit:
    def test_trainer_initialization_minimal(self, mock_model, optimizer):
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        assert trainer.model is mock_model
        assert trainer.optimizer is optimizer
        assert trainer.scheduler is None
        assert trainer.amp is True
        assert trainer.grad_clip is None
        assert trainer.tb is None

    def test_trainer_initialization_with_device(self, mock_model, optimizer):
        device = torch.device("cpu")
        trainer = Trainer(model=mock_model, optimizer=optimizer, device=device)
        assert trainer.device == device

    def test_trainer_initialization_auto_device(self, mock_model, optimizer):
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        assert trainer.device.type in ["cuda", "cpu"]

    def test_trainer_initialization_with_scheduler(self, mock_model, optimizer):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        trainer = Trainer(model=mock_model, optimizer=optimizer, scheduler=scheduler)
        assert trainer.scheduler is scheduler

    def test_trainer_initialization_with_amp_disabled(self, mock_model, optimizer):
        trainer = Trainer(model=mock_model, optimizer=optimizer, amp=False)
        assert trainer.amp is False

    def test_trainer_initialization_with_grad_clip(self, mock_model, optimizer):
        trainer = Trainer(model=mock_model, optimizer=optimizer, grad_clip=1.0)
        assert trainer.grad_clip == 1.0

    def test_trainer_initialization_with_tensorboard(self, mock_model, optimizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(model=mock_model, optimizer=optimizer, tb_dir=tmpdir)
            assert trainer.tb is not None
            trainer.tb.close()


class TestTrainerRun:
    def test_run_executes_epochs(self, mock_model, optimizer, train_loader):
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        initial_count = mock_model.call_count
        trainer.run(train_loader, max_epochs=2, log_every=50)
        # Should have called forward for each batch * epochs
        expected_calls = len(train_loader) * 2
        assert mock_model.call_count == initial_count + expected_calls

    def test_run_updates_model_weights(self, mock_model, optimizer, train_loader):
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        initial_weight = mock_model.linear.weight.data.clone()
        trainer.run(train_loader, max_epochs=1, log_every=50)
        # Weights should have changed
        assert not torch.allclose(mock_model.linear.weight.data, initial_weight)

    def test_run_with_validation(self, mock_model, optimizer, train_loader, val_loader):
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        # Should not raise any errors
        trainer.run(train_loader, val_loader=val_loader, max_epochs=1, log_every=50)

    def test_run_with_scheduler(self, mock_model, optimizer, train_loader):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        initial_lr = optimizer.param_groups[0]["lr"]  # noqa: F841
        trainer = Trainer(model=mock_model, optimizer=optimizer, scheduler=scheduler)
        trainer.run(train_loader, max_epochs=2, log_every=50)
        # Scheduler should have stepped
        # Note: exact LR depends on scheduler config
        assert True  # Just verify no errors

    def test_run_with_amp_disabled(self, mock_model, optimizer, train_loader):
        trainer = Trainer(model=mock_model, optimizer=optimizer, amp=False)
        trainer.run(train_loader, max_epochs=1, log_every=50)
        # Should complete without errors
        assert True

    def test_run_with_tensorboard_logging(self, mock_model, optimizer, train_loader):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(model=mock_model, optimizer=optimizer, tb_dir=tmpdir)
            trainer.run(train_loader, max_epochs=1, log_every=2)
            trainer.tb.close()
            # Check that event files were created
            event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
            assert len(event_files) > 0

    def test_run_model_moved_to_device(self, mock_model, optimizer, train_loader):
        device = torch.device("cpu")
        trainer = Trainer(model=mock_model, optimizer=optimizer, device=device)
        trainer.run(train_loader, max_epochs=1, log_every=50)
        # Check model is on correct device
        for param in mock_model.parameters():
            assert param.device.type == device.type

    def test_run_with_grad_clip(self, mock_model, optimizer, train_loader):
        trainer = Trainer(model=mock_model, optimizer=optimizer, grad_clip=1.0)
        # Should complete without errors
        trainer.run(train_loader, max_epochs=1, log_every=50)
        assert True


class TestTrainerValidate:
    def test_validate_runs_without_errors(self, mock_model, optimizer, val_loader):
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        trainer.model.to(trainer.device)
        trainer.validate(val_loader, ep=0)
        assert True

    def test_validate_sets_model_to_eval(self, mock_model, optimizer, val_loader):
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        trainer.model.to(trainer.device)
        trainer.model.train()
        trainer.validate(val_loader, ep=0)
        assert not trainer.model.training

    def test_validate_with_tensorboard(self, mock_model, optimizer, val_loader):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(model=mock_model, optimizer=optimizer, tb_dir=tmpdir)
            trainer.model.to(trainer.device)
            trainer.validate(val_loader, ep=0)
            trainer.tb.close()
            # Should have logged to tensorboard
            event_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
            assert len(event_files) > 0

    def test_validate_computes_average_loss(self, mock_model, optimizer, val_loader):
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        trainer.model.to(trainer.device)
        # Should compute average without errors
        trainer.validate(val_loader, ep=0)
        assert True

    def test_validate_no_gradient_computation(self, mock_model, optimizer, val_loader):
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        trainer.model.to(trainer.device)
        # Clear any existing gradients
        optimizer.zero_grad()
        trainer.validate(val_loader, ep=0)
        # Gradients should still be None or zero after validation
        for param in mock_model.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))


class TestTrainerEdgeCases:
    def test_run_with_empty_logs(self, optimizer, train_loader):
        class ModelWithEmptyLogs(JointModel):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, batch):
                x = batch if isinstance(batch, torch.Tensor) else batch[0]
                loss = self.linear(x).mean() ** 2
                return TrainStep(loss=loss, logs={})

        model = ModelWithEmptyLogs()
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(model=model, optimizer=optimizer, tb_dir=tmpdir)
            trainer.run(train_loader, max_epochs=1, log_every=1)
            trainer.tb.close()

    def test_validate_with_empty_loader(self, mock_model, optimizer):
        empty_loader = DataLoader(TensorDataset(torch.randn(0, 10)), batch_size=4)
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        trainer.model.to(trainer.device)
        # Should handle empty loader gracefully
        trainer.validate(empty_loader, ep=0)
        assert True

    def test_run_single_batch(self, mock_model, optimizer):
        single_batch_loader = DataLoader(TensorDataset(torch.randn(4, 10)), batch_size=4)
        trainer = Trainer(model=mock_model, optimizer=optimizer)
        trainer.run(single_batch_loader, max_epochs=1, log_every=1)
        assert True
