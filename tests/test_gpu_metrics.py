"""
Tests for GPU-specific metrics collection.

This module tests the GPU metrics collection for latency, memory, and energy
consumption for the DistilBERT benchmarking suite.
"""

import os
import time
import pytest
import torch
from unittest.mock import patch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available. GPU tests require a CUDA-capable device.",
)

# For direct imports from the source directory
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class DummyGPUModel(torch.nn.Module):
    """Create a dummy model for testing GPU metrics."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.embedding = torch.nn.Embedding(1000, 128).to(device)
        self.linear = torch.nn.Linear(128, 2).to(device)
        self.sleep_time = 0.01

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Simulate processing time
        time.sleep(self.sleep_time)

        # Actual computation that works with the input shapes
        embedded = self.embedding(input_ids)
        pooled = torch.mean(embedded, dim=1)  # Average over sequence length
        logits = self.linear(pooled)

        return {"logits": logits}


@pytest.fixture
def dummy_gpu_model():
    """Create a dummy GPU model for testing."""
    return DummyGPUModel()


@pytest.fixture
def dummy_gpu_inputs():
    """Create dummy inputs for the GPU model."""
    device = torch.device("cuda")
    input_ids = torch.randint(
        0, 1000, (2, 10), device=device
    )  # batch_size=2, seq_len=10
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# GPU Latency Tests
def test_gpu_latency_collector_initialization():
    """Test that the latency collector initializes correctly with CUDA device."""
    from src.metrics.latency import LatencyMetricCollector

    # Initialize with CUDA device
    collector = LatencyMetricCollector(device="cuda")
    assert collector.device == "cuda"
    assert collector.is_cuda is True


def test_gpu_latency_collector_events():
    """Test that CUDA events are created and used for GPU timing."""
    from src.metrics.latency import LatencyMetricCollector

    collector = LatencyMetricCollector(device="cuda")

    # Start capture with CUDA events
    collector.start_capture(batch_size=2, sequence_length=10)

    # Verify CUDA events were created
    assert collector.has_events is True
    assert hasattr(collector, "start_event")
    assert hasattr(collector, "end_event")

    # Simulate work
    time.sleep(0.01)

    # End capture
    collector.end_capture()

    # Verify GPU latencies were recorded
    assert len(collector.gpu_latencies) == 1
    assert collector.gpu_latencies[0] > 0


def test_gpu_latency_metrics(dummy_gpu_model, dummy_gpu_inputs):
    """Test measuring GPU latency metrics with a real model."""
    from src.metrics.latency import measure_inference_latency

    # Extract inputs
    input_ids = dummy_gpu_inputs["input_ids"]
    attention_mask = dummy_gpu_inputs["attention_mask"]

    # Measure latency
    metrics = measure_inference_latency(
        model=dummy_gpu_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device="cuda",
        warmup_runs=2,
        iterations=5,
    )

    # Check that GPU-specific metrics are included
    assert "gpu_latency_ms_mean" in metrics
    assert metrics["device"] == "cuda"
    assert metrics["gpu_latency_ms_mean"] > 0


# GPU Memory Tests
def test_gpu_memory_collector_initialization():
    """Test that the memory collector initializes correctly with CUDA device."""
    from src.metrics.memory import MemoryMetricCollector

    # Skip if pynvml is not available
    try:
        pass
    except ImportError:
        pytest.skip("pynvml not available")

    # Initialize with CUDA device and GPU tracking
    collector = MemoryMetricCollector(device="cuda", track_gpu=True)
    assert collector.device == "cuda"
    assert collector.track_gpu is True


def test_gpu_memory_collection(dummy_gpu_model, dummy_gpu_inputs):
    """Test measuring GPU memory usage with a real model."""
    from src.metrics.memory import measure_peak_memory_usage

    # Skip if pynvml is not available
    try:
        pass
    except ImportError:
        pytest.skip("pynvml not available")

    # Extract inputs
    input_ids = dummy_gpu_inputs["input_ids"]
    attention_mask = dummy_gpu_inputs["attention_mask"]

    # Define inference function
    def inference_func():
        outputs = []
        # Multiple iterations to ensure memory is allocated
        for _ in range(5):
            with torch.no_grad():
                output = dummy_gpu_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                outputs.append(output)
        return outputs

    # Measure memory usage
    metrics, _ = measure_peak_memory_usage(
        func=inference_func, device="cuda", track_gpu=True
    )

    # Check that GPU-specific metrics are included
    assert "gpu_memory_mb_max" in metrics
    assert metrics["device"] == "cuda"
    assert metrics["gpu_memory_mb_max"] > 0


def test_gpu_memory_fallback():
    """Test that memory collection falls back gracefully when pynvml is not available."""
    from src.metrics.memory import measure_peak_memory_usage

    # Mock pynvml as unavailable
    with patch.dict("sys.modules", {"pynvml": None}):
        # Simulate NVML_AVAILABLE = False
        with patch("src.metrics.memory.NVML_AVAILABLE", False):
            # Define simple function
            def simple_func():
                return torch.ones((100, 100), device="cuda")

            # Measure memory usage
            metrics, _ = measure_peak_memory_usage(
                func=simple_func,
                device="cuda",
                track_gpu=True,  # Request GPU tracking even though it's not available
            )

            # Check that only CPU metrics are included
            assert "cpu_memory_mb_max" in metrics
            assert "gpu_memory_mb_max" not in metrics
            assert metrics["gpu_tracking_enabled"] is False


# GPU Energy Tests
def test_gpu_energy_collector_initialization():
    """Test that the energy collector initializes correctly with CUDA device."""
    from src.metrics.energy import EnergyMetricCollector

    # Skip if pynvml is not available
    try:
        pass
    except ImportError:
        pytest.skip("pynvml not available")

    # Initialize with CUDA device and NVML
    collector = EnergyMetricCollector(device="cuda", use_nvml=True)
    assert collector.device == "cuda"
    assert collector.is_cuda is True
    assert collector.use_nvml is True


def test_gpu_energy_collection(dummy_gpu_model, dummy_gpu_inputs):
    """Test measuring GPU energy usage with a real model."""
    from src.metrics.energy import measure_energy_consumption

    # Skip if pynvml is not available
    try:
        import pynvml
    except ImportError:
        pytest.skip("pynvml not available")

    # Skip if the GPU doesn't support power measurement
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetPowerUsage(handle)
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        pytest.skip(f"GPU power measurement not supported on this device: {e}")

    # Extract inputs
    input_ids = dummy_gpu_inputs["input_ids"]
    attention_mask = dummy_gpu_inputs["attention_mask"]

    # Define inference function
    def inference_func():
        # Run multiple iterations to get meaningful measurements
        for _ in range(10):
            with torch.no_grad():
                output = dummy_gpu_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        return output

    # Measure energy consumption
    metrics, _ = measure_energy_consumption(
        func=inference_func, device="cuda", use_nvml=True
    )

    # Check that GPU-specific metrics are included
    assert "gpu_avg_power_w" in metrics
    assert metrics["device"] == "cuda"
    assert metrics["energy_tracking_enabled"]["gpu_nvml"] is True
    assert metrics["gpu_avg_power_w"] > 0


def test_gpu_energy_fallback():
    """Test that energy collection falls back gracefully when NVML is not available."""
    from src.metrics.energy import measure_energy_consumption

    # Mock pynvml as unavailable
    with patch.dict("sys.modules", {"pynvml": None}):
        # Simulate NVML_AVAILABLE = False
        with patch("src.metrics.energy.NVML_AVAILABLE", False):
            # Define simple function
            def simple_func():
                return torch.ones((100, 100), device="cuda")

            # Measure energy usage
            metrics, _ = measure_energy_consumption(
                func=simple_func,
                device="cuda",
                use_nvml=True,  # Request NVML even though it's not available
            )

            # Check fallback behavior
            assert metrics["energy_tracking_enabled"]["gpu_nvml"] is False
            assert "gpu_avg_power_w" not in metrics


def test_is_energy_measurement_available():
    """Test the function that checks if energy measurement is available."""
    from src.metrics.energy import is_energy_measurement_available

    # Get availability info
    availability = is_energy_measurement_available()

    # Should return a dictionary with at least these keys
    assert "cpu_energy_supported" in availability
    assert "gpu_energy_supported" in availability

    # GPU energy should be supported if NVML is available and we have CUDA
    try:
        pass

        # Still might not be supported on this specific GPU
        # so we don't assert the value, just that it exists
    except ImportError:
        assert availability["gpu_energy_supported"] is False


# Runner Integration Tests
def test_runner_gpu_detection():
    """Test that the runner correctly detects available GPU devices."""
    from src.runner import setup_arg_parser, parse_args_to_config

    # Set up parser and parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args(["--device", "cuda"])

    # Convert to config
    config = parse_args_to_config(args)

    # Verify that device is set to CUDA
    assert config.device == "cuda"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
