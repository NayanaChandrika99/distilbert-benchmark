"""
Tests for energy consumption metrics collection.

This module tests the energy consumption measurements for the DistilBERT benchmarking suite.
"""

import time
import pytest
import torch
from src.metrics.energy import (
    EnergyMetricCollector,
    measure_energy_consumption,
    is_energy_measurement_available,
)


def test_energy_collector_initialization():
    """Test that the energy collector initializes correctly."""
    # Initialize with default parameters
    collector = EnergyMetricCollector()
    assert collector.device == "cpu"
    assert collector.measurement_interval_ms == 0.05  # 50ms converted to seconds
    assert collector.is_cuda is False

    # Initialize with custom parameters
    collector = EnergyMetricCollector(
        device="cuda:0", measurement_interval_ms=100, use_rapl=False
    )
    assert collector.device == "cuda:0"
    assert collector.measurement_interval_ms == 0.1  # 100ms converted to seconds
    assert collector.is_cuda is True
    assert collector.use_rapl is False


def test_energy_collector_reset():
    """Test that the energy collector reset works correctly."""
    collector = EnergyMetricCollector()

    # Reset
    collector.reset()

    # Check reset state
    assert collector.cpu_energy_uj is None
    assert collector.gpu_power_samples == []
    assert collector.measurement_time == 0.0


def test_energy_metrics():
    """Test that energy metrics are calculated correctly."""
    collector = EnergyMetricCollector()

    # Mock measurement time
    collector.measurement_time = 2.0  # 2 seconds

    # Mock CPU energy data (microjoules)
    collector.cpu_energy_uj = {
        "package": 2_000_000,  # 2 joules
        "dram": 500_000,  # 0.5 joules
    }

    # Mock GPU power data (milliwatts)
    collector.gpu_power_samples = [5000, 5500, 6000]  # 5W, 5.5W, 6W

    # Get metrics
    metrics = collector.get_metrics()

    # Check CPU energy metrics were converted to joules
    assert metrics["device"] == "cpu"

    # Only check if RAPL is enabled
    if collector.use_rapl and "cpu_energy_j" in metrics:
        assert metrics["cpu_energy_j"] == pytest.approx(2.5)  # 2J + 0.5J = 2.5J
        assert metrics["cpu_package_energy_j"] == pytest.approx(2.0)
        assert metrics["cpu_dram_energy_j"] == pytest.approx(0.5)
        assert metrics["cpu_avg_power_w"] == pytest.approx(1.25)  # 2.5J / 2s = 1.25W


def test_energy_availability():
    """Test the energy measurement availability checker."""
    # This just tests that the function runs without errors
    availability = is_energy_measurement_available()

    # Should have these keys
    assert "rapl_available" in availability
    assert "nvml_available" in availability


def test_measure_energy_consumption():
    """Test measuring energy consumption of a function."""
    # Skip if energy measurement is not available
    availability = is_energy_measurement_available()
    if not availability.get("cpu_energy_supported", False) and not availability.get(
        "gpu_energy_supported", False
    ):
        pytest.skip("No energy measurement capabilities available on this system")

    # Create a function that consumes CPU cycles
    def energy_consumer():
        # Do some CPU-intensive work
        result = 0
        for i in range(10_000_000):
            result += i % 100

        # Create and manipulate some tensors
        tensors = []
        for _ in range(5):
            tensor = torch.randn(1000, 1000)
            tensor = tensor @ tensor.t()  # Matrix multiplication
            tensors.append(tensor)

        # Hold the tensors to prevent garbage collection
        time.sleep(0.1)
        return result, tensors

    # Measure energy consumption
    metrics, result = measure_energy_consumption(
        energy_consumer, device="cpu", sampling_interval_ms=50
    )

    # Verify we have measurement time
    assert metrics["measurement_time_s"] > 0

    # Verify energy metrics if available
    if "cpu_energy_j" in metrics:
        assert metrics["cpu_energy_j"] > 0
        assert metrics["cpu_avg_power_w"] > 0


def test_cuda_device_handling():
    """Test CUDA device handling in the energy collector."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test with "cuda" device
    collector = EnergyMetricCollector(device="cuda")
    assert collector.is_cuda is True

    # Test with specific CUDA device
    collector = EnergyMetricCollector(device="cuda:0")
    assert collector.is_cuda is True
    if hasattr(collector, "device_index"):
        assert collector.device_index == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
