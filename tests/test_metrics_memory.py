"""
Tests for memory usage metrics collection.

This module tests the memory usage measurements for the DistilBERT benchmarking suite.
"""

import time
import pytest
import torch
import psutil
from src.metrics.memory import (
    MemoryMetricCollector,
    measure_peak_memory_usage,
    get_gpu_info,
)


def test_memory_collector_initialization():
    """Test that the memory collector initializes correctly."""
    # Initialize with default parameters
    collector = MemoryMetricCollector()
    assert collector.device == "cpu"
    assert collector.interval_ms == 0.01  # 10ms converted to seconds
    assert isinstance(collector.process, psutil.Process)

    # Initialize with custom parameters
    collector = MemoryMetricCollector(device="cpu", interval_ms=50, track_gpu=False)
    assert collector.device == "cpu"
    assert collector.interval_ms == 0.05  # 50ms converted to seconds
    assert collector.track_gpu is False


def test_memory_collector_reset():
    """Test that the memory collector reset works correctly."""
    collector = MemoryMetricCollector()
    # Add some fake data
    collector.cpu_memory = [10000, 20000, 30000]
    collector.gpu_memory = [50000, 60000]

    # Reset
    collector.reset()

    # Check reset state
    assert collector.cpu_memory == []
    assert collector.gpu_memory == []
    assert collector.running is False
    assert collector.collection_thread is None


def test_memory_collection_start_stop():
    """Test starting and stopping memory collection."""
    collector = MemoryMetricCollector(interval_ms=100)  # Larger interval for stability

    # Start collection
    collector.start_collection()
    assert collector.running is True
    assert collector.collection_thread is not None
    assert collector.collection_thread.is_alive() is True

    # Let it collect a few samples
    time.sleep(0.3)  # Should collect about 3 samples

    # Stop collection
    collector.stop_collection()
    assert collector.running is False

    # Wait for thread to join
    time.sleep(0.2)

    # Check that we collected data
    assert len(collector.cpu_memory) > 0

    # It should have collected at least 2 samples
    assert len(collector.cpu_memory) >= 2


def test_memory_metrics():
    """Test that memory metrics are calculated correctly."""
    collector = MemoryMetricCollector()

    # Mock memory data (in bytes)
    collector.cpu_memory = [
        100 * 1024 * 1024,
        110 * 1024 * 1024,
        120 * 1024 * 1024,
    ]  # 100MB, 110MB, 120MB

    # Get metrics
    metrics = collector.get_metrics()

    # Check CPU metrics were converted to MB
    assert metrics["cpu_memory_mb_mean"] == pytest.approx(110.0)
    assert metrics["cpu_memory_mb_min"] == pytest.approx(100.0)
    assert metrics["cpu_memory_mb_max"] == pytest.approx(120.0)
    assert metrics["cpu_memory_mb_samples"] == 3
    assert metrics["cpu_memory_mb_initial"] == pytest.approx(100.0)
    assert metrics["cpu_memory_mb_final"] == pytest.approx(120.0)

    # Check configuration info
    assert metrics["device"] == "cpu"
    assert metrics["interval_ms"] == 10.0  # default interval


def test_empty_metrics():
    """Test handling of empty metrics."""
    collector = MemoryMetricCollector()
    metrics = collector.get_metrics()

    # Should still have configuration info
    assert "device" in metrics
    assert "interval_ms" in metrics
    assert "gpu_tracking_enabled" in metrics


def test_memory_allocation():
    """Test measuring memory usage with actual memory allocation."""

    # Create a function that allocates memory and sleeps
    def memory_allocator():
        # Allocate some tensors that should be visible to the memory collector
        tensors = []
        for _ in range(10):  # Allocate 10 tensors of 10MB each, total 100MB
            # Each float32 is 4 bytes, so 10M floats = 40MB
            tensor = torch.zeros(10 * 1000 * 1000, dtype=torch.float32)
            tensors.append(tensor)
            time.sleep(0.05)  # Give time for collection

        # Hold the tensors for a bit
        time.sleep(0.2)
        return tensors  # Return to prevent garbage collection during test

    # Measure peak memory usage
    metrics, tensors = measure_peak_memory_usage(
        memory_allocator, device="cpu", sampling_interval_ms=20, track_gpu=False
    )

    # Verify we have memory metrics
    assert "cpu_memory_mb_max" in metrics
    assert "cpu_memory_mb_initial" in metrics
    assert "cpu_memory_mb_final" in metrics

    # Memory should have increased during the test
    assert metrics["cpu_memory_mb_final"] > metrics["cpu_memory_mb_initial"]
    # We allocated about 100MB, so the increase should be substantial
    assert (
        metrics["cpu_memory_mb_max"] - metrics["cpu_memory_mb_initial"] > 50
    )  # At least 50MB increase


def test_get_gpu_info():
    """Test the get_gpu_info function."""
    # This just tests that the function runs without errors
    # GPU info may or may not be available
    gpu_info = get_gpu_info()

    # Should at least tell us if GPUs are available
    assert "available" in gpu_info


if __name__ == "__main__":
    pytest.main(["-v", __file__])
