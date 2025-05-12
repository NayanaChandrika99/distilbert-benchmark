"""
Tests for latency and throughput metrics collection.

This module tests the latency and throughput measurements for the DistilBERT benchmarking suite.
"""

import time
import pytest
import torch
from src.metrics.latency import LatencyMetricCollector, measure_inference_latency


def test_latency_collector_initialization():
    """Test that the latency collector initializes correctly."""
    # Initialize with default parameters
    collector = LatencyMetricCollector()
    assert collector.device == "cpu"
    assert collector.warmup_runs == 5
    assert collector.iterations == 20
    assert collector.is_cuda is False

    # Initialize with custom parameters
    collector = LatencyMetricCollector(device="cuda:0", warmup_runs=10, iterations=30)
    assert collector.device == "cuda:0"
    assert collector.warmup_runs == 10
    assert collector.iterations == 30
    assert collector.is_cuda is True


def test_latency_collector_reset():
    """Test that the latency collector reset works correctly."""
    collector = LatencyMetricCollector()
    # Add some fake data
    collector.cpu_latencies = [10, 20, 30]
    collector.batch_sizes = [1, 1, 1]
    collector.sequence_lengths = [128, 128, 128]

    # Reset
    collector.reset()

    # Check reset state
    assert collector.cpu_latencies == []
    assert collector.gpu_latencies == []
    assert collector.batch_sizes == []
    assert collector.sequence_lengths == []
    assert collector.has_events is False


def test_latency_collector_capture():
    """Test capturing latency measurements."""
    collector = LatencyMetricCollector()

    # Start capture
    batch_size = 4
    sequence_length = 64
    collector.start_capture(batch_size, sequence_length)

    # Ensure batch size and sequence length are stored
    assert collector.last_batch_size == batch_size
    assert collector.last_sequence_length == sequence_length

    # Simulate some work
    time.sleep(0.1)

    # End capture
    collector.end_capture()

    # Check measurements were recorded
    assert len(collector.cpu_latencies) == 1
    assert collector.cpu_latencies[0] > 0  # Should have measured some time
    assert collector.batch_sizes == [batch_size]
    assert collector.sequence_lengths == [sequence_length]


def test_latency_collector_metrics():
    """Test that latency metrics are calculated correctly."""
    collector = LatencyMetricCollector()

    # Mock latency data
    collector.cpu_latencies = [10.0, 15.0, 20.0, 25.0, 30.0]  # ms
    collector.batch_sizes = [4, 4, 4, 4, 4]
    collector.sequence_lengths = [64, 64, 64, 64, 64]

    # Get metrics
    metrics = collector.get_metrics()

    # Check metrics
    assert metrics["latency_ms_mean"] == 20.0
    assert metrics["latency_ms_median"] == 20.0
    assert metrics["latency_ms_min"] == 10.0
    assert metrics["latency_ms_max"] == 30.0

    # Check throughput calculation
    # For each latency value, throughput = (batch_size * 1000) / latency_ms
    # So expected throughputs: 400, 266.67, 200, 160, 133.33
    # Mean of these values should be around 232
    assert metrics["throughput_mean"] == pytest.approx(232.0, rel=1e-2)

    # Check batch info
    assert metrics["batch_size"] == 4
    assert metrics["sequence_length"] == 64
    assert metrics["num_measurements"] == 5


def test_empty_metrics():
    """Test handling of empty metrics."""
    collector = LatencyMetricCollector()
    metrics = collector.get_metrics()
    assert metrics == {}


class DummyModel(torch.nn.Module):
    """Create a dummy model for testing the latency measurement."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(1000, 128)
        self.linear = torch.nn.Linear(128, 2)
        self.sleep_time = 0.01

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Simulate processing time
        time.sleep(self.sleep_time)

        # Actual computation that works with the input shapes
        embedded = self.embedding(input_ids)
        pooled = torch.mean(embedded, dim=1)  # Average over sequence length
        logits = self.linear(pooled)

        return {"logits": logits}


def create_dummy_model():
    """Create a dummy model for testing the measure_inference_latency function."""
    return DummyModel()


def test_measure_inference_latency():
    """Test the measure_inference_latency function."""
    # Create dummy model and inputs
    model = create_dummy_model()
    input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    attention_mask = torch.ones_like(input_ids)

    # Measure latency
    metrics = measure_inference_latency(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        warmup_runs=2,
        iterations=5,
    )

    # Check that metrics are returned
    assert "latency_ms_mean" in metrics
    assert "throughput_mean" in metrics
    assert metrics["batch_size"] == 2
    assert metrics["sequence_length"] == 10
    assert metrics["num_measurements"] == 5
    assert metrics["device"] == "cpu"

    # Latency should be at least our sleep time
    assert metrics["latency_ms_mean"] >= 10  # 0.01s = 10ms


if __name__ == "__main__":
    pytest.main(["-v", __file__])
