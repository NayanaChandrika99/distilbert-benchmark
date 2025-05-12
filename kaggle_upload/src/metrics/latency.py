"""
Latency and throughput measurement for DistilBERT benchmarking.

This module provides utilities for measuring inference latency and calculating
throughput on both CPU and GPU devices.
"""

import time
import logging
from typing import Dict, Optional, Any

import torch
import numpy as np

logger = logging.getLogger(__name__)


class LatencyMetricCollector:
    """Collector for latency and throughput metrics during model inference."""

    def __init__(self, device: str = "cpu", warmup_runs: int = 5, iterations: int = 20):
        """
        Initialize the latency metric collector.

        Args:
            device: Device being used for inference ("cpu", "cuda", etc.)
            warmup_runs: Number of initial inference runs to discard
            iterations: Number of inference runs to average metrics over
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.iterations = iterations
        self.is_cuda = device.startswith("cuda")
        self.reset()

    def reset(self):
        """Reset all collected metrics."""
        self.cpu_latencies = []
        self.gpu_latencies = []
        self.batch_sizes = []
        self.sequence_lengths = []
        self.has_events = False
        self.last_batch_size = None
        self.last_sequence_length = None

    def start_capture(self, batch_size: int, sequence_length: int):
        """
        Start capturing latency for a new inference run.

        Args:
            batch_size: Batch size for this inference
            sequence_length: Sequence length for this inference
        """
        self.last_batch_size = batch_size
        self.last_sequence_length = sequence_length

        # Create CUDA events for GPU measurement if on CUDA device
        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.has_events = True

        # CPU timing using time.perf_counter() for high precision
        self.cpu_start_time = time.perf_counter()

    def end_capture(self):
        """End capturing latency for the current inference run."""
        # CPU timing
        self.cpu_end_time = time.perf_counter()
        self.cpu_latencies.append(
            (self.cpu_end_time - self.cpu_start_time) * 1000
        )  # ms

        # GPU timing
        if self.is_cuda and self.has_events:
            self.end_event.record()
            torch.cuda.synchronize()
            self.gpu_latencies.append(
                self.start_event.elapsed_time(self.end_event)
            )  # ms

        self.batch_sizes.append(self.last_batch_size)
        self.sequence_lengths.append(self.last_sequence_length)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return latency and throughput metrics.

        Returns:
            Dictionary with calculated metrics
        """
        if not self.cpu_latencies:
            logger.warning("No latency measurements collected")
            return {}

        # Calculate CPU-based metrics
        cpu_latencies = np.array(self.cpu_latencies)
        batch_sizes = np.array(self.batch_sizes)
        sequence_lengths = np.array(self.sequence_lengths)

        # Throughput in samples/second
        throughput = (batch_sizes * 1000) / cpu_latencies

        # Metrics to return
        metrics = {
            # Latency metrics (milliseconds)
            "latency_ms_mean": float(np.mean(cpu_latencies)),
            "latency_ms_median": float(np.median(cpu_latencies)),
            "latency_ms_std": float(np.std(cpu_latencies)),
            "latency_ms_min": float(np.min(cpu_latencies)),
            "latency_ms_max": float(np.max(cpu_latencies)),
            "latency_ms_p90": float(np.percentile(cpu_latencies, 90)),
            "latency_ms_p95": float(np.percentile(cpu_latencies, 95)),
            "latency_ms_p99": float(np.percentile(cpu_latencies, 99)),
            # Throughput metrics (samples/second)
            "throughput_mean": float(np.mean(throughput)),
            "throughput_median": float(np.median(throughput)),
            "throughput_max": float(np.max(throughput)),
            # Batch information
            "batch_size": int(batch_sizes[0])
            if len(np.unique(batch_sizes)) == 1
            else batch_sizes.tolist(),
            "sequence_length": int(sequence_lengths[0])
            if len(np.unique(sequence_lengths)) == 1
            else sequence_lengths.tolist(),
            "num_measurements": len(cpu_latencies),
            # Configuration
            "device": self.device,
            "warmup_runs": self.warmup_runs,
            "iterations": self.iterations,
        }

        # Add GPU-specific metrics if available
        if self.is_cuda and self.gpu_latencies:
            gpu_latencies = np.array(self.gpu_latencies)
            metrics.update(
                {
                    "gpu_latency_ms_mean": float(np.mean(gpu_latencies)),
                    "gpu_latency_ms_median": float(np.median(gpu_latencies)),
                    "gpu_latency_ms_min": float(np.min(gpu_latencies)),
                    "gpu_latency_ms_max": float(np.max(gpu_latencies)),
                    "gpu_latency_ms_p95": float(np.percentile(gpu_latencies, 95)),
                }
            )

        return metrics


def measure_inference_latency(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    warmup_runs: int = 5,
    iterations: int = 20,
    device: str = None,
) -> Dict[str, Any]:
    """
    Measure the latency of model inference.

    Args:
        model: The PyTorch model to measure
        input_ids: Input tensor of token ids
        attention_mask: Optional attention mask tensor
        token_type_ids: Optional token type ids tensor
        warmup_runs: Number of initial inference runs to discard
        iterations: Number of inference runs to average metrics over
        device: Device used for inference (auto-detected if None)

    Returns:
        Dictionary with latency metrics
    """
    if device is None:
        device = next(model.parameters()).device.type
        if device.startswith("cuda") and hasattr(
            next(model.parameters()).device, "index"
        ):
            device = f"{device}:{next(model.parameters()).device.index}"

    # Get batch size and sequence length from input tensors
    batch_size, sequence_length = input_ids.shape

    # Initialize collector
    collector = LatencyMetricCollector(device, warmup_runs, iterations)

    # Prepare inference function with needed arguments
    def inference_func():
        with torch.no_grad():
            if attention_mask is not None and token_type_ids is not None:
                return model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            elif attention_mask is not None:
                return model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                return model(input_ids=input_ids)

    # Warmup runs to ensure GPU is at steady state
    logger.info(f"Performing {warmup_runs} warmup runs")
    for _ in range(warmup_runs):
        inference_func()
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    # Measurement runs
    logger.info(f"Performing {iterations} measurement iterations")
    for _ in range(iterations):
        collector.start_capture(batch_size, sequence_length)
        inference_func()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        collector.end_capture()

    # Calculate metrics
    metrics = collector.get_metrics()

    return metrics
