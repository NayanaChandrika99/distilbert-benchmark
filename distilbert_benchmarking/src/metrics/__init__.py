"""
Metrics collection package for DistilBERT benchmarking.

This package provides utilities for measuring various performance metrics
during model inference, including latency, memory usage, and energy consumption.
"""

from src.metrics.latency import LatencyMetricCollector, measure_inference_latency
from src.metrics.memory import (
    MemoryMetricCollector,
    measure_peak_memory_usage,
    get_gpu_info,
)
from src.metrics.energy import (
    EnergyMetricCollector,
    measure_energy_consumption,
    is_energy_measurement_available,
)

__all__ = [
    # Latency metrics
    "LatencyMetricCollector",
    "measure_inference_latency",
    # Memory metrics
    "MemoryMetricCollector",
    "measure_peak_memory_usage",
    "get_gpu_info",
    # Energy metrics
    "EnergyMetricCollector",
    "measure_energy_consumption",
    "is_energy_measurement_available",
]
