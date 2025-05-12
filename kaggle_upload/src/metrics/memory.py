"""
Memory usage measurement for DistilBERT benchmarking.

This module provides utilities for measuring CPU and GPU memory usage
during model inference.
"""

import os
import logging
import threading
import time
from typing import Dict, Any, Callable

import psutil
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pynvml for GPU memory tracking
try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    logger.warning("pynvml not available. GPU memory tracking will be disabled.")
    NVML_AVAILABLE = False


class MemoryMetricCollector:
    """Collector for CPU and GPU memory metrics during model inference."""

    def __init__(
        self, device: str = "cpu", interval_ms: int = 10, track_gpu: bool = True
    ):
        """
        Initialize the memory metric collector.

        Args:
            device: Device being used for inference ("cpu", "cuda", etc.)
            interval_ms: Sampling interval in milliseconds
            track_gpu: Whether to track GPU memory (requires pynvml)
        """
        self.device = device
        self.interval_ms = interval_ms / 1000.0  # Convert to seconds
        self.track_gpu = track_gpu and NVML_AVAILABLE and device.startswith("cuda")
        self.process = psutil.Process(os.getpid())
        self.reset()

        # Initialize NVML if tracking GPU
        if self.track_gpu:
            try:
                pynvml.nvmlInit()
                if ":" in device:
                    self.device_index = int(device.split(":")[-1])
                else:
                    self.device_index = 0
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                logger.info(
                    f"GPU memory tracking enabled for device {self.device_index}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize NVML: {e}")
                self.track_gpu = False

    def __del__(self):
        """Clean up NVML resources when done."""
        if self.track_gpu and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                logger.debug(f"Error shutting down NVML: {e}")
                pass

    def reset(self):
        """Reset all collected metrics."""
        self.cpu_memory = []
        self.gpu_memory = []
        self.running = False
        self.collection_thread = None

    def start_collection(self):
        """Start collecting memory metrics in a background thread."""
        if self.running:
            logger.warning("Memory metrics collection already running")
            return

        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.daemon = True
        self.collection_thread.start()

    def stop_collection(self):
        """Stop collecting memory metrics."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)

    def _collect_metrics(self):
        """Background thread function for collecting memory metrics."""
        while self.running:
            # CPU memory (process RSS)
            try:
                mem_info = self.process.memory_info()
                self.cpu_memory.append(mem_info.rss)
            except Exception as e:
                logger.error(f"Error collecting CPU memory: {e}")

            # GPU memory (if enabled)
            if self.track_gpu:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_memory.append(mem_info.used)
                except Exception as e:
                    logger.error(f"Error collecting GPU memory: {e}")

            # Wait for next collection interval
            time.sleep(self.interval_ms)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return memory usage metrics.

        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}

        # Process CPU memory metrics
        if self.cpu_memory:
            # Convert bytes to MB for readability
            cpu_memory_mb = np.array(self.cpu_memory) / (1024 * 1024)

            metrics.update(
                {
                    # CPU memory metrics (MB)
                    "cpu_memory_mb_mean": float(np.mean(cpu_memory_mb)),
                    "cpu_memory_mb_max": float(np.max(cpu_memory_mb)),
                    "cpu_memory_mb_min": float(np.min(cpu_memory_mb)),
                    "cpu_memory_mb_samples": len(cpu_memory_mb),
                    "cpu_memory_mb_initial": float(cpu_memory_mb[0])
                    if len(cpu_memory_mb) > 0
                    else 0,
                    "cpu_memory_mb_final": float(cpu_memory_mb[-1])
                    if len(cpu_memory_mb) > 0
                    else 0,
                }
            )

        # Process GPU memory metrics
        if self.track_gpu and self.gpu_memory:
            # Convert bytes to MB for readability
            gpu_memory_mb = np.array(self.gpu_memory) / (1024 * 1024)

            metrics.update(
                {
                    # GPU memory metrics (MB)
                    "gpu_memory_mb_mean": float(np.mean(gpu_memory_mb)),
                    "gpu_memory_mb_max": float(np.max(gpu_memory_mb)),
                    "gpu_memory_mb_min": float(np.min(gpu_memory_mb)),
                    "gpu_memory_mb_samples": len(gpu_memory_mb),
                    "gpu_memory_mb_initial": float(gpu_memory_mb[0])
                    if len(gpu_memory_mb) > 0
                    else 0,
                    "gpu_memory_mb_final": float(gpu_memory_mb[-1])
                    if len(gpu_memory_mb) > 0
                    else 0,
                    "gpu_device_index": self.device_index
                    if hasattr(self, "device_index")
                    else 0,
                }
            )

        # Add configuration info
        metrics.update(
            {
                "device": self.device,
                "interval_ms": self.interval_ms
                * 1000,  # Convert back to ms for reporting
                "gpu_tracking_enabled": self.track_gpu,
            }
        )

        return metrics


def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPUs.

    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {"available": False}

    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": [],
    }

    # Basic PyTorch info
    for i in range(info["device_count"]):
        device_info = {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
        }
        info["devices"].append(device_info)

    # Add NVML info if available
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()

            for i in range(info["device_count"]):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Update existing device info
                info["devices"][i].update(
                    {
                        "total_memory_mb": mem_info.total / (1024 * 1024),
                        "free_memory_mb": mem_info.free / (1024 * 1024),
                        "used_memory_mb": mem_info.used / (1024 * 1024),
                    }
                )

                # Add utilization metrics if available
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    info["devices"][i]["gpu_utilization"] = util.gpu
                    info["devices"][i]["memory_utilization"] = util.memory
                except pynvml.NVMLError as e:
                    logger.debug(f"Error getting GPU utilization: {e}")
                    pass

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Error getting NVML GPU info: {e}")

    return info


def measure_peak_memory_usage(
    func: Callable,
    device: str = "cpu",
    sampling_interval_ms: int = 10,
    track_gpu: bool = True,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Measure peak memory usage during execution of a function.

    Args:
        func: Function to execute and measure
        device: Device being used ("cpu", "cuda", etc.)
        sampling_interval_ms: Memory sampling interval in milliseconds
        track_gpu: Whether to track GPU memory
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Dictionary with memory usage metrics
    """
    # Initialize collector
    collector = MemoryMetricCollector(device, sampling_interval_ms, track_gpu)

    # Measure memory usage
    collector.start_collection()
    result = func(*args, **kwargs)
    collector.stop_collection()

    # Get metrics
    metrics = collector.get_metrics()

    return metrics, result
