"""
Memory tracking tools for DistilBERT benchmarks.

Keeps an eye on CPU and GPU memory usage during model runs.
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

# Try to import NVIDIA tools for GPU memory tracking
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    logger.warning("Can't find pynvml package. Won't be able to track GPU memory.")
    NVML_AVAILABLE = False


class MemoryMetricCollector:
    """Tracks CPU and GPU memory usage during model runs."""

    def __init__(
        self, device: str = "cpu", interval_ms: int = 10, track_gpu: bool = True
    ):
        """
        Set up memory tracking.

        Args:
            device: Which device we're measuring ("cpu", "cuda", etc.)
            interval_ms: How often to check memory (in milliseconds)
            track_gpu: Whether to track GPU memory (needs pynvml)
        """
        self.device = device
        self.interval_ms = interval_ms / 1000.0  # Convert to seconds
        self.track_gpu = track_gpu and NVML_AVAILABLE and device.startswith("cuda")
        self.process = psutil.Process(os.getpid())
        self.reset()

        # Set up NVIDIA tools if we're tracking GPU
        if self.track_gpu:
            try:
                pynvml.nvmlInit()
                if ":" in device:
                    self.device_index = int(device.split(":")[-1])
                else:
                    self.device_index = 0
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                logger.info(
                    f"GPU memory tracking ready for device {self.device_index}"
                )
            except Exception as e:
                logger.error(f"Couldn't initialize NVML: {e}")
                self.track_gpu = False

    def __del__(self):
        """Clean up NVIDIA tools when we're done."""
        if self.track_gpu and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                logger.debug(f"Error shutting down NVML: {e}")
                pass

    def reset(self):
        """Clear all collected data."""
        self.cpu_memory = []
        self.gpu_memory = []
        self.running = False
        self.collection_thread = None

    def start_collection(self):
        """Start measuring memory in a background thread."""
        if self.running:
            logger.warning("Memory tracking already running!")
            return

        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.daemon = True
        self.collection_thread.start()

    def stop_collection(self):
        """Stop measuring memory."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)

    def _collect_metrics(self):
        """Background thread that actually collects the memory data."""
        while self.running:
            # Check CPU memory (process RSS)
            try:
                mem_info = self.process.memory_info()
                self.cpu_memory.append(mem_info.rss)
            except Exception as e:
                logger.error(f"Error getting CPU memory: {e}")

            # Check GPU memory if enabled
            if self.track_gpu:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_memory.append(mem_info.used)
                except Exception as e:
                    logger.error(f"Error getting GPU memory: {e}")

            # Wait until next check
            time.sleep(self.interval_ms)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate memory stats from collected data.

        Returns:
            Dict with calculated memory metrics
        """
        metrics = {}

        # Process CPU memory data
        if self.cpu_memory:
            # Convert bytes to MB for easier reading
            cpu_memory_mb = np.array(self.cpu_memory) / (1024 * 1024)

            metrics.update(
                {
                    # CPU memory stats in MB
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

        # Process GPU memory data
        if self.track_gpu and self.gpu_memory:
            # Convert bytes to MB for easier reading
            gpu_memory_mb = np.array(self.gpu_memory) / (1024 * 1024)

            metrics.update(
                {
                    # GPU memory stats in MB
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

        # Add setup info
        metrics.update(
            {
                "device": self.device,
                "interval_ms": self.interval_ms * 1000,  # Back to ms for reporting
                "gpu_tracking_enabled": self.track_gpu,
            }
        )

        return metrics


def get_gpu_info() -> Dict[str, Any]:
    """
    Get info about available GPUs.

    Returns:
        Dict with GPU specs and status
    """
    if not torch.cuda.is_available():
        return {"available": False}

    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": [],
    }

    # Get basic PyTorch info
    for i in range(info["device_count"]):
        device_info = {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
        }
        info["devices"].append(device_info)

    # Add NVIDIA-specific info if available
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()

            for i in range(info["device_count"]):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Add to existing device info
                info["devices"][i].update(
                    {
                        "total_memory_mb": mem_info.total / (1024 * 1024),
                        "free_memory_mb": mem_info.free / (1024 * 1024),
                        "used_memory_mb": mem_info.used / (1024 * 1024),
                    }
                )

                # Try to get utilization too
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    info["devices"][i]["gpu_utilization"] = util.gpu
                    info["devices"][i]["memory_utilization"] = util.memory
                except pynvml.NVMLError as e:
                    logger.debug(f"Can't get GPU utilization: {e}")
                    pass

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Problem getting NVIDIA GPU info: {e}")

    return info


def measure_peak_memory_usage(
    func: Callable,
    device: str = "cpu",
    sampling_interval_ms: int = 10,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a function and measure its peak memory usage.

    Args:
        func: Function to measure
        device: Device to measure ("cpu", "cuda:0", etc.)
        sampling_interval_ms: How often to check memory
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Dict with memory metrics and function result
    """
    # Set up the memory collector
    collector = MemoryMetricCollector(
        device=device,
        interval_ms=sampling_interval_ms,
        track_gpu=(device != "cpu"),
    )

    # Start collecting
    collector.start_collection()

    # Run the function
    try:
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
    finally:
        # Make sure we stop collecting even if there's an error
        collector.stop_collection()

    # Get memory metrics
    metrics = collector.get_metrics()

    # Add execution time and return value
    metrics.update(
        {
            "execution_time": execution_time,
            "result": result,
        }
    )

    return metrics
