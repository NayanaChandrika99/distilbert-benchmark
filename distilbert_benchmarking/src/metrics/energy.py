"""
Energy consumption measurement for DistilBERT benchmarking.

This module provides utilities for measuring energy consumption during model inference
using the pyRAPL library, which accesses Intel RAPL (Running Average Power Limit)
counters and NVML for GPU power measurements.
"""

import os
import logging
import time
from typing import Dict, Tuple, Any, Callable

import numpy as np
import torch
import psutil

logger = logging.getLogger(__name__)

# Try to import pyRAPL for CPU energy monitoring
try:
    import pyRAPL

    PYRAPL_AVAILABLE = True
except ImportError:
    logger.warning("pyRAPL not available. CPU energy tracking will be disabled.")
    PYRAPL_AVAILABLE = False

# Try to import pynvml for GPU energy tracking
try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    logger.warning("pynvml not available. GPU energy tracking will be disabled.")
    NVML_AVAILABLE = False


class EnergyMetricCollector:
    """Collector for energy consumption metrics during model inference."""

    def __init__(
        self,
        device: str = "cpu",
        measurement_interval_ms: int = 50,
        use_rapl: bool = True,
        use_nvml: bool = True,
    ):
        """
        Initialize the energy metric collector.

        Args:
            device: Device being used for inference ("cpu", "cuda", etc.)
            measurement_interval_ms: Interval for power measurements in milliseconds (NVML only)
            use_rapl: Whether to use RAPL for CPU energy monitoring
            use_nvml: Whether to use NVML for GPU energy monitoring
        """
        self.device = device
        self.measurement_interval_ms = (
            measurement_interval_ms / 1000.0
        )  # Convert to seconds
        self.is_cuda = device.startswith("cuda")

        # Configure CPU energy monitoring (RAPL)
        self.use_rapl = use_rapl and PYRAPL_AVAILABLE
        if self.use_rapl:
            try:
                pyRAPL.setup()
                logger.info("pyRAPL initialized for CPU energy monitoring")
            except Exception as e:
                logger.error(f"Failed to initialize pyRAPL: {e}")
                self.use_rapl = False

        # Configure GPU energy monitoring (NVML)
        self.use_nvml = use_nvml and NVML_AVAILABLE and self.is_cuda
        if self.use_nvml:
            try:
                pynvml.nvmlInit()
                if ":" in device:
                    self.device_index = int(device.split(":")[-1])
                else:
                    self.device_index = 0
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

                # Check if power measurement is supported
                try:
                    pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    self.gpu_power_supported = True
                    logger.info(
                        f"GPU power measurement enabled for device {self.device_index}"
                    )
                except pynvml.NVMLError as e:
                    self.gpu_power_supported = False
                    logger.warning(
                        f"GPU power measurement not supported on device {self.device_index}: {e}"
                    )
            except Exception as e:
                logger.error(f"Failed to initialize NVML: {e}")
                self.use_nvml = False

        self.reset()

    def __del__(self):
        """Clean up resources when done."""
        if self.use_nvml and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                logger.debug(f"Error shutting down NVML: {e}")
                pass

    def reset(self):
        """Reset all collected metrics."""
        self.cpu_energy_uj = None  # Microjoules
        self.gpu_power_samples = []  # Milliwatts
        self.measurement_time = 0.0  # Seconds
        self.rapl_meter = None if self.use_rapl else None

    def start_measurement(self):
        """Start energy measurement."""
        # Start CPU energy measurement (RAPL)
        if self.use_rapl:
            self.rapl_meter = pyRAPL.Measurement("distilbert-benchmark")
            self.rapl_meter.begin()

        # Prepare for GPU power sampling
        self.gpu_power_samples = []
        self.start_time = time.time()

    def sample_gpu_power(self):
        """Take a GPU power sample if enabled."""
        if self.use_nvml and self.gpu_power_supported:
            try:
                # Power in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                self.gpu_power_samples.append(power_mw)
            except Exception as e:
                logger.error(f"Error sampling GPU power: {e}")

    def end_measurement(self):
        """End energy measurement and collect results."""
        self.end_time = time.time()
        self.measurement_time = self.end_time - self.start_time

        # End CPU energy measurement (RAPL)
        if self.use_rapl and self.rapl_meter:
            self.rapl_meter.end()
            if hasattr(self.rapl_meter, "result"):
                self.cpu_energy_uj = {
                    "package": sum(self.rapl_meter.result.pkg),
                    "dram": sum(self.rapl_meter.result.dram)
                    if hasattr(self.rapl_meter.result, "dram")
                    else 0,
                }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return energy consumption metrics.

        Returns:
            Dictionary with calculated metrics
        """
        metrics = {
            "device": self.device,
            "measurement_time_s": self.measurement_time,
            "energy_tracking_enabled": {
                "cpu_rapl": self.use_rapl,
                "gpu_nvml": self.use_nvml and self.gpu_power_supported
                if hasattr(self, "gpu_power_supported")
                else False,
            },
        }

        # Add CPU energy metrics if available
        if self.use_rapl and self.cpu_energy_uj:
            # Convert microjoules to joules for readability
            pkg_energy_j = self.cpu_energy_uj["package"] / 1_000_000
            dram_energy_j = self.cpu_energy_uj["dram"] / 1_000_000
            total_energy_j = pkg_energy_j + dram_energy_j

            metrics.update(
                {
                    "cpu_energy_j": float(total_energy_j),
                    "cpu_package_energy_j": float(pkg_energy_j),
                    "cpu_dram_energy_j": float(dram_energy_j),
                    "cpu_avg_power_w": float(total_energy_j / self.measurement_time)
                    if self.measurement_time > 0
                    else 0,
                }
            )

        # Add GPU energy metrics if available
        if self.use_nvml and self.gpu_power_samples:
            # Convert milliwatts to watts for readability
            gpu_power_w = np.array(self.gpu_power_samples) / 1000.0

            # Calculate energy from power samples (power * time)
            sample_count = len(gpu_power_w)
            if sample_count > 0:
                # Average power in watts
                avg_power_w = float(np.mean(gpu_power_w))

                # Energy in joules (power * time)
                energy_j = avg_power_w * self.measurement_time

                metrics.update(
                    {
                        "gpu_energy_j": float(energy_j),
                        "gpu_avg_power_w": avg_power_w,
                        "gpu_min_power_w": float(np.min(gpu_power_w)),
                        "gpu_max_power_w": float(np.max(gpu_power_w)),
                        "gpu_power_samples": sample_count,
                    }
                )

        # Calculate total energy if both CPU and GPU measurements are available
        if "cpu_energy_j" in metrics and "gpu_energy_j" in metrics:
            metrics["total_energy_j"] = (
                metrics["cpu_energy_j"] + metrics["gpu_energy_j"]
            )

        return metrics


def measure_energy_consumption(
    func: Callable,
    device: str = "cpu",
    sampling_interval_ms: int = 50,
    use_rapl: bool = True,
    use_nvml: bool = True,
    *args,
    **kwargs,
) -> Tuple[Dict[str, Any], Any]:
    """
    Measure energy consumption during execution of a function.

    Args:
        func: Function to execute and measure
        device: Device being used ("cpu", "cuda", etc.)
        sampling_interval_ms: Interval for power sampling in milliseconds
        use_rapl: Whether to use RAPL for CPU energy monitoring
        use_nvml: Whether to use NVML for GPU energy monitoring
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Tuple of (energy metrics dictionary, function result)
    """
    # Initialize collector
    collector = EnergyMetricCollector(device, sampling_interval_ms, use_rapl, use_nvml)

    # Start measurement
    collector.start_measurement()

    # Run the function to measure
    result = func(*args, **kwargs)

    # If GPU sampling is enabled, take a final sample
    collector.sample_gpu_power()

    # End measurement
    collector.end_measurement()

    # Get metrics
    metrics = collector.get_metrics()

    return metrics, result


def is_energy_measurement_available() -> Dict[str, bool]:
    """
    Check if energy measurement capabilities are available on this system.

    Returns:
        Dictionary indicating which energy measurement methods are available
    """
    status = {
        "rapl_available": PYRAPL_AVAILABLE,
        "nvml_available": NVML_AVAILABLE,
        "cpu_energy_supported": False,
        "gpu_energy_supported": False,
    }

    # Check RAPL functionality
    if PYRAPL_AVAILABLE:
        try:
            pyRAPL.setup()
            with pyRAPL.Measurement("test"):
                time.sleep(0.1)
            status["cpu_energy_supported"] = True
        except Exception as e:
            logger.warning(f"pyRAPL available but not functional: {e}")

    # Check NVML power measurement functionality
    if NVML_AVAILABLE and torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetPowerUsage(handle)
            status["gpu_energy_supported"] = True
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"NVML available but power measurement not supported: {e}")

    return status
