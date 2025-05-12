"""
Energy usage tracking for DistilBERT benchmarks.

Measures how much power the model consumes during inference - helps
figure out if the model is efficient and environmentally friendly.
Uses Intel RAPL for CPU power and NVML for GPU power measurements.
"""

import os
import logging
import time
from typing import Dict, Tuple, Any, Callable

import numpy as np
import torch
import psutil

logger = logging.getLogger(__name__)

# Try loading pyRAPL for CPU power monitoring
try:
    import pyRAPL
    PYRAPL_AVAILABLE = True
except ImportError:
    logger.warning("Can't find pyRAPL package - won't be able to measure CPU energy")
    PYRAPL_AVAILABLE = False

# Try loading pynvml for GPU power tracking
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    logger.warning("Can't find pynvml package - won't be able to measure GPU energy")
    NVML_AVAILABLE = False


class EnergyMetricCollector:
    """Tracks how much energy the model uses during runs."""

    def __init__(
        self,
        device: str = "cpu",
        measurement_interval_ms: int = 50,
        use_rapl: bool = True,
        use_nvml: bool = True,
    ):
        """
        Set up power tracking.

        Args:
            device: Which device we're measuring ("cpu", "cuda", etc.)
            measurement_interval_ms: How often to check power (in ms, for GPU only)
            use_rapl: Whether to use Intel RAPL for CPU power
            use_nvml: Whether to use NVIDIA tools for GPU power
        """
        self.device = device
        self.measurement_interval_ms = measurement_interval_ms / 1000.0  # To seconds
        self.is_cuda = device.startswith("cuda")

        # Set up CPU power monitoring
        self.use_rapl = use_rapl and PYRAPL_AVAILABLE
        if self.use_rapl:
            try:
                pyRAPL.setup()
                logger.info("pyRAPL ready for CPU power monitoring")
            except Exception as e:
                logger.error(f"Couldn't initialize pyRAPL: {e}")
                self.use_rapl = False

        # Set up GPU power monitoring
        self.use_nvml = use_nvml and NVML_AVAILABLE and self.is_cuda
        if self.use_nvml:
            try:
                pynvml.nvmlInit()
                # Figure out which GPU we're using
                if ":" in device:
                    self.device_index = int(device.split(":")[-1])
                else:
                    self.device_index = 0
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

                # Not all GPUs support power measurement (especially older/consumer ones)
                try:
                    pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    self.gpu_power_supported = True
                    logger.info(f"GPU power tracking working on device {self.device_index}")
                except pynvml.NVMLError as e:
                    self.gpu_power_supported = False
                    logger.warning(f"This GPU doesn't support power measurement: {e}")
            except Exception as e:
                logger.error(f"Problem setting up NVML: {e}")
                self.use_nvml = False

        self.reset()

    def __del__(self):
        """Clean up when done."""
        if self.use_nvml and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                # Not a big deal if this fails
                pass

    def reset(self):
        """Clear collected data."""
        self.cpu_energy_uj = None  # Microjoules
        self.gpu_power_samples = []  # Milliwatts
        self.measurement_time = 0.0  # Seconds
        self.rapl_meter = None if self.use_rapl else None

    def start_measurement(self):
        """Start measuring energy usage."""
        # Start CPU energy tracking
        if self.use_rapl:
            self.rapl_meter = pyRAPL.Measurement("distilbert-benchmark")
            self.rapl_meter.begin()

        # Get ready to sample GPU power
        self.gpu_power_samples = []
        self.start_time = time.time()

    def sample_gpu_power(self):
        """Take a GPU power reading if we can."""
        if self.use_nvml and self.gpu_power_supported:
            try:
                # Power in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                self.gpu_power_samples.append(power_mw)
            except Exception as e:
                logger.error(f"Couldn't read GPU power: {e}")

    def end_measurement(self):
        """Stop measuring and gather results."""
        self.end_time = time.time()
        self.measurement_time = self.end_time - self.start_time

        # Finish CPU energy measurement
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
        Calculate energy stats from collected data.

        Returns:
            Dict with energy metrics
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

        # Add CPU energy stats if we have them
        if self.use_rapl and self.cpu_energy_uj:
            # Convert to joules because nobody thinks in microjoules
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

        # Add GPU energy stats if we have them
        if self.use_nvml and self.gpu_power_samples:
            # Convert milliwatts to watts
            gpu_power_w = np.array(self.gpu_power_samples) / 1000.0

            # Calculate energy (power × time)
            sample_count = len(gpu_power_w)
            if sample_count > 0:
                # Average power in watts
                avg_power_w = np.mean(gpu_power_w)
                
                # Total energy in joules (power * time)
                # This is approximate - better than nothing but not perfect
                # More samples = more accurate
                energy_j = avg_power_w * self.measurement_time

                metrics.update(
                    {
                        "gpu_energy_j": float(energy_j),
                        "gpu_avg_power_w": float(avg_power_w),
                        "gpu_power_samples": sample_count,
                        "gpu_power_min_w": float(np.min(gpu_power_w)),
                        "gpu_power_max_w": float(np.max(gpu_power_w)),
                    }
                )
                
                # Add std dev if we have enough samples
                if sample_count > 1:
                    metrics["gpu_power_std_w"] = float(np.std(gpu_power_w))

        # Add total energy if we have both CPU and GPU
        if "cpu_energy_j" in metrics and "gpu_energy_j" in metrics:
            metrics["total_energy_j"] = metrics["cpu_energy_j"] + metrics["gpu_energy_j"]
            metrics["total_avg_power_w"] = metrics["cpu_avg_power_w"] + metrics["gpu_avg_power_w"]
            
        # Add power efficiency metrics if available
        # This gives us energy per sample, which is useful for comparison
        if hasattr(self, "samples_processed") and self.samples_processed > 0:
            if "total_energy_j" in metrics:
                metrics["energy_per_sample_j"] = metrics["total_energy_j"] / self.samples_processed
            elif "cpu_energy_j" in metrics:
                metrics["energy_per_sample_j"] = metrics["cpu_energy_j"] / self.samples_processed
            elif "gpu_energy_j" in metrics:
                metrics["energy_per_sample_j"] = metrics["gpu_energy_j"] / self.samples_processed

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
    Run a function and measure how much energy it uses.
    
    This wrapper is great for benchmarking model efficiency.

    Args:
        func: Function to run
        device: Device it's running on ("cpu", "cuda:0", etc)
        sampling_interval_ms: How often to sample GPU power
        use_rapl: Whether to track CPU power
        use_nvml: Whether to track GPU power
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (energy metrics, function result)
    """
    # Set up energy monitoring
    collector = EnergyMetricCollector(
        device=device,
        measurement_interval_ms=sampling_interval_ms,
        use_rapl=use_rapl,
        use_nvml=use_nvml,
    )

    collector.start_measurement()

    # Start a background thread to sample GPU power if needed
    if collector.use_nvml and hasattr(collector, "gpu_power_supported") and collector.gpu_power_supported:
        import threading
        stop_sampling = False

        def power_sampling_thread():
            while not stop_sampling:
                collector.sample_gpu_power()
                time.sleep(collector.measurement_interval_ms)

        sampler_thread = threading.Thread(target=power_sampling_thread)
        sampler_thread.daemon = True
        sampler_thread.start()

    # Run the function
    try:
        result = func(*args, **kwargs)
        # If the result contains a sample count, store it for per-sample metrics
        if isinstance(result, dict) and "samples_processed" in result:
            collector.samples_processed = result["samples_processed"]
    finally:
        # Make sure we stop sampling and measuring
        if collector.use_nvml and hasattr(collector, "gpu_power_supported") and collector.gpu_power_supported:
            stop_sampling = True
            sampler_thread.join(timeout=1.0)
        collector.end_measurement()

    # Get the metrics
    metrics = collector.get_metrics()
    return metrics, result


def is_energy_measurement_available() -> Dict[str, bool]:
    """
    Check which energy measurement methods are available on this system.
    
    Returns:
        Dict with availability info for different methods
    """
    available = {
        "cpu_rapl": PYRAPL_AVAILABLE,
        "gpu_nvml": NVML_AVAILABLE,
    }
    
    # For NVML, let's check if we can actually get GPU power
    if NVML_AVAILABLE and torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                pynvml.nvmlDeviceGetPowerUsage(handle)
                available["gpu_power_reading"] = True
            except pynvml.NVMLError:
                available["gpu_power_reading"] = False
            pynvml.nvmlShutdown()
        except Exception:
            available["gpu_power_reading"] = False
    else:
        available["gpu_power_reading"] = False
        
    # Overall availability
    available["any_method_available"] = any(available.values())
    
    # Add some helpful context about why things might not be available
    if not available["cpu_rapl"]:
        available["cpu_rapl_install_hint"] = "Try: pip install pyRAPL"
    
    if not available["gpu_nvml"]:
        available["gpu_nvml_install_hint"] = "Try: pip install nvidia-ml-py"
        
    if not available["gpu_power_reading"]:
        available["gpu_power_hint"] = "Your GPU might not support power readings or you need admin privileges"
    
    return available
