"""
Runner script for DistilBERT benchmarking.

This script coordinates the benchmarking process by loading the dataset,
tokenizing it, loading the model, and measuring various metrics.
"""

import os
import sys
import time
import json
import logging
import argparse
import platform
import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

# Add parent directory to sys.path to allow importing from data module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
import pydantic
import colorama
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import load_dataset, tokenize_dataset
from src.model import load_model, get_model_metadata
from src.metrics.latency import measure_inference_latency
from src.metrics.memory import (
    measure_peak_memory_usage,
    get_gpu_info,
)
from src.metrics.energy import (
    measure_energy_consumption,
    is_energy_measurement_available,
)

# Add tokenizer import
from transformers import AutoTokenizer, DistilBertTokenizer, DistilBertTokenizerFast

# Initialize colorama for cross-platform colored terminal output
colorama.init()

from contextlib import nullcontext


# Add load_tokenizer function
def load_tokenizer(model_name: str, cache_dir: str = "data/cached") -> any:
    """
    Load tokenizer for the specified model.
    
    Args:
        model_name: HuggingFace model name or path
        cache_dir: Directory to cache the downloaded models
        
    Returns:
        The loaded tokenizer
    """
    logger.info(f"Loading tokenizer for {model_name}")
    
    try:
        # First try to load the tokenizer using AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        logger.info(f"Loaded tokenizer using AutoTokenizer: {tokenizer.__class__.__name__}")
        return tokenizer
    except Exception as e:
        # Fall back to specific tokenizer
        logger.warning(f"Failed to load tokenizer with AutoTokenizer, falling back to DistilBertTokenizer: {e}")
        try:
            # Try the fast tokenizer first
            tokenizer = DistilBertTokenizerFast.from_pretrained(
                model_name, 
                cache_dir=cache_dir
            )
            logger.info("Loaded DistilBertTokenizerFast")
            return tokenizer
        except Exception as e2:
            # Fall back to standard tokenizer
            logger.warning(f"Failed to load fast tokenizer, falling back to standard: {e2}")
            tokenizer = DistilBertTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir
            )
            logger.info("Loaded DistilBertTokenizer")
            return tokenizer


# Configure logging with colored output
class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored log output."""

    COLORS = {
        "DEBUG": colorama.Fore.BLUE,
        "INFO": colorama.Fore.GREEN,
        "WARNING": colorama.Fore.YELLOW,
        "ERROR": colorama.Fore.RED,
        "CRITICAL": colorama.Fore.RED + colorama.Style.BRIGHT,
    }

    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, colorama.Fore.WHITE)
        return f"{color}{log_message}{colorama.Style.RESET_ALL}"


# Set up the logger
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

# Console handler with colored output
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


class LogLevel(str, Enum):
    """Log levels for the benchmark runner."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class BenchmarkConfig(pydantic.BaseModel):
    """Pydantic model for benchmark configuration."""

    # Model configuration
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    model_task: str = "sequence-classification"
    max_length: int = pydantic.Field(default=128, ge=1, le=512)

    # Dataset configuration
    dataset_name: str = "glue"
    dataset_subset: str = "sst2"
    dataset_split: str = "validation"

    # Device configuration
    device: str = "cpu"
    cuda_non_blocking: bool = True  # Use non-blocking CUDA transfers
    mixed_precision: bool = False  # Use mixed precision (fp16) for GPU

    # Benchmark configuration
    batch_sizes: List[int] = pydantic.Field(
        default=[1, 2, 4, 8, 16, 32, 64], description="Batch sizes to benchmark"
    )
    warmup_runs: int = pydantic.Field(default=5, ge=0)
    iterations: int = pydantic.Field(default=20, ge=1)

    # Metrics configuration
    metrics: Dict[str, bool] = {
        "latency": True,
        "throughput": True,
        "memory": True,
        "energy": True,
    }

    # Output configuration
    output_file: str = "benchmark_results.jsonl"
    include_system_info: bool = True

    # Path configuration
    cache_dir: str = "data/cached"

    # Special modes
    smoke_test: bool = False
    verbose: LogLevel = LogLevel.INFO

    # Logging configuration
    log_file: Optional[str] = None

    @pydantic.field_validator("batch_sizes")
    @classmethod
    def validate_batch_sizes(cls, v):
        """Validate that batch sizes are positive integers."""
        if not all(isinstance(batch, int) and batch > 0 for batch in v):
            raise ValueError("All batch sizes must be positive integers")
        return v

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with configuration
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using default settings.")
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="DistilBERT Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Model to benchmark",
    )
    model_group.add_argument(
        "--max-length", type=int, default=128, help="Maximum sequence length"
    )

    # Dataset configuration
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--dataset-name", type=str, default="glue", help="Dataset name to use"
    )
    dataset_group.add_argument(
        "--dataset-subset", type=str, default="sst2", help="Dataset subset to use"
    )
    dataset_group.add_argument(
        "--dataset-split", type=str, default="validation", help="Dataset split to use"
    )

    # Benchmark configuration
    benchmark_group = parser.add_argument_group("Benchmark Configuration")
    benchmark_group.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Comma-separated list of batch sizes to benchmark",
    )
    benchmark_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu, cuda, cuda:0, etc.)",
    )
    benchmark_group.add_argument(
        "--cuda-non-blocking",
        action="store_true",
        help="Use non-blocking CUDA transfers (GPU only)",
    )
    benchmark_group.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision (fp16) for GPU inference",
    )
    benchmark_group.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup runs before measuring",
    )
    benchmark_group.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations to average metrics over",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-file",
        type=str,
        default="benchmark_results.jsonl",
        help="Path to output file for results (JSONL format)",
    )
    output_group.add_argument(
        "--verbose",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Verbosity level",
    )
    output_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: log to console only)",
    )

    # Path configuration
    path_group = parser.add_argument_group("Path Configuration")
    path_group.add_argument(
        "--config-file", type=str, default="config.yaml", help="Path to configuration file"
    )
    path_group.add_argument(
        "--cache-dir",
        type=str,
        default="data/cached",
        help="Directory to cache datasets and models",
    )

    # Special modes
    special_group = parser.add_argument_group("Special Modes")
    special_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run in smoke test mode (minimal configuration for testing)",
    )
    special_group.add_argument(
        "--list-devices", action="store_true", help="List available devices and exit"
    )

    return parser


def parse_args_to_config(args: argparse.Namespace) -> BenchmarkConfig:
    """
    Parse command line arguments into a BenchmarkConfig.

    Args:
        args: Command line arguments

    Returns:
        BenchmarkConfig instance
    """
    # Check for device listing mode
    if hasattr(args, "list_devices") and args.list_devices:
        devices = get_available_devices()
        print("\nAvailable devices:")
        print(f"CPU: {devices['cpu'][0]}")

        if devices["cuda"]:
            print(f"CUDA (GPU): {', '.join(devices['cuda'])}")
            # Print GPU details if available
            for i, device in enumerate(devices["cuda"]):
                print(f"\nCUDA:{i} - {torch.cuda.get_device_name(i)}")
                print(
                    f"  Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB"
                )
                print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
        else:
            print("CUDA (GPU): Not available")

        if devices["mps"]:
            print(f"MPS (Apple Silicon): {devices['mps'][0]}")
        else:
            print("MPS (Apple Silicon): Not available")

        print("\nUse --device <device> to select a specific device for benchmarking.\n")
        sys.exit(0)

    # Parse batch sizes from comma-separated string
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]

    # Get YAML config if specified
    yaml_config = load_config(args.config_file)

    # Extract values from YAML with command line args as fallback
    model_config = yaml_config.get("model", {})
    dataset_config = yaml_config.get("dataset", {})
    benchmark_config = yaml_config.get("benchmark", {})

    # Create metrics dict from YAML or default to all True
    metrics_config = benchmark_config.get("metrics", {})
    metrics = {
        "latency": metrics_config.get("latency", True),
        "throughput": metrics_config.get("throughput", True),
        "memory": metrics_config.get("cpu_memory", True)
        or metrics_config.get("memory", True),
        "energy": metrics_config.get("energy", True),
    }

    # Get device from args or YAML
    device = args.device or benchmark_config.get("devices", ["cpu"])[0]

    # Validate device
    available_devices = get_available_devices()
    all_devices = (
        available_devices["cpu"] + available_devices["cuda"] + available_devices["mps"]
    )
    if (
        device not in all_devices and device != "cuda"
    ):  # Special case for "cuda" which maps to cuda:0
        if (
            device.startswith("cuda:")
            and int(device.split(":")[-1]) >= torch.cuda.device_count()
        ):
            logger.warning(f"Device {device} not available. Falling back to CPU.")
            device = "cpu"

    # Create the config
    config = BenchmarkConfig(
        # Model configuration
        model_name=args.model_name or model_config.get("name"),
        model_task=model_config.get("task", "sequence-classification"),
        max_length=args.max_length or model_config.get("max_length", 128),
        # Dataset configuration
        dataset_name=args.dataset_name or dataset_config.get("name"),
        dataset_subset=args.dataset_subset or dataset_config.get("subset"),
        dataset_split=args.dataset_split or dataset_config.get("split"),
        # Device configuration
        device=device,
        cuda_non_blocking=args.cuda_non_blocking
        if hasattr(args, "cuda_non_blocking")
        else False,
        mixed_precision=args.mixed_precision
        if hasattr(args, "mixed_precision")
        else False,
        # Benchmark configuration
        batch_sizes=batch_sizes
        or benchmark_config.get("batch_sizes", [1, 2, 4, 8, 16, 32, 64]),
        warmup_runs=args.warmup_runs or benchmark_config.get("warmup_runs", 5),
        iterations=args.iterations or benchmark_config.get("iterations", 20),
        # Metrics configuration
        metrics=metrics,
        # Output configuration
        output_file=args.output_file,
        include_system_info=True,
        # Path configuration
        cache_dir=args.cache_dir or dataset_config.get("cache_dir"),
        # Special modes
        smoke_test=args.smoke_test,
        verbose=args.verbose,
        # Logging configuration
        log_file=args.log_file,
    )

    # Override with smoke test settings if enabled
    if config.smoke_test:
        logger.info("Running in smoke test mode with minimal configuration")
        config.batch_sizes = [1]
        config.warmup_runs = 1
        config.iterations = 2

    return config


def prepare_dataset(
    dataset_name: str,
    subset: str,
    split: str,
    model_name: str,
    max_length: int,
    cache_dir: str,
) -> torch.utils.data.Dataset:
    """
    Prepare the dataset for benchmarking by loading and tokenizing it.

    Args:
        dataset_name: Name of the dataset
        subset: Subset of the dataset
        split: Split of the dataset
        model_name: Model name for tokenizer
        max_length: Maximum sequence length
        cache_dir: Directory to cache datasets

    Returns:
        Tokenized dataset ready for benchmarking
    """
    logger.info(f"Preparing dataset: {dataset_name}/{subset}/{split}")

    # Load dataset
    start_time = time.time()
    dataset = load_dataset(
        dataset_name=dataset_name, subset=subset, split=split, cache_dir=cache_dir
    )
    logger.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    logger.info(f"Dataset size: {len(dataset)} examples")

    # Tokenize dataset
    start_time = time.time()
    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        model_name=model_name,
        max_length=max_length,
        cache_dir=cache_dir,
    )
    logger.info(f"Dataset tokenized in {time.time() - start_time:.2f} seconds")

    # Convert to PyTorch Dataset
    class BenchmarkDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"]),
                "attention_mask": torch.tensor(item["attention_mask"]),
                "label": torch.tensor(item["label"]),
            }

    return BenchmarkDataset(tokenized_dataset)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for benchmarking context.

    Returns:
        Dictionary with system information
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }

    # Add PyTorch version
    info["torch_version"] = torch.__version__

    # Add GPU information if available
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_info"] = []
        for i in range(torch.cuda.device_count()):
            info["gpu_info"].append(
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                }
            )

    return info


# Add metric collector wrapper classes to simplify testing and make the interface consistent
class LatencyMetricCollector:
    """Wrapper for latency measurement with a consistent collector interface."""

    def __init__(self, device: str):
        self.device = device

    def measure(self, model, input_ids, attention_mask, warmup_runs=5, iterations=20):
        """
        Measure inference latency.

        Args:
            model: The model to benchmark
            input_ids: Input tensor
            attention_mask: Attention mask tensor
            warmup_runs: Number of warmup runs before measuring
            iterations: Number of iterations to average metrics over

        Returns:
            Dictionary with metrics
        """
        return measure_inference_latency(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            warmup_runs=warmup_runs,
            iterations=iterations,
            device=self.device,
        )

    def get_metrics(self):
        """Return the collected metrics."""
        return self._metrics

    def set_metrics(self, metrics):
        """Set metrics for testing."""
        self._metrics = metrics


class MemoryMetricCollector:
    """Wrapper for memory measurement with a consistent collector interface."""

    def __init__(self, device: str):
        self.device = device
        self.track_gpu = device.startswith("cuda")
        self._metrics = {}

    def measure(self, func, sampling_interval_ms=10):
        """
        Measure peak memory usage during function execution.

        Args:
            func: The function to benchmark
            sampling_interval_ms: Sampling interval for memory measurement

        Returns:
            Dictionary with metrics
        """
        metrics, _ = measure_peak_memory_usage(
            func=func,
            device=self.device,
            sampling_interval_ms=sampling_interval_ms,
            track_gpu=self.track_gpu,
        )
        self._metrics = metrics
        return metrics

    def get_metrics(self):
        """Return the collected metrics."""
        return self._metrics

    def set_metrics(self, metrics):
        """Set metrics for testing."""
        self._metrics = metrics


class EnergyMetricCollector:
    """Wrapper for energy measurement with a consistent collector interface."""

    def __init__(self, device: str):
        self.device = device
        self.use_rapl = is_energy_measurement_available().get("cpu_energy_supported", False)
        self.use_nvml = (
            is_energy_measurement_available().get("gpu_energy_supported", False)
            and device.startswith("cuda")
        )
        self._metrics = {}

    def measure(self, func, sampling_interval_ms=50):
        """
        Measure energy consumption during function execution.

        Args:
            func: The function to benchmark
            sampling_interval_ms: Sampling interval for energy measurement

        Returns:
            Dictionary with metrics
        """
        if not self.use_rapl and not self.use_nvml:
            self._metrics = {"energy_measurement_available": False}
            return self._metrics

        metrics, _ = measure_energy_consumption(
            func=func,
            device=self.device,
            sampling_interval_ms=sampling_interval_ms,
            use_rapl=self.use_rapl,
            use_nvml=self.use_nvml,
        )
        self._metrics = metrics
        return metrics

    def get_metrics(self):
        """Return the collected metrics."""
        return self._metrics

    def set_metrics(self, metrics):
        """Set metrics for testing."""
        self._metrics = metrics


class BatchSweepRunner:
    """Benchmark runner that sweeps over batch sizes."""

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the runner.

        Args:
            config: Configuration object
        """
        # Clear any existing handlers to prevent duplicate logs
        logger.handlers.clear()
        self.config = config
        
        # Extract configuration values
        self.model_name = config.model_name
        self.model_task = config.model_task
        self.max_length = config.max_length
        self.dataset_name = config.dataset_name
        self.dataset_subset = config.dataset_subset
        self.dataset_split = config.dataset_split
        self.device = config.device
        self.batch_sizes = config.batch_sizes
        self.warmup_runs = config.warmup_runs
        self.iterations = config.iterations
        self.metrics = config.metrics
        self.output_file = config.output_file
        self.cache_dir = config.cache_dir
        self.smoke_test = config.smoke_test
        self.include_system_info = config.include_system_info
        self.mixed_precision = config.mixed_precision
        self.cuda_non_blocking = config.cuda_non_blocking

        # Set up logging
        log_level = getattr(logging, config.verbose.upper())
        logger.setLevel(log_level)
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)

        # Set up metric collectors
        self.latency_collector = LatencyMetricCollector(device=self.device)
        self.memory_collector = MemoryMetricCollector(device=self.device)
        self.energy_collector = EnergyMetricCollector(device=self.device)

        # Initialize output file with empty content
        logger.info(f"Initializing output file: {self.output_file}")
        with open(self.output_file, "w") as f:
            pass

    def _load_model_and_metadata(self):
        """
        Load the model and its associated metadata.
        Returns:
            Tuple of (model, metadata dict)
        """
        logger.info(f"Loading model: {self.model_name}")
        model = load_model(
            model_name=self.model_name,
            device=self.device,
            cache_dir=self.cache_dir,
        )
        model.eval()
        metadata = get_model_metadata(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
        )
        return model, metadata

    def _load_tokenizer(self):
        """
        Load tokenizer for the specified model.
        Returns:
            The loaded tokenizer instance
        """
        logger.info(f"Loading tokenizer for {self.model_name}")
        return load_tokenizer(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
        )

    def _prepare_dataset(self):
        """
        Prepare and tokenize the dataset for benchmarking.
        Returns:
            A PyTorch Dataset ready for DataLoader
        """
        logger.info(f"Preparing dataset: {self.dataset_name}/{self.dataset_subset}")
        return prepare_dataset(
            dataset_name=self.dataset_name,
            subset=self.dataset_subset,
            split=self.dataset_split,
            model_name=self.model_name,
            max_length=self.max_length,
            cache_dir=self.cache_dir,
        )

    def run(self):
        """Run the benchmark sweep."""
        logger.info(f"Starting benchmark sweep for {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch sizes: {self.batch_sizes}")

        # Load model and metadata
        model, model_metadata = self._load_model_and_metadata()
        
        # Load tokenizer
        tokenizer = self._load_tokenizer()

        # Prepare dataset
        dataset = self._prepare_dataset()

        # Configure precision
        amp_dtype = None
        amp_context = nullcontext()
        if self.device.startswith("cuda") and self.mixed_precision:
            logger.info("Using mixed precision (fp16)")
            amp_dtype = torch.float16
            amp_context = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

        # Run benchmarks for each batch size
        results = []
        for batch_size in tqdm(self.batch_sizes, desc="Benchmarking batch sizes"):
            logger.info(f"Running benchmark for batch size {batch_size}")

            # In smoke test mode, only run the smallest batch size
            if self.smoke_test and batch_size != min(self.batch_sizes):
                logger.info(f"Skipping batch size {batch_size} in smoke test mode")
                continue

            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=0,  # Disable multiprocessing for more accurate measurements
            )

            # Run the benchmark
            batch_result = self._run_batch_benchmark(model, dataloader, batch_size)

            # Add batch size to results
            batch_result["batch_size"] = batch_size

            # Create a result object with all information
            result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "batch_size": batch_size,
                "model": self.model_name,
                "device": self.device,
                "dataset": f"{self.dataset_name}/{self.dataset_subset}",
                "split": self.dataset_split,
                "max_length": self.max_length,
                "warmup_runs": self.warmup_runs,
                "iterations": self.iterations,
                "mixed_precision": self.mixed_precision,
                "metrics": batch_result,
                "smoke_test": self.smoke_test,
            }

            # Add system info if requested
            if self.include_system_info:
                result["system_info"] = get_system_info()

            # Add model info
            if model_metadata:
                result["model_info"] = model_metadata

            # Save result to output file
            with open(self.output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            results.append(result)

            logger.info(f"Completed benchmark for batch size {batch_size}")
            # Log a sample of the metrics
            if "latency_ms_mean" in batch_result:
                logger.info(f"  Latency: {batch_result['latency_ms_mean']:.2f} ms")
            if "throughput_mean" in batch_result:
                logger.info(
                    f"  Throughput: {batch_result['throughput_mean']:.2f} samples/s"
                )
            if self.device.startswith("cuda") and "gpu_memory_mb_max" in batch_result:
                logger.info(f"  GPU Memory: {batch_result['gpu_memory_mb_max']:.2f} MB")

        logger.info(f"Benchmark completed. Results saved to: {self.output_file}")
        return results

    def _run_batch_benchmark(self, model, dataloader, batch_size):
        """
        Run benchmark for a specific batch size.

        Args:
            model: The model to benchmark
            dataloader: DataLoader for the dataset
            batch_size: Current batch size

        Returns:
            Dictionary with metrics
        """
        # Measure data preparation time
        prep_start = time.perf_counter()
        batch = next(iter(dataloader))
        # Extract inputs
        if isinstance(batch["input_ids"], list):
            input_ids = torch.tensor(batch["input_ids"]).to(self.device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(self.device)
        else:
            non_blocking = (
                self.config.cuda_non_blocking
                if self.device.startswith("cuda")
                else False
            )
            input_ids = batch["input_ids"].to(self.device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(
                self.device, non_blocking=non_blocking
            )
        # Ensure CUDA transfers are complete before starting measurements
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        prep_end = time.perf_counter()
        data_prep_time = prep_end - prep_start

        # Set sequence length for reporting
        seq_length = input_ids.shape[1]

        # Initialize metrics with batch and preprocessing times
        metrics = {
            "batch_size": batch_size,
            "sequence_length": seq_length,
            "data_prep_time_s": data_prep_time,
        }

        # Measure latency
        if self.metrics.get("latency", True) or self.metrics.get("throughput", True):
            infer_start = time.perf_counter()
            latency_metrics = self.latency_collector.measure(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                warmup_runs=self.warmup_runs,
                iterations=self.iterations,
            )
            infer_end = time.perf_counter()
            # Record inference time
            metrics["inference_time_s"] = infer_end - infer_start
            metrics.update(latency_metrics)

        # Measure memory usage
        if self.metrics.get("memory", True):
            # Define inference function for memory measurement
            def memory_inference_func():
                # Run multiple iterations to get consistent memory measurements
                outputs = []
                for _ in range(5):
                    with torch.no_grad():
                        output = model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        outputs.append(output)
                return outputs

            memory_metrics = self.memory_collector.measure(
                func=memory_inference_func,
                sampling_interval_ms=10,
            )
            metrics.update(memory_metrics)

        # Measure energy consumption
        if self.metrics.get("energy", True):
            # Check if energy measurement is available
            energy_available = is_energy_measurement_available()

            # Determine if we should use RAPL and NVML
            use_rapl = energy_available.get("cpu_energy_supported", False)
            use_nvml = energy_available.get(
                "gpu_energy_supported", False
            ) and self.device.startswith("cuda")

            if use_rapl or use_nvml:
                # Define inference function for energy measurement
                def energy_inference_func():
                    # Run multiple batches to get meaningful energy measurements
                    with torch.no_grad():
                        for _ in range(10):
                            output = model(
                                input_ids=input_ids, attention_mask=attention_mask
                            )
                    return output

                energy_metrics = self.energy_collector.measure(
                    func=energy_inference_func,
                    sampling_interval_ms=50,
                )
                metrics.update(energy_metrics)
            else:
                logger.warning("Energy measurement not available on this system")
                metrics["energy_measurement_available"] = False

        return metrics


# Add this function to detect available GPU devices
def get_available_devices() -> Dict[str, List[str]]:
    """
    Detect available CPU and GPU devices for benchmarking.

    Returns:
        Dictionary with available devices by type
    """
    devices = {"cpu": ["cpu"], "cuda": [], "mps": []}

    # Check CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices["cuda"].append(f"cuda:{i}")
        logger.info(
            f"Found {len(devices['cuda'])} CUDA devices: {', '.join(devices['cuda'])}"
        )

    # Check MPS (Apple Silicon) device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices["mps"].append("mps")
        logger.info("Found Apple Silicon MPS device")

    return devices


def main():
    """Main function to run the benchmark."""
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Convert args to config
    config = parse_args_to_config(args)

    # Print banner
    print(
        colorama.Fore.CYAN
        + """
  ____  _     _   _ _ ____  _____ ____ _____   ____                  _                          _
 |  _ \\(_)___| |_(_) | __ )| ____|  _ \\_   _| | __ )  ___ _ __   ___| |__  _ __ ___   __ _ _ __| | __
 | | | | / __| __| | |  _ \\|  _| | |_) || |   |  _ \\ / _ \\ '_ \\ / __| '_ \\| '_ ` _ \\ / _` | '__| |/ /
 | |_| | \\__ \\ |_| | | |_) | |___|  _ < | |   | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <
 |____/|_|___/\\__|_|_|____/|_____|_| \\_\\|_|   |____/ \\___|_| |_|\\___|_| |_|_| |_| |_|\\__,_|_|  |_|\\_\\

"""
        + colorama.Style.RESET_ALL
    )

    # Create and run batch sweep
    runner = BatchSweepRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
