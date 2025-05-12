"""
Tests for the batch-sweep runner and CLI interface.

This module tests the benchmarking runner that orchestrates model inference
with various batch sizes and collects metrics.
"""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import torch
from pathlib import Path
from argparse import Namespace

# For direct imports from the source directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data.dataset as ds_module
from transformers import AutoTokenizer, DistilBertTokenizerFast, DistilBertTokenizer

from src.runner import (
    setup_arg_parser,
    parse_args_to_config,
    prepare_dataset,
    load_tokenizer,
    BatchSweepRunner,
    BenchmarkConfig,
)


def test_runner_import():
    """Test that we can import the runner module."""
    try:
        from src.runner import setup_arg_parser

        assert callable(setup_arg_parser)
    except ImportError:
        pytest.fail("Could not import runner module. Make sure src/runner.py exists.")


def test_benchmark_config_pydantic():
    """Test the benchmark configuration using pydantic."""
    from src.runner import BenchmarkConfig

    # Test with valid config
    config = BenchmarkConfig(
        model_name="distilbert-base-uncased",
        batch_sizes=[1, 2, 4, 8],
        device="cpu",
        dataset_name="glue",
        dataset_subset="sst2",
        dataset_split="validation",
        max_length=128,
        warmup_runs=5,
        iterations=10,
        metrics={"latency": True, "memory": True, "energy": True},
        output_file="results.jsonl",
        cache_dir="data/cached",
    )

    assert config.model_name == "distilbert-base-uncased"
    assert config.batch_sizes == [1, 2, 4, 8]
    assert config.device == "cpu"

    # Test with invalid batch sizes
    with pytest.raises(ValueError):
        BenchmarkConfig(
            model_name="distilbert-base-uncased",
            batch_sizes=[-1, 0, 4],  # Negative and zero batch sizes should be invalid
            device="cpu",
        )


def test_load_config_from_yaml():
    """Test loading configuration from a YAML file."""
    from src.runner import load_config

    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
# Test configuration
model:
  name: "distilbert-base-uncased"
  task: "sequence-classification"
  max_length: 128

benchmark:
  devices: ["cpu"]
  batch_sizes: [1, 2, 4, 8]
  warmup_runs: 3
  iterations: 5
  metrics:
    latency: true
    throughput: true
    cpu_memory: true
  output:
    format: "json"
        """
        )

    try:
        # Load the config
        config = load_config(f.name)

        # Check parsed values
        assert config["model"]["name"] == "distilbert-base-uncased"
        assert config["benchmark"]["batch_sizes"] == [1, 2, 4, 8]
        assert config["benchmark"]["metrics"]["latency"] is True
    finally:
        # Clean up the temporary file
        os.unlink(f.name)


def test_arg_parser():
    """Test the argument parser setup with updated flags."""
    from src.runner import setup_arg_parser

    parser = setup_arg_parser()

    # Test with minimal args using new --model-name flag
    args = parser.parse_args(["--model-name", "distilbert-base-uncased"])
    assert args.model_name == "distilbert-base-uncased"

    # Test with batch sizes (flag unchanged)
    args = parser.parse_args(["--batch-sizes", "1,2,4,8"])
    assert args.batch_sizes == "1,2,4,8"

    # Test with output file using new --output-file flag
    args = parser.parse_args(["--output-file", "results.jsonl"])
    assert args.output_file == "results.jsonl"


def test_batch_sweep_runner_init():
    """Test BatchSweepRunner initialization."""
    from src.runner import BatchSweepRunner, BenchmarkConfig

    # Create a configuration
    config = BenchmarkConfig(
        model_name="distilbert-base-uncased",
        batch_sizes=[1, 2],
        device="cpu",
        dataset_name="glue",
        dataset_subset="sst2",
        dataset_split="validation",
        max_length=128,
        warmup_runs=1,
        iterations=2,
        metrics={"latency": True},
        output_file="results.jsonl",
        cache_dir="data/cached",
    )

    # Initialize runner
    runner = BatchSweepRunner(config)

    # Verify runner properties
    assert runner.config == config
    assert runner.model_name == "distilbert-base-uncased"
    assert runner.batch_sizes == [1, 2]


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([101, 2054, 2003, 2019, 102]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "label": torch.tensor(1),
        }


class MockModel:
    """Mock model for testing."""

    def __init__(self):
        self.device = "cpu"
        self.config = MagicMock()

    def __call__(self, **kwargs):
        return {"logits": [[0.1, 0.9]]}

    def eval(self):
        return self


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for the runner."""
    with patch("src.runner.load_model") as mock_load_model, patch(
        "src.runner.load_tokenizer"
    ) as mock_load_tokenizer, patch(
        "src.runner.prepare_dataset"
    ) as mock_prepare_dataset, patch(
        "src.runner.get_model_metadata"
    ) as mock_get_model_metadata, patch(
        "src.runner.LatencyMetricCollector"
    ) as mock_latency, patch(
        "src.runner.MemoryMetricCollector"
    ) as mock_memory, patch(
        "src.runner.EnergyMetricCollector"
    ) as mock_energy:
        # Set up mocks
        mock_load_model.return_value = MockModel()
        mock_load_tokenizer.return_value = MagicMock()
        mock_prepare_dataset.return_value = MockDataset()
        mock_get_model_metadata.return_value = {"num_parameters": 66362880}
        
        # Set up the metric collectors to return valid metrics
        mock_latency_instance = MagicMock()
        mock_latency_instance.measure.return_value = {"latency_ms_mean": 10.0, "throughput_mean": 100.0}
        mock_latency.return_value = mock_latency_instance
        
        mock_memory_instance = MagicMock()
        mock_memory_instance.measure.return_value = {"cpu_memory_mb_max": 100.0}
        mock_memory.return_value = mock_memory_instance
        
        mock_energy_instance = MagicMock()
        mock_energy_instance.measure.return_value = {"measurement_time_s": 1.0}
        mock_energy.return_value = mock_energy_instance

        yield


def test_run_single_batch(mock_dependencies):
    """Test running a single batch."""
    from src.runner import BatchSweepRunner, BenchmarkConfig

    # Create a configuration
    config = BenchmarkConfig(
        model_name="distilbert-base-uncased",
        batch_sizes=[2],
        device="cpu",
        dataset_name="glue",
        dataset_subset="sst2",
        dataset_split="validation",
        max_length=128,
        warmup_runs=1,
        iterations=2,
        metrics={"latency": True},
        output_file="results.jsonl",
        cache_dir="data/cached",
    )

    # Create a temporary output file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        config.output_file = f.name

        # Initialize runner
        runner = BatchSweepRunner(config)

        # Run benchmark
        runner.run()

        # Check that output file was created and has content
        assert os.path.exists(f.name)
        with open(f.name, "r") as f_check:
            results = [json.loads(line) for line in f_check if line.strip()]
            assert len(results) == 1  # One batch size
            assert "batch_size" in results[0]
            assert "metrics" in results[0]
            assert "latency_ms_mean" in results[0]["metrics"]

        # Clean up
        os.unlink(f.name)


def test_run_batch_sweep(mock_dependencies):
    """Test running a sweep over multiple batch sizes."""
    from src.runner import BatchSweepRunner, BenchmarkConfig

    # Create a configuration with multiple batch sizes
    config = BenchmarkConfig(
        model_name="distilbert-base-uncased",
        batch_sizes=[1, 2, 4],
        device="cpu",
        dataset_name="glue",
        dataset_subset="sst2",
        dataset_split="validation",
        max_length=128,
        warmup_runs=1,
        iterations=2,
        metrics={"latency": True, "memory": True},
        output_file="results.jsonl",
        cache_dir="data/cached",
    )

    # Create a temporary output file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        config.output_file = f.name

        # Initialize runner
        runner = BatchSweepRunner(config)

        # Run benchmark
        runner.run()

        # Check that output file was created and has content
        assert os.path.exists(f.name)
        with open(f.name, "r") as f_check:
            results = [json.loads(line) for line in f_check if line.strip()]
            assert len(results) == 3  # Three batch sizes

            # Check that results include all batch sizes
            batch_sizes = [result["batch_size"] for result in results]
            assert set(batch_sizes) == {1, 2, 4}

            # Check that metrics are included
            assert "latency_ms_mean" in results[0]["metrics"]
            assert "cpu_memory_mb_max" in results[0]["metrics"]

        # Clean up
        os.unlink(f.name)


def test_smoke_test_mode(mock_dependencies):
    """Test running in smoke test mode with minimal configuration."""
    from src.runner import BatchSweepRunner, BenchmarkConfig

    # Create a configuration with smoke test flag
    config = BenchmarkConfig(
        model_name="distilbert-base-uncased",
        batch_sizes=[1],
        device="cpu",
        dataset_name="glue",
        dataset_subset="sst2",
        dataset_split="validation",
        max_length=128,
        warmup_runs=1,
        iterations=1,
        metrics={"latency": True},
        output_file="results.jsonl",
        cache_dir="data/cached",
        smoke_test=True,  # Enable smoke test mode
    )

    # Create a temporary output file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        config.output_file = f.name

        # Initialize runner
        runner = BatchSweepRunner(config)

        # Run benchmark
        runner.run()

        # Check that output file was created and has content
        assert os.path.exists(f.name)
        with open(f.name, "r") as f_check:
            results = [json.loads(line) for line in f_check if line.strip()]
            assert len(results) == 1
            assert results[0]["smoke_test"] is True

        # Clean up
        os.unlink(f.name)


def test_cli_parse_returns_correct_config():
    parser = setup_arg_parser()
    args = parser.parse_args([
        "--model-name", "test-model",
        "--dataset-name", "mydata",
        "--dataset-subset", "sub",
        "--dataset-split", "split",
        "--output-file", "out.jsonl",
        "--config-file", "cfg.yaml",
        "--batch-sizes", "2,4,8",
        "--device", "cpu",
        "--warmup-runs", "3",
        "--iterations", "5",
        "--verbose", "debug",
    ])
    config = parse_args_to_config(args)
    assert config.model_name == "test-model"
    assert config.dataset_name == "mydata"
    assert config.dataset_subset == "sub"
    assert config.dataset_split == "split"
    assert config.output_file == "out.jsonl"
    assert config.batch_sizes == [2, 4, 8]
    assert config.device == "cpu"
    assert config.warmup_runs == 3
    assert config.iterations == 5
    assert config.verbose.value == "debug"


def test_prepare_dataset(monkeypatch):
    dummy = [{"input_ids": [1, 2], "attention_mask": [1, 1], "label": 0}]
    # Mock the functions directly in src.runner module where they're imported
    monkeypatch.setattr("src.runner.load_dataset", lambda dataset_name, subset, split, cache_dir: dummy)
    monkeypatch.setattr("src.runner.tokenize_dataset", lambda dataset, model_name, max_length, cache_dir: dummy)
    dataset = prepare_dataset("dname", "sub", "split", "mname", 10, "cache")
    assert len(dataset) == 1
    item = dataset[0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["label"], torch.Tensor)


def test_load_tokenizer_fallback(monkeypatch):
    # Force both AutoTokenizer and fast fallback to error out, then succeed with standard
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("auto fail")))
    monkeypatch.setattr(DistilBertTokenizerFast, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("fast fail")))
    monkeypatch.setattr(DistilBertTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    tok = load_tokenizer("model", cache_dir="cache")
    assert tok is not None


def test_smoke_benchmark_run(tmp_path, monkeypatch):
    # Define DummyModel class to be used by the monkeypatch
    class DummyModel:
        def eval(self):
            return self
        def __call__(self, input_ids, attention_mask):
            return {"logits": torch.tensor([[0.1, 0.9]])}
        def to(self, device):
            return self

    # Monkeypatch model loading and metrics
    monkeypatch.setattr("src.runner.load_model", lambda model_name, device, cache_dir: DummyModel())
    monkeypatch.setattr("src.runner.get_model_metadata", lambda model_name, cache_dir: {"info": True})
    monkeypatch.setattr("src.runner.measure_inference_latency", lambda **kwargs: {"latency_ms_mean": 1.0, "throughput_mean": 2.0})
    monkeypatch.setattr("src.runner.measure_peak_memory_usage", lambda func, device, sampling_interval_ms, track_gpu: ({"cpu_memory_mb_max": 1.0}, None))
    monkeypatch.setattr("src.runner.measure_energy_consumption", lambda func, device, sampling_interval_ms, use_rapl, use_nvml: ({"energy_j": 0.5}, None))
    monkeypatch.setattr("src.runner.is_energy_measurement_available", lambda: {"cpu_energy_supported": False, "gpu_energy_supported": False})
    # Use a minimal dataset for one batch
    monkeypatch.setattr(BatchSweepRunner, "_prepare_dataset", lambda self: [{"input_ids": torch.tensor([[0]]), "attention_mask": torch.tensor([[1]]), "label": torch.tensor([0])}])

    # Prepare config for smoke test, writing to tmp path
    out_file = tmp_path / "results.jsonl"
    config = BenchmarkConfig(output_file=str(out_file), smoke_test=True)
    runner = BatchSweepRunner(config)
    results = runner.run()

    # Should run only the first (smallest) batch once
    assert isinstance(results, list)
    assert len(results) == 1
    # Check output file content
    text = out_file.read_text().strip().splitlines()
    assert len(text) == 1
    # Validate JSON structure
    record = json.loads(text[0])
    assert record.get("metrics", {}).get("latency_ms_mean") == 1.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
