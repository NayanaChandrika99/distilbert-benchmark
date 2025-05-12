"""
Tests for analytics and visualization components.

This module tests the analysis scripts for result processing and visualization.
"""

import os
import sys
import json
import tempfile
import unittest
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import analysis modules
from analysis import plot_metrics, generate_report


class TestAnalysisScripts(unittest.TestCase):
    """Test suite for analysis scripts."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create sample benchmark results
        self.sample_results = [
            {
                "timestamp": "2025-05-01T12:00:00.000000",
                "model": "distilbert-base-uncased",
                "batch_size": 1,
                "device": "cpu",
                "dataset": "glue/sst2/validation",
                "metrics": {
                    "latency_ms_mean": 50.0,
                    "latency_ms_std": 5.0,
                    "latency_ms_p90": 55.0,
                    "throughput_mean": 20.0,
                    "batch_size": 1,
                    "sequence_length": 128,
                    "num_measurements": 10,
                    "device": "cpu",
                    "warmup_runs": 2,
                    "iterations": 10,
                    "cpu_memory_mb_mean": 300.0,
                    "cpu_memory_mb_max": 320.0,
                },
                "system_info": {
                    "platform": "test-platform",
                    "processor": "test-cpu",
                    "cpu_count": 4,
                    "python_version": "3.8.0",
                    "torch_version": "1.8.0",
                    "torch_cuda_available": False,
                },
                "model_info": {
                    "name": "distilbert-base-uncased",
                    "hidden_size": 768,
                    "num_hidden_layers": 6,
                    "num_attention_heads": 12,
                    "intermediate_size": 3072,
                    "vocab_size": 30522,
                    "num_parameters": 66955010,
                },
            },
            {
                "timestamp": "2025-05-01T12:10:00.000000",
                "model": "distilbert-base-uncased",
                "batch_size": 8,
                "device": "cpu",
                "dataset": "glue/sst2/validation",
                "metrics": {
                    "latency_ms_mean": 200.0,
                    "latency_ms_std": 20.0,
                    "latency_ms_p90": 220.0,
                    "throughput_mean": 40.0,
                    "batch_size": 8,
                    "sequence_length": 128,
                    "num_measurements": 10,
                    "device": "cpu",
                    "warmup_runs": 2,
                    "iterations": 10,
                    "cpu_memory_mb_mean": 500.0,
                    "cpu_memory_mb_max": 550.0,
                },
                "system_info": {
                    "platform": "test-platform",
                    "processor": "test-cpu",
                    "cpu_count": 4,
                    "python_version": "3.8.0",
                    "torch_version": "1.8.0",
                    "torch_cuda_available": False,
                },
                "model_info": {
                    "name": "distilbert-base-uncased",
                    "hidden_size": 768,
                    "num_hidden_layers": 6,
                    "num_attention_heads": 12,
                    "intermediate_size": 3072,
                    "vocab_size": 30522,
                    "num_parameters": 66955010,
                },
            },
            {
                "timestamp": "2025-05-01T12:20:00.000000",
                "model": "distilbert-base-uncased",
                "batch_size": 32,
                "device": "cpu",
                "dataset": "glue/sst2/validation",
                "metrics": {
                    "latency_ms_mean": 600.0,
                    "latency_ms_std": 60.0,
                    "latency_ms_p90": 660.0,
                    "throughput_mean": 53.33,
                    "batch_size": 32,
                    "sequence_length": 128,
                    "num_measurements": 10,
                    "device": "cpu",
                    "warmup_runs": 2,
                    "iterations": 10,
                    "cpu_memory_mb_mean": 800.0,
                    "cpu_memory_mb_max": 900.0,
                },
                "system_info": {
                    "platform": "test-platform",
                    "processor": "test-cpu",
                    "cpu_count": 4,
                    "python_version": "3.8.0",
                    "torch_version": "1.8.0",
                    "torch_cuda_available": False,
                },
                "model_info": {
                    "name": "distilbert-base-uncased",
                    "hidden_size": 768,
                    "num_hidden_layers": 6,
                    "num_attention_heads": 12,
                    "intermediate_size": 3072,
                    "vocab_size": 30522,
                    "num_parameters": 66955010,
                },
            },
        ]

        # Create a sample GPU result for comparison tests
        self.sample_gpu_results = [
            {
                "timestamp": "2025-05-01T12:30:00.000000",
                "model": "distilbert-base-uncased",
                "batch_size": 32,
                "device": "cuda",
                "dataset": "glue/sst2/validation",
                "metrics": {
                    "latency_ms_mean": 60.0,
                    "latency_ms_std": 6.0,
                    "latency_ms_p90": 66.0,
                    "throughput_mean": 533.33,
                    "batch_size": 32,
                    "sequence_length": 128,
                    "num_measurements": 10,
                    "device": "cuda",
                    "warmup_runs": 2,
                    "iterations": 10,
                    "cpu_memory_mb_mean": 400.0,
                    "cpu_memory_mb_max": 450.0,
                    "gpu_memory_mb_max": 2000.0,
                    "gpu_avg_power_w": 80.0,
                },
                "system_info": {
                    "platform": "test-platform",
                    "processor": "test-cpu",
                    "cpu_count": 4,
                    "python_version": "3.8.0",
                    "torch_version": "1.8.0",
                    "torch_cuda_available": True,
                    "cuda_version": "11.1",
                    "cuda_device_count": 1,
                    "gpu_info": [{"name": "Test GPU", "total_memory_mb": 8192}],
                },
                "model_info": {
                    "name": "distilbert-base-uncased",
                    "hidden_size": 768,
                    "num_hidden_layers": 6,
                    "num_attention_heads": 12,
                    "intermediate_size": 3072,
                    "vocab_size": 30522,
                    "num_parameters": 66955010,
                },
            }
        ]

        # Write sample results to temporary files
        self.cpu_results_file = os.path.join(self.temp_dir.name, "cpu_results.jsonl")
        with open(self.cpu_results_file, "w") as f:
            for result in self.sample_results:
                f.write(json.dumps(result) + "\n")

        self.gpu_results_file = os.path.join(self.temp_dir.name, "gpu_results.jsonl")
        with open(self.gpu_results_file, "w") as f:
            for result in self.sample_gpu_results:
                f.write(json.dumps(result) + "\n")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_load_results(self):
        """Test loading benchmark results from JSONL files."""
        # Test loading CPU results
        cpu_results = plot_metrics.load_results(self.cpu_results_file)
        self.assertEqual(len(cpu_results), 3, "Should load 3 CPU result entries")

        # Test loading GPU results
        gpu_results = plot_metrics.load_results(self.gpu_results_file)
        self.assertEqual(len(gpu_results), 1, "Should load 1 GPU result entry")

    def test_results_to_dataframe(self):
        """Test converting results to DataFrame."""
        # Load results
        cpu_results = plot_metrics.load_results(self.cpu_results_file)

        # Convert to DataFrame
        df = plot_metrics.results_to_dataframe(cpu_results)

        # Check DataFrame properties
        self.assertIsInstance(df, pd.DataFrame, "Should return a pandas DataFrame")
        self.assertEqual(len(df), 3, "DataFrame should have 3 rows")
        self.assertIn(
            "batch_size", df.columns, "DataFrame should have batch_size column"
        )
        self.assertIn(
            "latency_ms_mean",
            df.columns,
            "DataFrame should have latency_ms_mean column",
        )
        self.assertIn(
            "throughput_mean",
            df.columns,
            "DataFrame should have throughput_mean column",
        )

    def test_plot_generation(self):
        """Test generating plots from benchmark results."""
        # Set up output directory
        output_dir = os.path.join(self.temp_dir.name, "figures")

        # Load results
        cpu_results = plot_metrics.load_results(self.cpu_results_file)
        df = plot_metrics.results_to_dataframe(cpu_results)

        # Generate plots
        plot_metrics.setup_plot_style()
        plot_metrics.plot_latency_vs_batch_size(df, output_dir, format="png", dpi=72)
        plot_metrics.plot_throughput_vs_batch_size(df, output_dir, format="png", dpi=72)
        plot_metrics.plot_memory_usage_vs_batch_size(
            df, output_dir, format="png", dpi=72
        )

        # Check if plot files were created
        self.assertTrue(
            os.path.exists(os.path.join(output_dir, "latency_vs_batch_size.png")),
            "Latency plot file should exist",
        )
        self.assertTrue(
            os.path.exists(os.path.join(output_dir, "throughput_vs_batch_size.png")),
            "Throughput plot file should exist",
        )
        self.assertTrue(
            os.path.exists(os.path.join(output_dir, "memory_vs_batch_size.png")),
            "Memory usage plot file should exist",
        )

    def test_plot_with_comparison(self):
        """Test generating comparison plots."""
        # Set up output directory
        output_dir = os.path.join(self.temp_dir.name, "comparison_figures")

        # Load results
        cpu_results = plot_metrics.load_results(self.cpu_results_file)
        gpu_results = plot_metrics.load_results(self.gpu_results_file)

        # Convert to DataFrames
        cpu_df = plot_metrics.results_to_dataframe(cpu_results)
        gpu_df = plot_metrics.results_to_dataframe(gpu_results)

        # Generate comparison plots
        plot_metrics.setup_plot_style()
        plot_metrics.plot_latency_vs_batch_size(
            cpu_df, output_dir, gpu_df, format="png", dpi=72
        )

        # Check if plot file was created
        self.assertTrue(
            os.path.exists(os.path.join(output_dir, "latency_vs_batch_size.png")),
            "Comparison latency plot file should exist",
        )

    def test_insights_generation(self):
        """Test generating insights from benchmark results."""
        # Load results
        cpu_results = plot_metrics.load_results(self.cpu_results_file)

        # Convert to DataFrame
        df = plot_metrics.results_to_dataframe(cpu_results)

        # Extract model and system info
        model_info = cpu_results[0]["model_info"]
        system_info = cpu_results[0]["system_info"]

        # Generate insights
        insights = generate_report.generate_insights(df, model_info, system_info)

        # Check insights
        self.assertIn("device", insights, "Insights should include device type")
        self.assertEqual(insights["device"], "cpu", "Device should be 'cpu'")

        self.assertIn(
            "optimal_batch_size", insights, "Insights should include optimal batch size"
        )
        self.assertEqual(
            insights["optimal_batch_size"], 32, "Optimal batch size should be 32"
        )

        self.assertIn(
            "min_latency_batch_size",
            insights,
            "Insights should include min latency batch size",
        )
        self.assertEqual(
            insights["min_latency_batch_size"], 1, "Min latency batch size should be 1"
        )

    def test_report_generation(self):
        """Test generating markdown report."""
        # Set up output path
        output_path = os.path.join(self.temp_dir.name, "insights.md")

        # Load results
        cpu_results = plot_metrics.load_results(self.cpu_results_file)

        # Convert to DataFrame
        df = plot_metrics.results_to_dataframe(cpu_results)

        # Extract model and system info
        model_info = cpu_results[0]["model_info"]
        system_info = cpu_results[0]["system_info"]

        # Generate insights
        insights = generate_report.generate_insights(df, model_info, system_info)

        # Generate report
        generate_report.generate_markdown_report(
            df, insights, model_info, system_info, output_path=output_path
        )

        # Check if report file was created
        self.assertTrue(os.path.exists(output_path), "Report file should exist")

        # Check report content
        with open(output_path, "r") as f:
            content = f.read()

            self.assertIn(
                "# DistilBERT Benchmark Insights Report",
                content,
                "Report should have the correct title",
            )
            self.assertIn(
                "## Model Information",
                content,
                "Report should have a model information section",
            )
            self.assertIn(
                "## Performance Insights",
                content,
                "Report should have a performance insights section",
            )


if __name__ == "__main__":
    unittest.main()
