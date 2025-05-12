#!/usr/bin/env python3
"""
Smoke test for the DistilBERT Benchmarking Suite.

This script validates that the environment is correctly set up and all required
dependencies are installed and functional.
"""

import os
import sys
import platform
import importlib
import argparse
from typing import Optional


def print_header(message: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)


def check_dependency(package_name: str, min_version: Optional[str] = None) -> bool:
    """Check if a Python package is installed and meets minimum version requirements."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")

        if min_version and version != "unknown":
            import pkg_resources

            if pkg_resources.parse_version(version) < pkg_resources.parse_version(
                min_version
            ):
                print(
                    f"❌ {package_name} version {version} is installed, but {min_version}+ is required"
                )
                return False

        print(f"✅ {package_name} {version}")
        return True

    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False


def check_system_info():
    """Print system information."""
    print_header("System Information")
    print(f"Python version: {platform.python_version()}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")

    # Check for CUDA
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed - CUDA info unavailable")


def check_dependencies():
    """Check that all required dependencies are available."""
    print_header("Dependency Check")

    dependencies = {
        "torch": "2.0.0",
        "transformers": "4.30.0",
        "datasets": "2.13.0",
        "pydantic": "2.0.0",
        "psutil": "5.9.0",
        "numpy": "1.20.0",
        "pandas": "2.0.0",
        "matplotlib": "3.7.0",
        "pyyaml": "6.0",
    }

    optional_dependencies = {
        "pynvml": "11.5.0",  # For GPU memory monitoring
        "pyRAPL": "0.2.3",  # For energy measurements
        "pytest": "7.3.0",  # For running tests
        "pandoc": "2.3",  # For report generation
    }

    # Check required dependencies
    missing_required = 0
    for package, min_version in dependencies.items():
        if not check_dependency(package, min_version):
            missing_required += 1

    # Check optional dependencies
    print("\nOptional Dependencies:")
    missing_optional = 0
    for package, min_version in optional_dependencies.items():
        if not check_dependency(package, min_version):
            missing_optional += 1

    return missing_required, missing_optional


def check_model_loading():
    """Test loading a small DistilBERT model."""
    print_header("Model Loading Test")
    try:
        # Import locally to avoid dependency issues
        from transformers import (
            DistilBertTokenizer,
            DistilBertForSequenceClassification,
        )

        # Use a very small model for quick testing
        model_name = "distilbert-base-uncased"
        print(f"Loading model: {model_name}")

        # Load tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name)

        # Tokenize sample text
        text = "This is a smoke test for the DistilBERT benchmarking suite."
        inputs = tokenizer(text, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            model(**inputs)

        print("✅ Successfully loaded model and ran inference")
        return True

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def check_project_structure():
    """Check that the expected project directories and files exist."""
    print_header("Project Structure Check")

    expected_directories = [
        "data",
        "src",
        "src/metrics",
        "cluster",
        "analysis",
    ]

    expected_files = [
        "environment.yml",
        "Makefile",
        "config.yaml",
        "src/model.py",
        "src/runner.py",
        "src/metrics/__init__.py",
        "src/metrics/latency.py",
        "src/metrics/memory.py",
        "src/metrics/energy.py",
        "cluster/bench_distilbert.slurm",
    ]

    # Check directories
    print("Checking directories:")
    missing_dirs = 0
    for directory in expected_directories:
        if os.path.isdir(directory):
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} (missing)")
            missing_dirs += 1

    # Check files
    print("\nChecking files:")
    missing_files = 0
    for file in expected_files:
        if os.path.isfile(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing_files += 1

    return missing_dirs, missing_files


def main():
    """Run the smoke test."""
    parser = argparse.ArgumentParser(
        description="DistilBERT Benchmarking Suite Smoke Test"
    )
    parser.add_argument(
        "--skip-model", action="store_true", help="Skip model loading test"
    )
    args = parser.parse_args()

    print("DistilBERT Benchmarking Suite - Smoke Test")
    print("------------------------------------------")

    # Check system information
    check_system_info()

    # Check dependencies
    missing_required, missing_optional = check_dependencies()

    # Check project structure
    missing_dirs, missing_files = check_project_structure()

    # Check model loading
    model_test_passed = True
    if not args.skip_model:
        model_test_passed = check_model_loading()
    else:
        print_header("Model Loading Test")
        print("Skipped")

    # Print summary
    print_header("Summary")
    if missing_required > 0:
        print(f"❌ {missing_required} required dependencies missing")
    else:
        print("✅ All required dependencies installed")

    if missing_optional > 0:
        print(f"⚠️  {missing_optional} optional dependencies missing")
    else:
        print("✅ All optional dependencies installed")

    if missing_dirs > 0 or missing_files > 0:
        print(
            f"❌ Project structure incomplete ({missing_dirs} directories and {missing_files} files missing)"
        )
    else:
        print("✅ Project structure complete")

    if not model_test_passed and not args.skip_model:
        print("❌ Model loading test failed")
    elif not args.skip_model:
        print("✅ Model loading test passed")

    # Overall status
    if (
        missing_required > 0
        or (missing_dirs > 0 or missing_files > 0)
        or (not model_test_passed and not args.skip_model)
    ):
        print("\n❌ Smoke test FAILED")
        return 1
    else:
        print("\n✅ Smoke test PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
