"""
Tests for the device-agnostic model loader module.

This module tests the DistilBERT model loading functionality on different devices
(CPU and CUDA if available) for the DistilBERT benchmarking suite.
"""

import os
import pytest
import torch
from pathlib import Path

# Constants for testing
TEST_CACHE_DIR = "data/test_cache_model"
TEST_MODEL_NAME = "distilbert-base-uncased"


@pytest.fixture(scope="module")
def cleanup_test_cache():
    """Remove test cache before and after tests."""
    # Setup: ensure test directory is clean
    if os.path.exists(TEST_CACHE_DIR):
        import shutil

        shutil.rmtree(TEST_CACHE_DIR)
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)

    # Run tests
    yield

    # Teardown: clean up after tests
    if os.path.exists(TEST_CACHE_DIR):
        import shutil

        shutil.rmtree(TEST_CACHE_DIR)


def test_model_import():
    """Test that we can import the model module."""
    try:
        from src.model import load_model, get_model_metadata

        assert callable(load_model)
        assert callable(get_model_metadata)
    except ImportError:
        pytest.fail("Could not import model module. Make sure src/model.py exists.")


def test_load_model_cpu(cleanup_test_cache):
    """Test loading the DistilBERT model on CPU."""
    from src.model import load_model

    # Load the model on CPU
    model = load_model(
        model_name=TEST_MODEL_NAME, device="cpu", cache_dir=TEST_CACHE_DIR
    )

    # Verify the model
    assert model is not None
    assert hasattr(model, "config")
    assert model.device.type == "cpu"

    # Verify eval mode
    assert not model.training

    # Verify model file is cached
    cache_files = list(Path(TEST_CACHE_DIR).glob("**/*.bin"))
    assert len(cache_files) > 0


def test_load_model_cuda():
    """Test loading the DistilBERT model on CUDA if available."""
    from src.model import load_model

    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available. Skipping GPU test.")

    # Load the model on CUDA
    model = load_model(
        model_name=TEST_MODEL_NAME, device="cuda", cache_dir=TEST_CACHE_DIR
    )

    # Verify the model
    assert model is not None
    assert hasattr(model, "config")
    assert model.device.type == "cuda"

    # Verify eval mode
    assert not model.training


def test_model_inference(cleanup_test_cache):
    """Test running inference with the loaded model."""
    from src.model import load_model, load_tokenizer

    # Load model and tokenizer
    model = load_model(
        model_name=TEST_MODEL_NAME, device="cpu", cache_dir=TEST_CACHE_DIR
    )

    tokenizer = load_tokenizer(model_name=TEST_MODEL_NAME, cache_dir=TEST_CACHE_DIR)

    # Prepare input
    text = "This is a test."
    inputs = tokenizer(text, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Verify outputs
    assert "logits" in outputs
    assert outputs.logits.shape[0] == 1  # Batch size 1
    assert (
        outputs.logits.shape[1] > 0
    )  # At least 1 class (may not be exactly 2 for base model)


def test_model_metadata(cleanup_test_cache):
    """Test getting model metadata."""
    from src.model import get_model_metadata

    # Get model metadata
    metadata = get_model_metadata(model_name=TEST_MODEL_NAME, cache_dir=TEST_CACHE_DIR)

    # Verify metadata
    assert "name" in metadata
    assert "hidden_size" in metadata
    assert "num_hidden_layers" in metadata
    assert "num_attention_heads" in metadata
    assert "vocab_size" in metadata
    assert "num_parameters" in metadata


def test_model_fallback():
    """Test model fallback to CPU when CUDA is requested but not available."""
    from src.model import load_model
    from unittest.mock import patch

    # Mock torch.cuda.is_available to return False
    with patch("torch.cuda.is_available", return_value=False):
        # Request CUDA but should fallback to CPU
        model = load_model(
            model_name=TEST_MODEL_NAME, device="cuda", cache_dir=TEST_CACHE_DIR
        )

        # Verify fallback to CPU
        assert model.device.type == "cpu"


def test_model_compile():
    """Test model compilation with torch.compile if available."""
    from src.model import load_model
    import torch

    # Skip if torch.compile is not available (PyTorch < 2.0)
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available. Skipping compile test.")

    # Load model with compile enabled
    model = load_model(
        model_name=TEST_MODEL_NAME,
        device="cpu",
        use_compile=True,
        compile_options={"backend": "inductor"},
        cache_dir=TEST_CACHE_DIR,
    )

    # This just verifies that compilation didn't crash
    # Can't easily check if a model is compiled
    assert model is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
