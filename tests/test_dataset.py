"""
Tests for the dataset ingestion and tokenization module.

This module tests the dataset loading, tokenization, and caching functionality
for the DistilBERT benchmarking suite.
"""

import os
import json
import pytest
from pathlib import Path

# Will import from our implementation once it exists
# from data.dataset import load_dataset, tokenize_dataset, get_dataset_info, verify_dataset_hash

# Constants for testing
TEST_CACHE_DIR = "data/test_cache"
TEST_DATASET_NAME = "glue"
TEST_DATASET_SUBSET = "sst2"
TEST_DATASET_SPLIT = "validation"


@pytest.fixture(scope="module")
def cleanup_test_cache():
    """Remove test cache before and after tests."""
    # Setup: ensure the test directory is clean
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


def test_dataset_import():
    """Test that we can import the dataset module."""
    try:
        from data.dataset import load_dataset, tokenize_dataset

        assert callable(load_dataset)
        assert callable(tokenize_dataset)
    except ImportError:
        pytest.fail(
            "Could not import dataset module. Make sure data/dataset.py exists."
        )


def test_load_dataset(cleanup_test_cache):
    """Test loading the SST-2 dataset."""
    from data.dataset import load_dataset

    # Load the dataset
    dataset = load_dataset(
        dataset_name=TEST_DATASET_NAME,
        subset=TEST_DATASET_SUBSET,
        split=TEST_DATASET_SPLIT,
        cache_dir=TEST_CACHE_DIR,
    )

    # Verify the dataset structure
    assert dataset is not None
    assert hasattr(dataset, "__len__")
    assert len(dataset) > 0

    # Check a sample
    sample = dataset[0]
    assert "sentence" in sample
    assert "label" in sample

    # Check that the dataset is cached
    cache_files = list(Path(TEST_CACHE_DIR).glob("**/dataset_info.json"))
    assert len(cache_files) > 0


def test_tokenize_dataset(cleanup_test_cache):
    """Test tokenizing the SST-2 dataset."""
    from data.dataset import load_dataset, tokenize_dataset

    # Load the dataset
    dataset = load_dataset(
        dataset_name=TEST_DATASET_NAME,
        subset=TEST_DATASET_SUBSET,
        split=TEST_DATASET_SPLIT,
        cache_dir=TEST_CACHE_DIR,
    )

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        model_name="distilbert-base-uncased",
        max_length=128,
        cache_dir=TEST_CACHE_DIR,
    )

    # Verify the tokenized dataset structure
    assert tokenized_dataset is not None
    assert hasattr(tokenized_dataset, "__len__")
    assert len(tokenized_dataset) == len(dataset)

    # Check a sample
    sample = tokenized_dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert isinstance(sample["input_ids"], list)
    assert all(isinstance(id, int) for id in sample["input_ids"])

    # Check that the tokenized dataset is cached
    cache_files = list(Path(TEST_CACHE_DIR).glob("**/tokenized_dataset.pt"))
    assert len(cache_files) > 0


def test_dataset_hash_verification(cleanup_test_cache):
    """Test hash verification for the dataset."""
    from data.dataset import load_dataset, get_dataset_info, verify_dataset_hash

    # Load the dataset
    dataset = load_dataset(
        dataset_name=TEST_DATASET_NAME,
        subset=TEST_DATASET_SUBSET,
        split=TEST_DATASET_SPLIT,
        cache_dir=TEST_CACHE_DIR,
    )

    # Get dataset info and generate hash
    info = get_dataset_info(dataset)
    assert "num_examples" in info
    assert "features" in info
    assert "hash" in info

    # Verify the hash
    assert verify_dataset_hash(dataset, info["hash"])

    # Check that manifest is saved
    manifest_path = os.path.join(TEST_CACHE_DIR, "manifest.json")
    assert os.path.exists(manifest_path)

    # Check manifest content
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert TEST_DATASET_NAME in manifest
    assert TEST_DATASET_SUBSET in manifest[TEST_DATASET_NAME]
    assert "hash" in manifest[TEST_DATASET_NAME][TEST_DATASET_SUBSET]


def test_dataset_offline_mode(cleanup_test_cache):
    """Test that we can use pre-cached dataset in offline mode."""
    from data.dataset import load_dataset

    # First, load the dataset to cache it
    dataset = load_dataset(
        dataset_name=TEST_DATASET_NAME,
        subset=TEST_DATASET_SUBSET,
        split=TEST_DATASET_SPLIT,
        cache_dir=TEST_CACHE_DIR,
    )

    # Now, try to load it in offline mode
    offline_dataset = load_dataset(
        dataset_name=TEST_DATASET_NAME,
        subset=TEST_DATASET_SUBSET,
        split=TEST_DATASET_SPLIT,
        cache_dir=TEST_CACHE_DIR,
        offline=True,
    )

    # Verify the dataset
    assert offline_dataset is not None
    assert len(offline_dataset) == len(dataset)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
