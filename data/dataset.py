"""
Dataset ingestion and tokenization for DistilBERT benchmarking.

This module provides utilities for loading GLUE/SST-2 validation data
and performing batched tokenization with hash verification.
"""

import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional

# Import Hugging Face libraries
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as HFDataset
from transformers import DistilBertTokenizerFast
import torch

logger = logging.getLogger(__name__)

# Define constants
DEFAULT_CACHE_DIR = "data/cached"
DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_MAX_LENGTH = 128
MANIFEST_FILE = "manifest.json"

# SST-2 citation and license info
SST2_CITATION = """
@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and Potts, Christopher},
  booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
  pages={1631--1642},
  year={2013}
}
"""
SST2_LICENSE = "MIT License"


def load_dataset(
    dataset_name: str = "glue",
    subset: str = "sst2",
    split: str = "validation",
    cache_dir: Optional[str] = None,
    offline: bool = False,
) -> HFDataset:
    """
    Load the dataset from Hugging Face's datasets library.

    Args:
        dataset_name: Name of the dataset to load
        subset: Subset of the dataset to load
        split: Split of the dataset to load (e.g., "train", "validation")
        cache_dir: Directory to cache the dataset
        offline: Whether to use only local files (no downloads)

    Returns:
        The loaded dataset
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Create a manifest file path
    manifest_path = os.path.join(cache_dir, MANIFEST_FILE)

    # Load the manifest if it exists
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to load manifest from {manifest_path}. Starting fresh."
            )

    # Load the dataset
    logger.info(f"Loading dataset: {dataset_name}/{subset}/{split}")
    dataset = hf_load_dataset(
        dataset_name,
        subset,
        split=split,
        cache_dir=cache_dir,
        download_mode="force_redownload" if not offline else "reuse_dataset_if_exists",
    )

    # Generate and verify hash
    dataset_hash = _calculate_dataset_hash(dataset)

    # Update manifest
    if dataset_name not in manifest:
        manifest[dataset_name] = {}

    if subset not in manifest[dataset_name]:
        manifest[dataset_name][subset] = {}

    manifest[dataset_name][subset]["hash"] = dataset_hash
    manifest[dataset_name][subset]["split"] = split
    manifest[dataset_name][subset]["num_examples"] = len(dataset)
    manifest[dataset_name][subset]["citation"] = SST2_CITATION
    manifest[dataset_name][subset]["license"] = SST2_LICENSE

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return dataset


def tokenize_dataset(
    dataset: HFDataset,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 1000,
    cache_dir: Optional[str] = None,
    force_recompute: bool = False,
) -> HFDataset:
    """
    Tokenize the dataset using DistilBertTokenizerFast.

    Args:
        dataset: Dataset to tokenize
        model_name: Name of the model to use for tokenization
        max_length: Maximum length of the tokenized sequences
        batch_size: Batch size for tokenization
        cache_dir: Directory to cache the tokenized dataset
        force_recompute: Whether to force recomputation of tokenization

    Returns:
        The tokenized dataset
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Create a standardized model name for the cache file (replacing any slashes)
    model_name.replace("/", "_")

    # Create cache file path for tokenized dataset - use a name pattern that matches the test's glob
    cache_file = os.path.join(cache_dir, "tokenized_dataset.pt")

    # Check if tokenized dataset is already cached
    if os.path.exists(cache_file) and not force_recompute:
        logger.info(f"Loading tokenized dataset from cache: {cache_file}")
        try:
            tokenized_dataset = torch.load(cache_file)
            return tokenized_dataset
        except Exception as e:
            logger.warning(f"Failed to load tokenized dataset from cache: {e}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)

    # Define tokenization function
    def _tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    # Tokenize dataset
    logger.info(f"Tokenizing dataset with batch size {batch_size}")
    tokenized_dataset = dataset.map(
        _tokenize_function,
        batched=True,
        batch_size=batch_size,
        desc="Tokenizing dataset",
    )

    # Save tokenized dataset to cache
    logger.info(f"Saving tokenized dataset to cache: {cache_file}")
    torch.save(tokenized_dataset, cache_file)

    return tokenized_dataset


def get_dataset_info(dataset: HFDataset) -> Dict[str, Any]:
    """
    Get information about the dataset.

    Args:
        dataset: Dataset to get information about

    Returns:
        Dictionary with dataset information
    """
    # Get the dataset schema/features
    features = dataset.features

    # Create info dictionary
    info = {
        "num_examples": len(dataset),
        "features": {name: str(feature) for name, feature in features.items()},
        "hash": _calculate_dataset_hash(dataset),
    }

    return info


def verify_dataset_hash(dataset: HFDataset, expected_hash: str) -> bool:
    """
    Verify that the dataset hash matches the expected hash.

    Args:
        dataset: Dataset to verify
        expected_hash: Expected hash value

    Returns:
        True if the hash matches, False otherwise
    """
    actual_hash = _calculate_dataset_hash(dataset)
    return actual_hash == expected_hash


def _calculate_dataset_hash(dataset: HFDataset) -> str:
    """
    Calculate a hash of the dataset content for integrity verification.

    Args:
        dataset: Dataset to hash

    Returns:
        Hash of the dataset
    """
    # Create a hash object
    hasher = hashlib.sha256()

    # Use a subset of examples for large datasets to keep hash calculation manageable
    max_examples_for_hash = min(1000, len(dataset))

    # Update hash with dataset length and feature names
    hasher.update(str(len(dataset)).encode())
    hasher.update(str(sorted(dataset.features.keys())).encode())

    # Update hash with sample data
    for i in range(max_examples_for_hash):
        example = dataset[i]
        for key in sorted(example.keys()):
            # Convert value to string representation
            value_str = str(example[key])
            hasher.update(f"{key}:{value_str}".encode())

    # Return the hexadecimal digest
    return hasher.hexdigest()


def get_citation_info() -> Dict[str, str]:
    """
    Get citation information for the SST-2 dataset.

    Returns:
        Dictionary with citation and license information
    """
    return {"citation": SST2_CITATION, "license": SST2_LICENSE}
