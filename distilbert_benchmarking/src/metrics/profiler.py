"""
Profiler for detailed performance analysis of model inference.

This module provides functions to profile different components of the inference pipeline,
including data movement, forward pass, and post-processing.
"""

import time
import logging
import statistics
from typing import Dict, Any, List, Callable, Tuple, Optional
import torch

# Set up logger
logger = logging.getLogger(__name__)


def profile_forward_pass(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    iterations: int = 10,
    warmup_runs: int = 2,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Profile the model's forward pass to measure time spent in different operations.
    
    Args:
        model: The model to profile
        input_ids: Input tensor
        attention_mask: Attention mask tensor
        iterations: Number of iterations to average over
        warmup_runs: Number of warmup runs before measuring
        device: Device to run on
    
    Returns:
        Dictionary with profiling metrics
    """
    profiling_metrics = {}
    
    # Use CUDA events for GPU timing if on CUDA
    using_cuda = device.startswith("cuda") and torch.cuda.is_available()
    
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Ensure CUDA operations are complete after warmup
    if using_cuda:
        torch.cuda.synchronize()
    
    # Measure model forward pass time
    forward_pass_times = []
    for _ in range(iterations):
        # Time the core forward pass
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if using_cuda:
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        forward_pass_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Compute statistics
    profiling_metrics["forward_pass_time_ms_mean"] = statistics.mean(forward_pass_times)
    profiling_metrics["forward_pass_time_ms_median"] = statistics.median(forward_pass_times)
    profiling_metrics["forward_pass_time_ms_min"] = min(forward_pass_times)
    profiling_metrics["forward_pass_time_ms_max"] = max(forward_pass_times)
    if len(forward_pass_times) > 1:
        profiling_metrics["forward_pass_time_ms_std"] = statistics.stdev(forward_pass_times)
    else:
        profiling_metrics["forward_pass_time_ms_std"] = 0
    
    return profiling_metrics


def profile_data_movement(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    iterations: int = 10,
    to_device: str = "cpu",
) -> Dict[str, Any]:
    """
    Profile data movement costs (e.g., CPU to GPU transfers).
    
    Args:
        input_ids: Input tensor (on source device)
        attention_mask: Attention mask tensor (on source device)
        iterations: Number of iterations to average over
        to_device: Destination device for data movement
    
    Returns:
        Dictionary with data movement profiling metrics
    """
    profiling_metrics = {}
    
    # We'll time moving the tensors to the target device
    movement_times = []
    
    # Determine if device is CUDA
    using_cuda = to_device.startswith("cuda") and torch.cuda.is_available()
    
    # Get original device to move back after each iteration
    original_device = input_ids.device
    
    for _ in range(iterations):
        # Reset to original device
        input_ids_original = input_ids.to(original_device)
        attention_mask_original = attention_mask.to(original_device)
        
        # Time data movement
        start_time = time.perf_counter()
        moved_input_ids = input_ids_original.to(to_device)
        moved_attention_mask = attention_mask_original.to(to_device)
        
        # Ensure transfers are complete
        if using_cuda:
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        movement_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Compute statistics
    profiling_metrics["data_movement_time_ms_mean"] = statistics.mean(movement_times)
    profiling_metrics["data_movement_time_ms_median"] = statistics.median(movement_times)
    profiling_metrics["data_movement_time_ms_min"] = min(movement_times)
    profiling_metrics["data_movement_time_ms_max"] = max(movement_times)
    if len(movement_times) > 1:
        profiling_metrics["data_movement_time_ms_std"] = statistics.stdev(movement_times)
    else:
        profiling_metrics["data_movement_time_ms_std"] = 0
    
    return profiling_metrics


def profile_batch_preprocessing(
    batch: Dict[str, Any],
    device: str = "cpu",
    iterations: int = 10,
) -> Dict[str, Any]:
    """
    Profile batch preprocessing operations (extracting tensors, reshaping, etc.).
    
    Args:
        batch: Batch from dataloader
        device: Target device 
        iterations: Number of iterations to average over
    
    Returns:
        Dictionary with preprocessing profiling metrics
    """
    profiling_metrics = {}
    
    # Time preprocessing operations
    preprocessing_times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        
        # Extract inputs (common preprocessing pattern)
        if isinstance(batch["input_ids"], list):
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
        
        # Ensure transfers are complete
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        preprocessing_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Compute statistics
    profiling_metrics["preprocessing_time_ms_mean"] = statistics.mean(preprocessing_times)
    profiling_metrics["preprocessing_time_ms_median"] = statistics.median(preprocessing_times)
    profiling_metrics["preprocessing_time_ms_min"] = min(preprocessing_times)
    profiling_metrics["preprocessing_time_ms_max"] = max(preprocessing_times)
    if len(preprocessing_times) > 1:
        profiling_metrics["preprocessing_time_ms_std"] = statistics.stdev(preprocessing_times)
    else:
        profiling_metrics["preprocessing_time_ms_std"] = 0
    
    return profiling_metrics


def run_detailed_profiling(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    batch_idx: int = 0,
    device: str = "cpu",
    iterations: int = 10,
    warmup_runs: int = 2,
) -> Dict[str, Any]:
    """
    Run detailed profiling of the entire inference pipeline.
    
    Args:
        model: The model to profile
        dataloader: DataLoader with the dataset 
        batch_idx: Index of the batch to profile (default: 0)
        device: Device to run on
        iterations: Number of iterations to average over
        warmup_runs: Number of warmup runs before measuring
    
    Returns:
        Dictionary with detailed profiling metrics
    """
    profiling_metrics = {}
    
    # Get a batch
    batches = []
    for i, batch in enumerate(dataloader):
        if i == batch_idx:
            batches.append(batch)
            break
    
    if not batches:
        logger.error(f"Could not find batch with index {batch_idx}")
        return {"error": f"Batch index {batch_idx} not found"}
    
    batch = batches[0]
    
    # Profile batch preprocessing
    preprocessing_metrics = profile_batch_preprocessing(
        batch=batch,
        device=device,
        iterations=iterations,
    )
    profiling_metrics.update(preprocessing_metrics)
    
    # Extract tensors for forward pass profiling
    if isinstance(batch["input_ids"], list):
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
    else:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
    
    # Profile data movement
    data_movement_metrics = profile_data_movement(
        input_ids=input_ids,
        attention_mask=attention_mask,
        iterations=iterations,
        to_device=device,
    )
    profiling_metrics.update(data_movement_metrics)
    
    # Profile forward pass
    forward_pass_metrics = profile_forward_pass(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        iterations=iterations,
        warmup_runs=warmup_runs,
        device=device,
    )
    profiling_metrics.update(forward_pass_metrics)
    
    # Add batch info
    profiling_metrics["batch_size"] = input_ids.shape[0]
    profiling_metrics["sequence_length"] = input_ids.shape[1]
    
    # Calculate total time
    profiling_metrics["total_time_ms"] = (
        profiling_metrics.get("preprocessing_time_ms_mean", 0) +
        profiling_metrics.get("data_movement_time_ms_mean", 0) +
        profiling_metrics.get("forward_pass_time_ms_mean", 0)
    )
    
    # Calculate percentage of total time
    if profiling_metrics["total_time_ms"] > 0:
        profiling_metrics["preprocessing_percent"] = (
            profiling_metrics.get("preprocessing_time_ms_mean", 0) / 
            profiling_metrics["total_time_ms"] * 100
        )
        profiling_metrics["data_movement_percent"] = (
            profiling_metrics.get("data_movement_time_ms_mean", 0) / 
            profiling_metrics["total_time_ms"] * 100
        )
        profiling_metrics["forward_pass_percent"] = (
            profiling_metrics.get("forward_pass_time_ms_mean", 0) / 
            profiling_metrics["total_time_ms"] * 100
        )
    
    return profiling_metrics 