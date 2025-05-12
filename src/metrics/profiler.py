"""
Performance profiling for model inference.

Tools to measure where time is spent during inference - data movement, 
forward pass, preprocessing, etc. Helpful for debugging slow models.
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
    Time how long the model's forward pass takes.
    
    Args:
        model: Model to profile
        input_ids: Input tensor
        attention_mask: Attention mask tensor
        iterations: How many runs to average over
        warmup_runs: Warmup iterations (not timed)
        device: Where to run this thing
    
    Returns:
        Dict with timing stats (mean, median, etc.)
    """
    results = {}
    
    # Use CUDA events for GPU timing if we're on GPU
    using_cuda = device.startswith("cuda") and torch.cuda.is_available()
    
    # Run a few times to warm up the cache, JIT, etc.
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Make sure CUDA's actually done
    if using_cuda:
        torch.cuda.synchronize()
    
    # Now let's measure for real
    times = []
    for _ in range(iterations):
        # Time the forward pass
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if using_cuda:
            torch.cuda.synchronize()  # Wait for GPU to finish
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # ms
    
    # Calculate stats
    results["forward_pass_time_ms_mean"] = statistics.mean(times)
    results["forward_pass_time_ms_median"] = statistics.median(times)
    results["forward_pass_time_ms_min"] = min(times)
    results["forward_pass_time_ms_max"] = max(times)
    if len(times) > 1:
        results["forward_pass_time_ms_std"] = statistics.stdev(times)
    else:
        results["forward_pass_time_ms_std"] = 0
    
    return results


def profile_data_movement(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    iterations: int = 10,
    to_device: str = "cpu",
) -> Dict[str, Any]:
    """
    Measure how long it takes to move data between devices (CPU→GPU or GPU→CPU).
    
    Args:
        input_ids: Input tensor on source device
        attention_mask: Attention mask on source device
        iterations: How many times to test it
        to_device: Where we're copying to
    
    Returns:
        Dict with timing stats
    """
    results = {}
    
    # Track how long each transfer takes
    times = []
    
    # Check if we're going to/from GPU
    using_cuda = to_device.startswith("cuda") and torch.cuda.is_available()
    
    # Remember where the tensors started
    original_device = input_ids.device
    
    for _ in range(iterations):
        # Start from original location each time
        input_ids_original = input_ids.to(original_device)
        attention_mask_original = attention_mask.to(original_device)
        
        # Time the transfer
        start_time = time.perf_counter()
        moved_input_ids = input_ids_original.to(to_device)
        moved_attention_mask = attention_mask_original.to(to_device)
        
        # Make sure transfer is complete
        if using_cuda:
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # ms
    
    # Calculate stats
    results["data_movement_time_ms_mean"] = statistics.mean(times)
    results["data_movement_time_ms_median"] = statistics.median(times)
    results["data_movement_time_ms_min"] = min(times)
    results["data_movement_time_ms_max"] = max(times)
    if len(times) > 1:
        results["data_movement_time_ms_std"] = statistics.stdev(times)
    else:
        results["data_movement_time_ms_std"] = 0
    
    return results


def profile_batch_preprocessing(
    batch: Dict[str, Any],
    device: str = "cpu",
    iterations: int = 10,
) -> Dict[str, Any]:
    """
    Measure preprocessing time (extracting tensors, reshaping, etc.).
    
    Args:
        batch: Batch from dataloader
        device: Where to put the processed data
        iterations: How many times to test
    
    Returns:
        Dict with timing stats
    """
    results = {}
    
    # Measure preprocessing time
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        
        # Do typical preprocessing steps
        if isinstance(batch["input_ids"], list):
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
        
        # Ensure everything's finished
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # ms
    
    # Calculate stats
    results["preprocessing_time_ms_mean"] = statistics.mean(times)
    results["preprocessing_time_ms_median"] = statistics.median(times)
    results["preprocessing_time_ms_min"] = min(times)
    results["preprocessing_time_ms_max"] = max(times)
    if len(times) > 1:
        results["preprocessing_time_ms_std"] = statistics.stdev(times)
    else:
        results["preprocessing_time_ms_std"] = 0
    
    return results


def run_detailed_profiling(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    batch_idx: int = 0,
    device: str = "cpu",
    iterations: int = 10,
    warmup_runs: int = 2,
) -> Dict[str, Any]:
    """
    Profile the whole inference pipeline from end to end.
    
    Args:
        model: Model to profile
        dataloader: Data source
        batch_idx: Which batch to use (defaults to first one)
        device: Where to run the model
        iterations: How many test runs
        warmup_runs: How many warmup runs
    
    Returns:
        Dict with all profiling results
    """
    all_results = {}
    
    # Put model on the right device
    model.eval()
    model.to(device)
    
    # Grab a batch for testing
    for i, batch in enumerate(dataloader):
        if i == batch_idx:
            break
    else:
        # If we didn't break, dataloader was empty
        raise ValueError("Dataloader doesn't have enough batches")
    
    # Check what kind of batch we got
    if isinstance(batch, dict) and "input_ids" in batch and "attention_mask" in batch:
        # HF-style batch - no need to extract
        pass
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        # Tuple/list batch - extract inputs and reshape as needed
        batch = {
            "input_ids": batch[0],
            "attention_mask": batch[1] if len(batch) > 1 else None
        }
    else:
        raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    # 1. Profile preprocessing if needed
    all_results["preprocessing"] = profile_batch_preprocessing(
        batch, device=device, iterations=iterations
    )
    
    # Get the data in the right format
    if isinstance(batch["input_ids"], list):
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device) if batch["attention_mask"] is not None else None
    else:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device) if batch["attention_mask"] is not None else None
    
    # 2. Profile data movement (most relevant for GPU)
    if device != "cpu" and input_ids.device.type == "cpu":
        all_results["data_movement"] = profile_data_movement(
            input_ids, attention_mask, iterations=iterations, to_device=device
        )
    
    # 3. Profile model forward pass
    all_results["forward_pass"] = profile_forward_pass(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        iterations=iterations,
        warmup_runs=warmup_runs,
        device=device,
    )
    
    # 4. Calculate total time
    if "data_movement" in all_results:
        total_time = (
            all_results["preprocessing"]["preprocessing_time_ms_mean"] + 
            all_results["data_movement"]["data_movement_time_ms_mean"] + 
            all_results["forward_pass"]["forward_pass_time_ms_mean"]
        )
    else:
        total_time = (
            all_results["preprocessing"]["preprocessing_time_ms_mean"] + 
            all_results["forward_pass"]["forward_pass_time_ms_mean"]
        )
    all_results["total_time_ms"] = total_time
    
    # 5. Calculate percentages for pie charts
    all_results["preprocessing_pct"] = all_results["preprocessing"]["preprocessing_time_ms_mean"] / total_time * 100
    if "data_movement" in all_results:
        all_results["data_movement_pct"] = all_results["data_movement"]["data_movement_time_ms_mean"] / total_time * 100
    all_results["forward_pass_pct"] = all_results["forward_pass"]["forward_pass_time_ms_mean"] / total_time * 100
    
    return all_results


def profile_model_with_shape(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    seq_length: int = 128,
    iterations: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Quick way to profile a model with dummy inputs of a specific shape.
    
    Args:
        model: The model to test
        input_shape: Shape of a single input (no batch dim)
        batch_size: How many samples in the batch
        seq_length: Sequence length for NLP models
        iterations: How many runs to average
        device: Where to run it
    
    Returns:
        Dict with profiling results
    """
    # For transformer models
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        # Create dummy input
        input_ids = torch.randint(
            0, model.config.vocab_size, (batch_size, seq_length)
        ).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # Time the forward pass
        results = profile_forward_pass(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            iterations=iterations,
            device=device
        )
    else:
        # For generic models, just use the input_shape
        # Create batch dimension
        batch_input_shape = (batch_size,) + input_shape
        dummy_input = torch.randn(batch_input_shape).to(device)
        
        # Time the forward pass
        times = []
        model.eval()
        model.to(device)
        
        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = model(dummy_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                
        # Measure
        for _ in range(iterations):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # ms
        
        # Stats
        results = {
            "forward_pass_time_ms_mean": statistics.mean(times),
            "forward_pass_time_ms_median": statistics.median(times),
            "forward_pass_time_ms_min": min(times),
            "forward_pass_time_ms_max": max(times),
        }
        if len(times) > 1:
            results["forward_pass_time_ms_std"] = statistics.stdev(times)
        else:
            results["forward_pass_time_ms_std"] = 0
    
    # Add batch info
    results["batch_size"] = batch_size
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        results["seq_length"] = seq_length
    results["iterations"] = iterations
    results["device"] = device
    
    return results


def estimate_throughput(times_ms: List[float], batch_size: int) -> Dict[str, float]:
    """
    Calculate throughput in samples/second from timing results.
    
    Args:
        times_ms: List of timing measurements in ms
        batch_size: Batch size used
    
    Returns:
        Dict with throughput stats
    """
    # Convert ms to seconds and calculate samples/sec
    throughputs = [batch_size / (t / 1000) for t in times_ms]
    
    return {
        "throughput_mean": statistics.mean(throughputs),
        "throughput_median": statistics.median(throughputs),
        "throughput_min": min(throughputs),
        "throughput_max": max(throughputs),
        "batch_size": batch_size,
    }


def get_throughput_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    iterations: int = 10,
) -> Dict[str, Any]:
    """
    Measure model throughput with real data.
    
    Args:
        model: Model to test
        dataloader: Source of data batches
        device: Where to run it
        iterations: How many batches to test
    
    Returns:
        Dict with throughput metrics
    """
    model.eval()
    model.to(device)
    
    times = []
    batch_sizes = []
    
    # Use at most 'iterations' batches
    for i, batch in enumerate(dataloader):
        if i >= iterations:
            break
            
        # Handle different batch formats
        if isinstance(batch, dict) and "input_ids" in batch and "attention_mask" in batch:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.size(0)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            batch_size = input_ids.size(0)
        else:
            logger.warning(f"Skipping batch with unsupported format: {type(batch)}")
            continue
            
        # Time the forward pass
        torch.cuda.synchronize() if device.startswith("cuda") else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
        torch.cuda.synchronize() if device.startswith("cuda") else None
        end_time = time.perf_counter()
        
        # Convert to ms
        elapsed_time_ms = (end_time - start_time) * 1000
        times.append(elapsed_time_ms)
        batch_sizes.append(batch_size)
        
    # Calculate throughput for each batch
    batch_throughputs = []
    for time_ms, batch_size in zip(times, batch_sizes):
        batch_throughputs.append(batch_size / (time_ms / 1000))  # samples/sec
    
    # Overall stats
    avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
    
    return {
        "throughput_mean": statistics.mean(batch_throughputs) if batch_throughputs else 0,
        "throughput_min": min(batch_throughputs) if batch_throughputs else 0,
        "throughput_max": max(batch_throughputs) if batch_throughputs else 0,
        "latency_ms_mean": statistics.mean(times) if times else 0,
        "latency_ms_min": min(times) if times else 0,
        "latency_ms_max": max(times) if times else 0,
        "avg_batch_size": avg_batch_size,
        "num_batches": len(batch_sizes),
    } 