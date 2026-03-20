"""
Debug and introspection utilities for biopb services.
"""

import functools
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import threading


@dataclass
class ServiceStats:
    """Thread-safe request statistics tracker."""

    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(self, latency_ms: float, error: bool = False):
        """Record a request with its latency."""
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += latency_ms
            if error:
                self.error_count += 1

    def get_stats(self) -> dict:
        """Get current statistics."""
        with self._lock:
            avg_latency = (
                self.total_latency_ms / self.request_count
                if self.request_count > 0
                else 0.0
            )
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "total_latency_ms": round(self.total_latency_ms, 2),
                "avg_latency_ms": round(avg_latency, 2),
            }

    def reset(self):
        """Reset statistics."""
        with self._lock:
            self.request_count = 0
            self.error_count = 0
            self.total_latency_ms = 0.0


# Global stats instance
_global_stats = ServiceStats()


def get_stats() -> ServiceStats:
    """Get the global service statistics tracker."""
    return _global_stats


def get_gpu_memory_info() -> Optional[dict]:
    """
    Get GPU memory information if CUDA is available.

    Returns:
        dict with memory info or None if CUDA not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        return {
            "device": torch.cuda.get_device_name(device),
            "total_mb": round(total / 1024 / 1024, 2),
            "allocated_mb": round(allocated / 1024 / 1024, 2),
            "reserved_mb": round(reserved / 1024 / 1024, 2),
            "free_mb": round((total - allocated) / 1024 / 1024, 2),
        }
    except ImportError:
        return None
    except Exception:
        return None


def get_system_info() -> dict:
    """
    Get system information including CPU, memory, and platform details.

    Returns:
        dict with system information.
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # Try to get memory info
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]
                    meminfo[key] = int(value)
            if "MemTotal" in meminfo:
                info["memory_total_mb"] = round(meminfo["MemTotal"] / 1024, 2)
            if "MemAvailable" in meminfo:
                info["memory_available_mb"] = round(meminfo["MemAvailable"] / 1024, 2)
    except (FileNotFoundError, PermissionError):
        pass

    # Add GPU info if available
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        info["gpu"] = gpu_info

    return info


def get_model_info(model: Any) -> dict:
    """
    Extract model metadata.

    Args:
        model: The model instance

    Returns:
        dict with model information.
    """
    info = {"model_type": type(model).__name__}

    # Try common attributes
    for attr in ["model_type", "name", "version", "__version__"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, str):
                info[attr] = val

    # Try to get parameter count for torch models
    try:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            info["total_params"] = total_params
            info["trainable_params"] = trainable_params
    except ImportError:
        pass

    return info


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.

    Logs memory delta before and after function execution.
    Works with PyTorch CUDA memory tracking.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        gpu_before = get_gpu_memory_info()
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            error = None
        except Exception as e:
            error = e
            result = None

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        gpu_after = get_gpu_memory_info()

        # Log memory delta
        if gpu_before and gpu_after:
            delta = gpu_after["allocated_mb"] - gpu_before["allocated_mb"]
            print(
                f"[PROFILE] {func.__name__}: "
                f"{elapsed_ms:.2f}ms, GPU memory delta: {delta:+.2f}MB"
            )
        else:
            print(f"[PROFILE] {func.__name__}: {elapsed_ms:.2f}ms")

        if error:
            raise error

        return result

    return wrapper


def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution and record in stats.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        error = False

        try:
            result = func(*args, **kwargs)
        except Exception:
            error = True
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _global_stats.record_request(elapsed_ms, error)

        return result

    return wrapper