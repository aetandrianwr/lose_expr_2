"""
Reproducibility utilities for ensuring deterministic behavior across runs.

This module provides functions to set random seeds for all relevant libraries
and configure PyTorch for deterministic execution.
"""

import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int, cuda_deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        cuda_deterministic: If True, sets CUDA operations to be deterministic.
                          This may reduce performance but ensures reproducibility.
    
    Note:
        Setting cuda_deterministic=True may impact performance. For production
        training where slight variations are acceptable, consider setting it to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
    # Set CUDA deterministic behavior
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    else:
        # Enable cudnn auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info(f"Random seed set to {seed} (deterministic={cuda_deterministic})")


def get_device(device_name: str = "cuda") -> torch.device:
    """
    Get PyTorch device with fallback logic.
    
    Args:
        device_name: Requested device ("cuda", "cpu", "mps")
        
    Returns:
        torch.device: Available device
    """
    if device_name == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA requested but not available. Using CPU.")
    elif device_name == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS device")
        else:
            device = torch.device("cpu")
            logger.warning("MPS requested but not available. Using CPU.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def log_system_info():
    """Log system and library version information."""
    logger.info("=" * 50)
    logger.info("System Information:")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info("=" * 50)
