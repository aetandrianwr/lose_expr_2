"""
Unit tests for reproducibility utilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import random
from utils.reproducibility import set_seed


def test_seed_reproducibility():
    """Test that seed setting produces reproducible results."""
    
    def generate_random_numbers(seed):
        set_seed(seed, cuda_deterministic=True)
        
        # Python random
        py_rand = random.random()
        
        # NumPy random
        np_rand = np.random.rand()
        
        # PyTorch random
        torch_rand = torch.rand(1).item()
        
        return py_rand, np_rand, torch_rand
    
    # Generate with seed 42
    results1 = generate_random_numbers(42)
    
    # Generate again with different seed
    _ = generate_random_numbers(123)
    
    # Generate again with seed 42
    results2 = generate_random_numbers(42)
    
    # Check reproducibility
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        assert abs(r1 - r2) < 1e-9, f"Results differ at index {i}: {r1} vs {r2}"
    
    print("✓ Seed reproducibility test passed")


def test_cuda_operations_reproducibility():
    """Test reproducibility of CUDA operations if available."""
    if not torch.cuda.is_available():
        print("⊘ CUDA not available, skipping CUDA reproducibility test")
        return
    
    def run_cuda_operation(seed):
        set_seed(seed, cuda_deterministic=True)
        
        # Create tensors
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        
        # Matrix multiplication
        z = torch.matmul(x, y)
        
        return z.cpu()
    
    # Run twice with same seed
    result1 = run_cuda_operation(42)
    set_seed(123, cuda_deterministic=True)  # Different seed
    result2 = run_cuda_operation(42)
    
    # Check reproducibility
    assert torch.allclose(result1, result2, atol=1e-6), "CUDA operations not reproducible"
    
    print("✓ CUDA operations reproducibility test passed")


if __name__ == '__main__':
    print("Running reproducibility tests...")
    test_seed_reproducibility()
    test_cuda_operations_reproducibility()
    print("\n✓ All tests passed!")
