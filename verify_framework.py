#!/usr/bin/env python3
"""
Quick test script to verify the production framework.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data.dataset_v2 import create_dataloaders, TrajectoryDataset
        print("✓ data.dataset_v2")
        
        from utils.config import load_config, validate_config
        print("✓ utils.config")
        
        from utils.reproducibility import set_seed, get_device
        print("✓ utils.reproducibility")
        
        from utils.experiment_tracker import ExperimentTracker
        print("✓ utils.experiment_tracker")
        
        from utils.metrics_v2 import calculate_metrics
        print("✓ utils.metrics_v2")
        
        from models.temporal_fusion import TemporalFusionModel
        print("✓ models.temporal_fusion")
        
        print("\n✓ All imports successful!")
        return True
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from utils.config import load_config, validate_config
        
        # Load default config
        config = load_config('configs/default.yaml')
        print(f"✓ Loaded config with {len(config)} top-level keys")
        
        # Validate
        validate_config(config)
        print("✓ Configuration validation passed")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reproducibility():
    """Test reproducibility utilities."""
    print("\nTesting reproducibility...")
    
    try:
        import torch
        from utils.reproducibility import set_seed, get_device
        
        # Set seed
        set_seed(42, cuda_deterministic=False)
        print("✓ Seed set successfully")
        
        # Get device
        device = get_device("cuda")
        print(f"✓ Device: {device}")
        
        return True
    except Exception as e:
        print(f"✗ Reproducibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading (if data exists)."""
    print("\nTesting dataset loading...")
    
    try:
        from data.dataset_v2 import get_dataset_paths
        
        # Check if geolife dataset exists
        try:
            train_path, val_path, test_path = get_dataset_paths("geolife", "data")
            print(f"✓ Found Geolife dataset")
            print(f"  Train: {Path(train_path).stat().st_size / 1024 / 1024:.1f} MB")
            print(f"  Val: {Path(val_path).stat().st_size / 1024 / 1024:.1f} MB")
            print(f"  Test: {Path(test_path).stat().st_size / 1024 / 1024:.1f} MB")
        except FileNotFoundError:
            print("⊘ Geolife dataset not found")
        
        # Check if DIY dataset exists
        try:
            train_path, val_path, test_path = get_dataset_paths("diy", "data")
            print(f"✓ Found DIY dataset")
            print(f"  Train: {Path(train_path).stat().st_size / 1024 / 1024:.1f} MB")
            print(f"  Val: {Path(val_path).stat().st_size / 1024 / 1024:.1f} MB")
            print(f"  Test: {Path(test_path).stat().st_size / 1024 / 1024:.1f} MB")
        except FileNotFoundError:
            print("⊘ DIY dataset not found")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics...")
    
    try:
        import torch
        from utils.metrics_v2 import calculate_metrics
        
        # Create dummy data
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        # Calculate metrics
        metrics = calculate_metrics(logits, targets)
        
        print(f"✓ Calculated {len(metrics)} metrics")
        for key in ['accuracy', 'top5_accuracy', 'mrr', 'precision', 'recall', 'f1_score']:
            print(f"  {key}: {metrics[key]:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Production Framework Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config_loading()))
    results.append(("Reproducibility", test_reproducibility()))
    results.append(("Dataset", test_dataset_loading()))
    results.append(("Metrics", test_metrics()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed! Framework is ready to use.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
