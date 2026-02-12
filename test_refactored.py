#!/usr/bin/env python3
"""
Quick test script to verify the refactored hdf_preparator_v2.py works correctly.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, '/home/marc/dev/fdq_diffusion')

try:
    from hdf_preparator_v2 import (
        DatasetType, DatasetConfig, SampleInfo, 
        TensorFormatter, HDFKeyProcessor, 
        HDF5FileProcessor, H5Dataset, DatasetFactory,
        create_datasets
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_tensor_formatter():
    """Test the TensorFormatter class."""
    formatter = TensorFormatter()
    
    # Test 2D tensor
    tensor_2d = torch.randn(28, 28)
    result = formatter.to_BCHW(tensor_2d)
    assert result.shape == (1, 1, 28, 28), f"Expected (1,1,28,28), got {result.shape}"
    
    # Test 3D tensor (CHW format)
    tensor_3d_chw = torch.randn(3, 28, 28)
    result = formatter.to_BCHW(tensor_3d_chw)
    assert result.shape == (1, 3, 28, 28), f"Expected (1,3,28,28), got {result.shape}"
    
    # Test 3D tensor (BHW format)
    tensor_3d_bhw = torch.randn(32, 28, 28)
    result = formatter.to_BCHW(tensor_3d_bhw)
    assert result.shape == (32, 1, 28, 28), f"Expected (32,1,28,28), got {result.shape}"
    
    print("✓ TensorFormatter tests passed")

def test_hdf_key_processor():
    """Test the HDFKeyProcessor class."""
    processor = HDFKeyProcessor()
    
    # Test None input
    result = processor.cleanup_hdf_keys(None)
    assert result == (None, None), f"Expected (None, None), got {result}"
    
    # Test string list input
    keys = ["image", "label"]
    result = processor.cleanup_hdf_keys(keys)
    assert result == (keys, None), f"Expected ({keys}, None), got {result}"
    
    # Test dictionary input
    dict_keys = [{"image": "img_transform"}, {"label": "label_transform"}]
    result = processor.cleanup_hdf_keys(dict_keys)
    expected = (["image", "label"], ["img_transform", "label_transform"])
    assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ HDFKeyProcessor tests passed")

def test_dataset_config():
    """Test the DatasetConfig dataclass."""
    # Test default values
    config = DatasetConfig()
    assert config.is_3d == False
    assert config.train_batch_size == 32
    assert config.num_workers == 4
    
    # Test custom values
    config = DatasetConfig(is_3d=True, train_batch_size=64, num_workers=8)
    assert config.is_3d == True
    assert config.train_batch_size == 64
    assert config.num_workers == 8
    
    print("✓ DatasetConfig tests passed")

def test_sample_info():
    """Test the SampleInfo dataclass."""
    info = SampleInfo(file_idx=0, group_idx=1, dataset_idx=2, sample_idx=3)
    assert info.file_idx == 0
    assert info.group_idx == 1
    assert info.dataset_idx == 2
    assert info.sample_idx == 3
    
    print("✓ SampleInfo tests passed")

def test_dataset_type():
    """Test the DatasetType enum."""
    assert DatasetType.TRAIN.value == "train"
    assert DatasetType.VAL.value == "val"
    assert DatasetType.TEST.value == "test"
    assert DatasetType.ALL.value == "all"
    
    print("✓ DatasetType tests passed")

def main():
    """Run all tests."""
    print("Testing refactored hdf_preparator_v2.py...")
    print("=" * 50)
    
    try:
        test_tensor_formatter()
        test_hdf_key_processor()
        test_dataset_config()
        test_sample_info()
        test_dataset_type()
        
        print("=" * 50)
        print("✓ All tests passed! The refactored code is working correctly.")
        print("✓ The create_datasets function is available for backward compatibility.")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
