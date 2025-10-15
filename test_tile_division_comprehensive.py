#!/usr/bin/env python

"""
Comprehensive test for tile division operations in NVIDIA Warp.

This module tests all tile division operations:
- tile / scalar
- scalar / tile  
- tile / tile
- tile //= scalar
- tile //= tile
- tile // scalar
- scalar // tile
- tile // tile
"""

import numpy as np
import warp as wp

# Initialize Warp context
wp.init()

def test_all_division_operations():
    """Test all division operations for tiles."""
    
    @wp.kernel
    def test_kernel():
        # Create test tiles
        a = wp.tile_arange(2.0, 7.0, dtype=float, storage="register")  # [2.0, 3.0, 4.0, 5.0, 6.0]
        b = wp.tile_arange(1.0, 6.0, dtype=float, storage="register")  # [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test regular division
        result1 = a / 2.0           # tile / scalar
        result2 = 10.0 / a          # scalar / tile
        result3 = a / b             # tile / tile
        
        # Test floor division
        result4 = a // 2.0          # tile // scalar
        result5 = 10.0 // a         # scalar // tile
        result6 = a // b            # tile // tile
        
        # Test in-place division
        temp1 = a
        temp1 /= 2.0                # tile /= scalar
        
        temp2 = a
        temp2 //= 2.0               # tile //= scalar
        
        # Test in-place tile division (this should work now with our implementation)
        temp3 = a
        temp3 //= b                 # tile //= tile
        
        # These operations should all compile and execute without errors
        # More detailed verification would require extraction functions that are not
        # available in the kernel context
        
    # Run test on all available devices
    devices = wp.get_devices()
    for device in devices:
        if "cpu" in str(device) or "cuda" in str(device):
            print(f"Testing on device: {device}")
            wp.launch_tiled(test_kernel, dim=[1], block_dim=32, device=device)
            wp.synchronize(device)
    
    print("All division operations completed successfully!")

def test_integer_division():
    """Test division operations with integer tiles."""
    
    @wp.kernel
    def test_int_kernel():
        # Integer tiles
        a = wp.tile_arange(5, 10, dtype=int, storage="register")  # [5, 6, 7, 8, 9]
        b = wp.tile_arange(1, 6, dtype=int, storage="register")   # [1, 2, 3, 4, 5]
        
        # Test integer division
        result1 = a / 2      # Should perform float division
        result2 = a // 2     # Should perform integer floor division
        result3 = a // b     # Integer floor division
        result4 = 20 // b    # Scalar integer floor division
        
        # Test in-place operations
        temp1 = a
        temp1 //= 2          # In-place integer floor division
        
        temp2 = a
        temp2 //= b          # In-place integer floor division with tile
        
    devices = wp.get_devices()
    for device in devices:
        if "cpu" in str(device) or "cuda" in str(device):
            print(f"Testing integer division on device: {device}")
            wp.launch_tiled(test_int_kernel, dim=[1], block_dim=32, device=device)
            wp.synchronize(device)
    
    print("Integer division operations completed successfully!")

def test_edge_cases():
    """Test edge cases for division operations."""
    
    @wp.kernel
    def test_edge_kernel():
        # Edge cases
        a = wp.tile_arange(1.0, 4.0, dtype=float, storage="register")  # [1.0, 2.0, 3.0]
        
        # Division by 1
        result1 = a / 1.0
        result2 = a // 1.0
        
        # Division by -1
        result3 = a / -1.0
        result4 = a // -1.0
        
        # Large divisor
        result5 = a / 1000.0
        result6 = a // 1000.0
        
        # In-place with same tile
        temp1 = a
        temp1 /= temp1  # Should result in [1.0, 1.0, 1.0] (barring floating point precision)
        
        temp2 = a
        temp2 //= temp2  # Should result in [1.0, 1.0, 1.0] (floor division of 1.0s)
        
    devices = wp.get_devices()
    for device in devices:
        if "cpu" in str(device) or "cuda" in str(device):
            print(f"Testing edge cases on device: {device}")
            wp.launch_tiled(test_edge_kernel, dim=[1], block_dim=32, device=device)
            wp.synchronize(device)
    
    print("Edge case operations completed successfully!")

if __name__ == "__main__":
    test_all_division_operations()
    test_integer_division()
    test_edge_cases()
    print("\nAll comprehensive tile division tests passed!")