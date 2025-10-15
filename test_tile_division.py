#!/usr/bin/env python

import numpy as np
import warp as wp

# Test tile division operations
def test_tile_division():
    print("Testing tile division operations...")
    
    wp.init()
    
    @wp.kernel
    def test_div_kernel():
        # Create tiles for testing
        a = wp.tile_arange(1.0, 11.0, dtype=float, storage="register")  # [1.0, 2.0, ..., 10.0]
        b = wp.tile_ones(shape=(10,), dtype=float, storage="register")  # [1.0, 1.0, ..., 1.0] 
        c = wp.tile_arange(2.0, 12.0, dtype=float, storage="register")  # [2.0, 3.0, ..., 11.0]
        
        # Test tile / scalar
        result1 = a / 2.0  # Should be [0.5, 1.0, 1.5, ..., 5.0]
        
        # Test scalar / tile  
        result2 = 10.0 / b  # Should be [10.0, 10.0, ..., 10.0]
        
        # Test tile / tile
        result3 = a / c   # Should be [1.0/2.0, 2.0/3.0, ..., 10.0/11.0]
        
        # Test in-place division
        temp = wp.tile_arange(2.0, 12.0, dtype=float, storage="register")  # [2.0, 3.0, ..., 11.0]
        temp /= 2.0  # Should become [1.0, 1.5, ..., 5.5]
        
        # Test floor division
        result4 = a // 3.0  # Should be [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0] (integers)
        
        # Print results for verification
        print("a:", a)
        print("a / 2.0:", result1) 
        print("10.0 / b:", result2)
        print("a / c:", result3)
        print("temp (after temp /= 2.0):", temp)
        print("a // 3.0:", result4)
    
    # Launch the test kernel
    wp.launch_tiled(test_div_kernel, dim=[1], block_dim=32)
    wp.synchronize()

if __name__ == "__main__":
    test_tile_division()
    print("Test completed successfully!")