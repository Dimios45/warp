#!/usr/bin/env python

"""
Simplified test for tile division operations in NVIDIA Warp.
"""

import warp as wp

# Initialize Warp context
wp.init()

def test_division_simple():
    """Test basic division operations for tiles."""
    
    @wp.kernel
    def test_kernel():
        # Create test tiles
        a = wp.tile_arange(2.0, 7.0, dtype=float, storage="register")  # [2.0, 3.0, 4.0, 5.0, 6.0]
        
        # Test basic div operations that should work
        result1 = wp.tile_map(wp.div, a, 2.0)           # tile / scalar using tile_map directly
        result2 = wp.tile_map(wp.div, 2.0, a)           # scalar / tile using tile_map directly
        
        # Test basic floordiv operations that should work
        result3 = wp.tile_map(wp.floordiv, a, 2.0)      # tile // scalar using tile_map directly
        result4 = wp.tile_map(wp.floordiv, 2.0, a)      # scalar // tile using tile_map directly
        
    # Run test on CPU
    try:
        print("Testing tile_map operations directly on CPU...")
        wp.launch_tiled(test_kernel, dim=[1], block_dim=32, device="cpu")
        wp.synchronize("cpu")
        print("Direct tile_map operations completed successfully!")
    except Exception as e:
        print(f"Direct tile_map operations failed: {e}")
    
    # Run test on CUDA if available
    devices = wp.get_devices()
    for device in devices:
        if "cuda" in str(device):
            try:
                print(f"Testing tile_map operations directly on {device}...")
                wp.launch_tiled(test_kernel, dim=[1], block_dim=32, device=device)
                wp.synchronize(device)
                print(f"Direct tile_map operations completed successfully on {device}!")
            except Exception as e:
                print(f"Direct tile_map operations failed on {device}: {e}")

def test_tile_class_operations():
    """Test tile class operator methods."""
    
    @wp.kernel
    def test_kernel():
        # Create test tiles
        a = wp.tile_arange(2.0, 7.0, dtype=float, storage="register")  # [2.0, 3.0, 4.0, 5.0, 6.0]
        
        # Test if we can use the tile class's overloaded operators
        # Now that we have the __ifloordiv__ method implemented
        result1 = a / 2.0      # tile / scalar
        result2 = 10.0 / a     # scalar / tile
        result3 = a // 2.0     # tile // scalar
        result4 = 10.0 // a    # scalar // tile
        
        # Test in-place operations
        temp_a = a
        temp_a /= 2.0          # tile /= scalar
        temp_b = a
        temp_b //= 2.0         # tile //= scalar
        
    # Run test on CPU
    try:
        print("Testing tile class operations on CPU...")
        wp.launch_tiled(test_kernel, dim=[1], block_dim=32, device="cpu")
        wp.synchronize("cpu")
        print("Tile class operations completed successfully!")
    except Exception as e:
        print(f"Tile class operations failed: {e}")

if __name__ == "__main__":
    test_division_simple()
    test_tile_class_operations()
    print("\nTest run completed!")