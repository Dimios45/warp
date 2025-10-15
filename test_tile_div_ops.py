#!/usr/bin/env python
"""
Simple test for tile division functionality - focusing on tile/tile operations
"""
import warp as wp
wp.init()

@wp.kernel
def test_tile_div():
    a = wp.tile_arange(2.0, 5.0, dtype=float)  # [2.0, 3.0, 4.0]
    b = wp.tile_arange(1.0, 4.0, dtype=float)  # [1.0, 2.0, 3.0]
    result = a / b  # Should be [2.0, 1.5, 1.33...]
    # We can't easily print results in kernels, but we can test that it compiles and runs

@wp.kernel
def test_tile_floor_div():
    a = wp.tile_arange(7.0, 10.0, dtype=float)  # [7.0, 8.0, 9.0]
    b = wp.tile_arange(2.0, 5.0, dtype=float)  # [2.0, 3.0, 4.0]
    result = a // b  # Should be [3.0, 2.0, 2.0] (floor division)
    # We can't easily print results in kernels, but we can test that it compiles and runs

# Test tile/tile division
wp.launch_tiled(test_tile_div, dim=[1], block_dim=32)
wp.synchronize()
print('Tile/tile division test completed successfully!')

# Test tile/tile floor division
wp.launch_tiled(test_tile_floor_div, dim=[1], block_dim=32) 
wp.synchronize()
print('Tile/tile floor division test completed successfully!')

# Test the tile class methods (which internally use tile_map)
@wp.kernel
def test_tile_scalar_via_class():
    a = wp.tile_arange(2.0, 5.0, dtype=float)  # [2.0, 3.0, 4.0]
    result = a / 2.0  # Uses tile class __truediv__ which calls tile_map internally
    # This should work because of the tile class methods we implemented

wp.launch_tiled(test_tile_scalar_via_class, dim=[1], block_dim=32)
wp.synchronize()
print('Tile/scalar division via tile class methods completed successfully!')