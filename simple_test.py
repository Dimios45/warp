#!/usr/bin/env python
"""
Simple test for tile division functionality
"""
import warp as wp
wp.init()

# Define the test kernel in a file
@wp.kernel
def test_div():
    a = wp.tile_arange(1.0, 4.0, dtype=float)  # [1.0, 2.0, 3.0]
    result = a / 2.0  # Should be [0.5, 1.0, 1.5]
    # Note: We can't easily print the result in the kernel for verification
    # but we can at least test that it doesn't error out

# Launch the test
wp.launch_tiled(test_div, dim=[1], block_dim=32)
wp.synchronize()
print('Tile division test completed successfully!')