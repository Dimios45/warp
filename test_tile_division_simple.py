#!/usr/bin/env python

"""
Test to verify that tile division operations work properly by creating a minimal test.
"""

import warp as wp

# Initialize Warp context
wp.init()

def test_basic_division():
    """Test if basic tile division works by using tile_map with builtins."""
    
    # First let's make sure that the basic div/floordiv functions exist by testing them on regular arrays
    @wp.kernel
    def test_basic_div_kernel():
        i = wp.tid()
        # Regular array division should work with these builtins
        pass  # Just testing if kernel compiles with division concepts
    
    # Test if we can run it
    try:
        print("Testing basic kernel compilation...")
        wp.launch(test_basic_div_kernel, dim=10)
        wp.synchronize()
        print("Basic kernel compilation test passed!")
    except Exception as e:
        print(f"Basic kernel compilation test failed: {e}")

def test_tile_class_division_methods():
    """Test the tile class division methods by creating a simple kernel."""
    
    @wp.kernel
    def simple_tile_kernel():
        # Create tile and use the tile class operators
        # This should use our new __ifloordiv__ method and the other division methods
        t = wp.tile_arange(1.0, 6.0, dtype=float)
        
        # Test all division operations
        result_truediv = t / 2.0          # Should use __truediv__
        result_rtruediv = 10.0 / t        # Should use __rtruediv__ 
        result_itruediv = t               # Copy original
        result_itruediv /= 2.0            # Should use __itruediv__
        
        result_floordiv = t // 2.0        # Should use __floordiv__
        result_rfloordiv = 10.0 // t      # Should use __rfloordiv__
        result_ifloordiv = t              # Copy original  
        result_ifloordiv //= 2.0          # Should use __ifloordiv__ (our new addition)
    
    # Try to run on CPU
    try:
        print("Testing tile division kernel on CPU...")
        wp.launch_tiled(simple_tile_kernel, dim=[1], block_dim=32, device="cpu")
        wp.synchronize("cpu")
        print("Tile division kernel passed on CPU!")
    except Exception as e:
        print(f"Tile division kernel failed on CPU: {e}")
        
        # This error gives us insight into what's wrong
        import traceback
        traceback.print_exc()

def test_tile_map_with_div():
    """Test tile_map directly with div operation."""
    
    @wp.kernel
    def test_tile_map_kernel():
        t = wp.tile_arange(1.0, 6.0, dtype=float)
        s = wp.tile_ones(shape=(5,), dtype=float)
        
        # Test if tile_map works with wp.add (which should be a known operation)
        result_add = wp.tile_map(wp.add, t, s)  # This should work
        
        # Now test if tile_map works with wp.div
        result_div = wp.tile_map(wp.div, t, s)  # This might work if wp.div is available
    
    try:
        print("Testing tile_map with div on CPU...")
        wp.launch_tiled(test_tile_map_kernel, dim=[1], block_dim=32, device="cpu")
        wp.synchronize("cpu")
        print("tile_map with div test passed!")
    except Exception as e:
        print(f"tile_map with div test failed: {e}")

if __name__ == "__main__":
    test_basic_division()
    test_tile_class_division_methods()
    test_tile_map_with_div()
    print("\nTests completed.")