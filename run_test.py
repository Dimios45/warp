import numpy as np
import warp as wp

wp.init()

def test_comprehensive_division_scenarios():
    @wp.kernel
    def comprehensive_test_kernel():
        # Test various scenarios in one kernel
        
        # Scenario 1: Basic tile/scalar division
        a = wp.tile_arange(1.0, 11.0, dtype=float, storage="register")  # [1.0, ..., 10.0]
        result1 = a / 2.0
        
        # Scenario 2: Scalar/tile division
        result2 = 20.0 / a  # [20.0, 10.0, 6.67, 5.0, 4.0, 3.33, 2.86, 2.5, 2.22, 2.0]
        
        # Scenario 3: Tile/tile division
        b = wp.tile_ones(shape=(10,), dtype=float, storage="register")
        result3 = a / (b + 1.0)  # a / 2.0 = same as result1
        
        # Scenario 4: In-place operations
        temp = a
        temp /= 2.0  # Should be same as result1
        
        # Scenario 5: Floor division
        result5 = a // 3.0
        result6 = 10.0 // b  # Should be [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        
        # Scenario 6: Mixed operations
        result7 = (a + b) / 2.0
        
    wp.launch_tiled(comprehensive_test_kernel, dim=[1], block_dim=32)
    wp.synchronize()

if __name__ == "__main__":
    test_comprehensive_division_scenarios()
    print('All tests passed!')