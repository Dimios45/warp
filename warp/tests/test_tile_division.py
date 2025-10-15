#!/usr/bin/env python
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np

import warp as wp
from warp.utils import array_equal

wp.init()


@wp.kernel
def test_kernel():
    # Simple kernel to test tile division operations
    tid = wp.tid()
    wp.printf("Thread %d\n", tid)


def test_tile_division_basic():
    """Test basic tile division functionality: tile/scalar, scalar/tile, tile/tile"""
    
    device = "cpu"
    
    # Test tile / scalar division
    @wp.kernel
    def tile_div_scalar_kernel(
        input_tile: wp.array(dtype=wp.float32),
        output_tile: wp.array(dtype=wp.float32),
        divisor: wp.float32
    ):
        tid = wp.tid()
        t = wp.tile_load(input_tile, shape=(4,), storage="register")
        result = t / divisor
        wp.tile_store(output_tile, result, offset=(tid * 4))
        
    # Create input data
    input_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    input_tile = wp.array(input_data, device=device)
    output_tile = wp.zeros_like(input_tile)
    
    wp.launch(
        kernel=tile_div_scalar_kernel,
        dim=1,
        inputs=[input_tile, output_tile, 2.0],
        device=device
    )
    
    expected = input_data / 2.0
    actual = output_tile.numpy()
    assert np.allclose(actual, expected), f"Expected {expected}, got {actual}"
    
    # Test scalar / tile division
    @wp.kernel
    def scalar_div_tile_kernel(
        input_tile: wp.array(dtype=wp.float32),
        output_tile: wp.array(dtype=wp.float32),
        dividend: wp.float32
    ):
        tid = wp.tid()
        t = wp.tile_load(input_tile, shape=(4,), storage="register")
        result = dividend / t
        wp.tile_store(output_tile, result, offset=(tid * 4))
        
    # Use different input data to avoid division by zero
    input_data = np.array([2.0, 4.0, 5.0, 10.0], dtype=np.float32)
    input_tile = wp.array(input_data, device=device)
    output_tile = wp.zeros_like(input_tile)
    
    wp.launch(
        kernel=scalar_div_tile_kernel,
        dim=1,
        inputs=[input_tile, output_tile, 20.0],
        device=device
    )
    
    expected = 20.0 / input_data
    actual = output_tile.numpy()
    assert np.allclose(actual, expected), f"Expected {expected}, got {actual}"
    
    # Test tile / tile division
    @wp.kernel
    def tile_div_tile_kernel(
        input_tile_a: wp.array(dtype=wp.float32),
        input_tile_b: wp.array(dtype=wp.float32),
        output_tile: wp.array(dtype=wp.float32)
    ):
        tid = wp.tid()
        a = wp.tile_load(input_tile_a, shape=(4,), storage="register")
        b = wp.tile_load(input_tile_b, shape=(4,), storage="register")
        result = a / b
        wp.tile_store(output_tile, result, offset=(tid * 4))
        
    input_data_a = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    input_data_b = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_tile_a = wp.array(input_data_a, device=device)
    input_tile_b = wp.array(input_data_b, device=device)
    output_tile = wp.zeros(4, dtype=wp.float32, device=device)
    
    wp.launch(
        kernel=tile_div_tile_kernel,
        dim=1,
        inputs=[input_tile_a, input_tile_b, output_tile],
        device=device
    )
    
    expected = input_data_a / input_data_b
    actual = output_tile.numpy()
    assert np.allclose(actual, expected), f"Expected {expected}, got {actual}"


def test_tile_floordiv_basic():
    """Test basic tile floor division functionality: tile//scalar, scalar//tile, tile//tile"""
    
    device = "cpu"
    
    # Test tile // scalar floor division
    @wp.kernel
    def tile_floordiv_scalar_kernel(
        input_tile: wp.array(dtype=wp.float32),
        output_tile: wp.array(dtype=wp.float32),
        divisor: wp.float32
    ):
        tid = wp.tid()
        t = wp.tile_load(input_tile, shape=(4,), storage="register")
        result = t // divisor
        wp.tile_store(output_tile, result, offset=(tid * 4))
        
    # Create input data
    input_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    input_tile = wp.array(input_data, device=device)
    output_tile = wp.zeros_like(input_tile)
    
    wp.launch(
        kernel=tile_floordiv_scalar_kernel,
        dim=1,
        inputs=[input_tile, output_tile, 3.0],
        device=device
    )
    
    expected = input_data // 3.0
    actual = output_tile.numpy()
    assert np.array_equal(actual, expected), f"Expected {expected}, got {actual}"
    
    # Test scalar // tile floor division
    @wp.kernel
    def scalar_floordiv_tile_kernel(
        input_tile: wp.array(dtype=wp.float32),
        output_tile: wp.array(dtype=wp.float32),
        dividend: wp.float32
    ):
        tid = wp.tid()
        t = wp.tile_load(input_tile, shape=(4,), storage="register")
        result = dividend // t
        wp.tile_store(output_tile, result, offset=(tid * 4))
        
    # Use different input data to avoid division by zero
    input_data = np.array([2.0, 4.0, 5.0, 3.0], dtype=np.float32)
    input_tile = wp.array(input_data, device=device)
    output_tile = wp.zeros_like(input_tile)
    
    wp.launch(
        kernel=scalar_floordiv_tile_kernel,
        dim=1,
        inputs=[input_tile, output_tile, 20.0],
        device=device
    )
    
    expected = 20.0 // input_data
    actual = output_tile.numpy()
    assert np.array_equal(actual, expected), f"Expected {expected}, got {actual}"
    
    # Test tile // tile floor division
    @wp.kernel
    def tile_floordiv_tile_kernel(
        input_tile_a: wp.array(dtype=wp.float32),
        input_tile_b: wp.array(dtype=wp.float32),
        output_tile: wp.array(dtype=wp.float32)
    ):
        tid = wp.tid()
        a = wp.tile_load(input_tile_a, shape=(4,), storage="register")
        b = wp.tile_load(input_tile_b, shape=(4,), storage="register")
        result = a // b
        wp.tile_store(output_tile, result, offset=(tid * 4))
        
    input_data_a = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    input_data_b = np.array([3.0, 7.0, 4.0, 6.0], dtype=np.float32)
    input_tile_a = wp.array(input_data_a, device=device)
    input_tile_b = wp.array(input_data_b, device=device)
    output_tile = wp.zeros(4, dtype=wp.float32, device=device)
    
    wp.launch(
        kernel=tile_floordiv_tile_kernel,
        dim=1,
        inputs=[input_tile_a, input_tile_b, output_tile],
        device=device
    )
    
    expected = input_data_a // input_data_b
    actual = output_tile.numpy()
    assert np.array_equal(actual, expected), f"Expected {expected}, got {actual}"


def test_tile_inplace_division():
    """Test in-place tile division functionality: tile /= scalar, tile /= tile"""
    
    device = "cpu"
    
    # Test tile /= scalar in-place division
    @wp.kernel
    def tile_itruediv_kernel(
        input_tile: wp.array(dtype=wp.float32),
        divisor: wp.float32
    ):
        tid = wp.tid()
        t = wp.tile_load(input_tile, shape=(4,), storage="register")
        t /= divisor
        wp.tile_store(input_tile, t, offset=(tid * 4))
        
    # Create input data
    input_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    input_tile = wp.array(input_data.copy(), device=device)
    
    wp.launch(
        kernel=tile_itruediv_kernel,
        dim=1,
        inputs=[input_tile, 2.0],
        device=device
    )
    
    expected = input_data / 2.0
    actual = input_tile.numpy()
    assert np.allclose(actual, expected), f"Expected {expected}, got {actual}"
    
    # Test tile //= scalar in-place floor division
    @wp.kernel
    def tile_ifloordiv_kernel(
        input_tile: wp.array(dtype=wp.float32),
        divisor: wp.float32
    ):
        tid = wp.tid()
        t = wp.tile_load(input_tile, shape=(4,), storage="register")
        t //= divisor
        wp.tile_store(input_tile, t, offset=(tid * 4))
        
    # Create input data
    input_data = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    input_tile = wp.array(input_data.copy(), device=device)
    
    wp.launch(
        kernel=tile_ifloordiv_kernel,
        dim=1,
        inputs=[input_tile, 3.0],
        device=device
    )
    
    expected = input_data // 3.0
    actual = input_tile.numpy()
    assert np.array_equal(actual, expected), f"Expected {expected}, got {actual}"


def test_tile_division_types():
    """Test tile division with different data types"""
    
    device = "cpu"
    
    # Test integer division
    @wp.kernel
    def tile_div_int_kernel(
        input_tile: wp.array(dtype=wp.int32),
        output_tile: wp.array(dtype=wp.int32),
        divisor: wp.int32
    ):
        tid = wp.tid()
        t = wp.tile_load(input_tile, shape=(4,), storage="register")
        result = t / divisor
        wp.tile_store(output_tile, result, offset=(tid * 4))
        
    # Create input data
    input_data = np.array([10, 20, 30, 40], dtype=np.int32)
    input_tile = wp.array(input_data, device=device)
    output_tile = wp.zeros(4, dtype=wp.int32, device=device)
    
    wp.launch(
        kernel=tile_div_int_kernel,
        dim=1,
        inputs=[input_tile, output_tile, 2],
        device=device
    )
    
    expected = input_data / 2
    actual = output_tile.numpy()
    assert np.allclose(actual, expected), f"Expected {expected}, got {actual}"
    
    # Test different scalar types
    types_to_test = [wp.float16, wp.float32, wp.float64]
    for dtype in types_to_test:
        # Create input data
        input_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=getattr(np, str(dtype)[5:]))
        input_tile = wp.array(input_data, dtype=dtype, device=device)
        output_tile = wp.zeros(4, dtype=dtype, device=device)
        
        @wp.kernel
        def tile_div_type_kernel(
            input_tile: wp.array(dtype=Any),
            output_tile: wp.array(dtype=Any),
            divisor: Any
        ):
            tid = wp.tid()
            t = wp.tile_load(input_tile, shape=(4,), storage="register")
            result = t / divisor
            wp.tile_store(output_tile, result, offset=(tid * 4))
            
        # Use wp.Launch to avoid the 'Any' typing issue in the kernel signature definition
        wp.launch(
            kernel=tile_div_type_kernel,
            dim=1,
            inputs=[input_tile, output_tile, 2.0],
            device=device
        )
        
        expected = input_data / 2.0
        actual = output_tile.numpy()
        assert np.allclose(actual, expected), f"Type {dtype}: Expected {expected}, got {actual}"


def test_tile_division_shapes():
    """Test tile division with different shapes"""
    
    device = "cpu"
    
    # Test 2D tile division
    @wp.kernel
    def tile_2d_div_kernel(
        input_tile_a: wp.array2d(dtype=wp.float32),
        input_tile_b: wp.array2d(dtype=wp.float32),
        output_tile: wp.array2d(dtype=wp.float32)
    ):
        tid = wp.tid()
        a = wp.tile_load(input_tile_a, shape=(2, 3), storage="register")
        b = wp.tile_load(input_tile_b, shape=(2, 3), storage="register")
        result = a / b
        wp.tile_store(output_tile, result, offset=(tid * 2, 0))
        
    # Create input data
    input_data_a = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    input_data_b = np.array([[2.0, 4.0, 5.0], [8.0, 10.0, 12.0]], dtype=np.float32)
    input_tile_a = wp.array(input_data_a, device=device)
    input_tile_b = wp.array(input_data_b, device=device)
    output_tile = wp.zeros_like(input_tile_a)
    
    wp.launch(
        kernel=tile_2d_div_kernel,
        dim=1,
        inputs=[input_tile_a, input_tile_b, output_tile],
        device=device
    )
    
    expected = input_data_a / input_data_b
    actual = output_tile.numpy()
    assert np.allclose(actual, expected), f"Expected {expected}, got {actual}"


def test_tile_division_edge_cases():
    """Test tile division edge cases"""
    
    device = "cpu"
    
    # Test division by zero (should handle gracefully in kernel)
    @wp.kernel
    def tile_div_zero_kernel(
        input_tile: wp.array(dtype=wp.float32),
        output_tile: wp.array(dtype=wp.float32),
        divisor: wp.float32
    ):
        tid = wp.tid()
        t = wp.tile_load(input_tile, shape=(4,), storage="register")
        result = t / divisor
        wp.tile_store(output_tile, result, offset=(tid * 4))
    
    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_tile = wp.array(input_data, device=device)
    output_tile = wp.zeros_like(input_tile)
    
    # This should produce inf/nan values, which we can check
    wp.launch(
        kernel=tile_div_zero_kernel,
        dim=1,
        inputs=[input_tile, output_tile, 0.0],
        device=device
    )
    
    result = output_tile.numpy()
    # Check that we have inf values (not crashes)
    assert np.isinf(result).any() or np.isnan(result).any(), "Division by zero did not produce inf/nan as expected"
    
    # Test with negative values
    @wp.kernel
    def tile_div_negative_kernel(
        input_tile: wp.array(dtype=wp.float32),
        output_tile: wp.array(dtype=wp.float32),
        divisor: wp.float32
    ):
        tid = wp.tid()
        t = wp.tile_load(input_tile, shape=(4,), storage="register")
        result = t / divisor
        wp.tile_store(output_tile, result, offset=(tid * 4))
        
    input_data = np.array([-10.0, 20.0, -30.0, 40.0], dtype=np.float32)
    input_tile = wp.array(input_data, device=device)
    output_tile = wp.zeros_like(input_tile)
    
    wp.launch(
        kernel=tile_div_negative_kernel,
        dim=1,
        inputs=[input_tile, output_tile, -2.0],
        device=device
    )
    
    expected = input_data / -2.0
    actual = output_tile.numpy()
    assert np.allclose(actual, expected), f"Expected {expected}, got {actual}"


def test_tile_division_with_different_dtypes():
    """Test tile division with different data types combinations"""
    
    device = "cpu"
    
    # Testing compatibility between different dtypes
    test_cases = [
        (wp.float32, wp.float32),
        (wp.float32, wp.int32),
        (wp.int32, wp.float32),
    ]
    
    for dtype_a, dtype_b in test_cases:
        # Create input data
        input_data_a = np.array([10.0, 20.0, 30.0, 40.0], dtype=getattr(np, str(dtype_a)[5:] if str(dtype_a)[5:] != 'float' else 'float32'))
        input_data_b = np.array([2.0, 4.0, 5.0, 8.0], dtype=getattr(np, str(dtype_b)[5:] if str(dtype_b)[5:] != 'float' else 'float32'))
        
        input_tile_a = wp.array(input_data_a, dtype=dtype_a, device=device)
        input_tile_b = wp.array(input_data_b, dtype=dtype_b, device=device)
        output_tile = wp.zeros(4, dtype=wp.float32, device=device)  # result will be in float32
        
        @wp.kernel
        def tile_div_dtypes_kernel(
            input_tile_a: wp.array(dtype=Any),
            input_tile_b: wp.array(dtype=Any),
            output_tile: wp.array(dtype=Any)
        ):
            tid = wp.tid()
            a = wp.tile_load(input_tile_a, shape=(4,), storage="register")
            b = wp.tile_load(input_tile_b, shape=(4,), storage="register")
            result = a / b
            wp.tile_store(output_tile, result, offset=(tid * 4))
        
        wp.launch(
            kernel=tile_div_dtypes_kernel,
            dim=1,
            inputs=[input_tile_a, input_tile_b, output_tile],
            device=device
        )
        
        expected = input_data_a.astype(np.float32) / input_data_b.astype(np.float32)
        actual = output_tile.numpy()
        assert np.allclose(actual, expected), f"Case {dtype_a} / {dtype_b}: Expected {expected}, got {actual}"


if __name__ == "__main__":
    test_tile_division_basic()
    test_tile_floordiv_basic()
    test_tile_inplace_division()
    test_tile_division_types()
    test_tile_division_shapes()
    test_tile_division_edge_cases()
    test_tile_division_with_different_dtypes()
    print("All tests passed!")