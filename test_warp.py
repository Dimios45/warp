import warp as wp
import numpy as np

# Initialize Warp
wp.init()

# Create a simple kernel
@wp.kernel
def add_arrays(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# Create arrays
n = 10
a = wp.array(np.ones(n, dtype=np.float32))
b = wp.array(np.ones(n, dtype=np.float32) * 2.0)
c = wp.zeros_like(a)

# Launch the kernel
wp.launch(kernel=add_arrays, dim=n, inputs=[a, b, c])

# Wait for the computation to complete
wp.synchronize()

# Print result
result = c.numpy()
print('Result array:', result)
print('Expected: array of 3s, actual first 5 elements:', result[:5])

# Verify correctness
assert np.allclose(result, 3.0), 'Kernel did not produce expected result'
print('SUCCESS: Warp kernel executed correctly!')