import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit  # Do not remove: it initializes CUDA context!
from pycuda.compiler import SourceModule

DEVICE_DATA = pycuda.tools.DeviceData()
BLOCK_SIZE = 32

n = 16
a = (np.random.randn(n, n) * 100).astype(np.float32)
b = (np.random.randn(n, n) * 100).astype(np.float32)
c = np.empty((n, n)).astype(np.float32)

mod = SourceModule(open("kernel_basic.cu", "r").read())
matmul = mod.get_function("matmul")

# set grid size
grid = (n // BLOCK_SIZE + 1 if n % BLOCK_SIZE != 0 else 0, n // BLOCK_SIZE + 1 if n % BLOCK_SIZE != 0 else 0, 1)

# call gpu function
start = time.time()
matmul(
    np.int32(n), cuda.In(a), cuda.In(b), cuda.Out(c),
    block=(np.int(BLOCK_SIZE), np.int(BLOCK_SIZE), 1),
    grid=grid
)
print(f'Matrix multiplication - time: {time.time() - start:.5f} s')

assert (np.allclose(np.dot(a, b), c))
np.testing.assert_allclose(np.dot(a, b), c, 1e-5)
