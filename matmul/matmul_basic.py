import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit  # Do not remove: it initializes CUDA context!
from pycuda.compiler import SourceModule


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[1] != b.shape[0]:
        raise ValueError("Matrices are not corresponding for multiplication!")

    start = time.time()
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)

    mod = SourceModule(open("kernel_basic.cu", "r").read())
    matmul = mod.get_function("matmul")

    block = (16, 16, 1)
    grid = (
        int(np.ceil(result.shape[0] / block[0])),
        int(np.ceil(result.shape[1] / block[1]))
    )

    matmul(
        np.int32(result.shape[0]),
        np.int32(a.shape[1]),
        np.int32(result.shape[1]),
        cuda.In(a),
        cuda.In(b),
        cuda.Out(result),
        block=block,
        grid=grid
    )
    print(f'multiply - calculation time: {time.time() - start:.5f} s')
    return result


def assert_close_multiplication(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    np.testing.assert_allclose(np.dot(a, b), c, atol=1e-5)
    assert np.allclose(np.dot(a, b), c, atol=1e-5)


def test_multiplication():
    n = 200
    m = 300
    p = 400
    a = (np.random.randn(n, m) * 100).astype(np.float32)
    b = (np.random.randn(m, p) * 100).astype(np.float32)
    c = multiply(a, b)

    assert_close_multiplication(a, b, c)


if __name__ == '__main__':
    test_multiplication()
