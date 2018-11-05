import pycuda.driver as cuda
import pycuda.autoinit  # Do not remove: it initializes CUDA context!
import time
from pycuda.compiler import SourceModule
import numpy

n, m, p = numpy.int32(100), numpy.int32(400), numpy.int32(16)

a = numpy.random.random_sample((n, m)).astype(numpy.float32)
b = numpy.random.random_sample((m, p)).astype(numpy.float32)
c = numpy.zeros((n, p)).astype(numpy.float32)

mod = SourceModule(open("kernel.cu", "r").read())

func = mod.get_function("multiply")
start = time.time()
func(n, m, p, cuda.In(a), cuda.In(b), cuda.Out(c),
     block=(numpy.int(n), numpy.int(p), 1),
     grid=(1, 1),
     shared=0)
print(f'Matrix multiplication - time: {time.time() - start:.5f} s')
assert (numpy.allclose(numpy.dot(a, b), c))
numpy.testing.assert_allclose(numpy.dot(a, b), c, 1e-5)
