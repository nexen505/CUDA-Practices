from typing import Tuple

import pyfftw
import numpy
import cupy
import unittest
import time


class FFTTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        shape = int(5e3)
        self.test_array = pyfftw.empty_aligned(shape, dtype='complex128')
        self.test_array[:] = numpy.random.randn(shape) + 1j * numpy.random.randn(shape)
        self.test_cupy_array = cupy.array(self.test_array, dtype='complex128')

    def get_ffts(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        start = time.time()
        expected_f = cupy.fft.fft(self.test_cupy_array)
        cupy.cuda.Device().synchronize()
        print("Time cupy.fft.fft: %.5f s" % (time.time() - start))
        start = time.time()
        stream = cupy.cuda.stream.Stream()
        stream.use()
        f = cupy.fft.fft(self.test_cupy_array)
        stream.synchronize()
        print("Time cupy.fft.fft with stream: %.5f s" % (time.time() - start))
        return expected_f, f

    def test_pyfftw(self):
        start = time.time()
        b = pyfftw.interfaces.numpy_fft.fft(self.test_array)
        print("Time pyfftw.interfaces.numpy_fft.fft: %.5f s" % (time.time() - start))
        start = time.time()
        c = numpy.fft.fft(self.test_array)
        print("Time numpy.fft.fft: %.5f s" % (time.time() - start))
        start = time.time()
        allclose = numpy.allclose(b, c)
        print("Time numpy.allclose: %.5f s" % (time.time() - start))
        self.assertTrue(allclose)

    def test_cupy_equal_ffts(self):
        expected_f, f = self.get_ffts()
        start = time.time()
        cupy.testing.assert_array_equal(f, expected_f)
        print("Time cupy.testing.assert_array_equal: %.5f s" % (time.time() - start))

    def test_cupy_close_ffts(self):
        c = numpy.fft.fft(self.test_array)
        expected_f, f = self.get_ffts()
        start = time.time()
        cupy.testing.assert_allclose(f, c)
        print("Time cupy.testing.assert_allclose: %.5f s" % (time.time() - start))
        start = time.time()
        cupy.testing.assert_allclose(expected_f, c)
        print("Time cupy.testing.assert_allclose: %.5f s" % (time.time() - start))


if __name__ == '__main__':
    unittest.main()
