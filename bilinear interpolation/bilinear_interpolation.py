import os
import time

import matplotlib.pyplot as plt
import numpy as np
import imageio
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule


def get_bilinear_pixel(image: np.ndarray, x: int, y: int, scale: int = 2) -> list:
    out = []
    position_x = x / scale
    position_y = y / scale

    modXi = int(position_x)
    modYi = int(position_y)
    modXf = position_x - modXi
    modYf = position_y - modYi
    modXiPlusOneLim = 0 if modXi + 1 >= image.shape[0] else modXi + 1
    modYiPlusOneLim = 0 if modYi + 1 >= image.shape[1] else modYi + 1

    for chan in range(image.shape[2]):
        bl = image[modXi, modYi, chan]
        br = image[modXiPlusOneLim, modYi, chan]
        tl = image[modXi, modYiPlusOneLim, chan]
        tr = image[modXiPlusOneLim, modYiPlusOneLim, chan]

        b = (1.0 - modXf) * bl + modXf * br
        t = (1.0 - modXf) * tl + modXf * tr
        pxf = (1.0 - modYf) * b + modYf * t
        out.append(int(pxf))

    return out


def get_image_enlarged_shape(image: np.ndarray, scale: int = 2) -> list:
    return list(map(int, [image.shape[0] * scale, image.shape[1] * scale, image.shape[2]]))


def interpolate_image_manually(image: np.ndarray, scale: int = 2):
    start = time.time()
    enlarged_image_shape = get_image_enlarged_shape(image, scale)
    enlarged_image = np.empty(enlarged_image_shape, dtype=np.uint8)
    for x in range(enlarged_image.shape[0]):
        for y in range(enlarged_image.shape[1]):
            enlarged_image[x, y] = get_bilinear_pixel(image, x, y, scale)
    print(f'interpolate_image_manually - calculation time: {time.time() - start:.5f} s')
    return enlarged_image


def interpolate_image_by_cuda(image: np.ndarray, scale: int = 2):
    start = time.time()
    cu_module = SourceModule(open("kernel.cu", "r").read())

    interpolate = cu_module.get_function("interpolate")

    cu_tex = cu_module.get_texref("tex")
    cu_tex.set_filter_mode(cuda.filter_mode.POINT)
    cu_tex.set_address_mode(0, cuda.address_mode.CLAMP)
    cu_tex.set_address_mode(1, cuda.address_mode.CLAMP)

    enlarged_image_shape = get_image_enlarged_shape(image, scale)

    rgba_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
    for x in range(rgba_image.shape[0]):
        for y in range(rgba_image.shape[1]):
            for ch in range(image.shape[2]):
                rgba_image[x, y] += image[x, y, ch] << (8 * (image.shape[2] - ch - 1))
    cuda.matrix_to_texref(rgba_image, cu_tex, order="C")

    output = np.zeros((enlarged_image_shape[0], enlarged_image_shape[1]), dtype=np.uint32)
    block = (np.int(16), np.int(16), 1)
    grid = (
        int(np.ceil(enlarged_image_shape[0] / block[0])),
        int(np.ceil(enlarged_image_shape[1] / block[1]))
    )

    interpolate(cuda.Out(output),
                np.int32(image.shape[1]),
                np.int32(image.shape[0]),
                np.int32(enlarged_image_shape[1]),
                np.int32(enlarged_image_shape[0]),
                block=block,
                grid=grid,
                texrefs=[cu_tex])

    output_image = np.zeros((enlarged_image_shape[0], enlarged_image_shape[1], 4), dtype=np.uint32)
    for x in range(output_image.shape[0]):
        for y in range(output_image.shape[1]):
            for ch in range(output_image.shape[2]):
                output_image[x, y, output_image.shape[2] - ch - 1] = output[x, y] % 256
                output[x, y] //= 256

    print(f'interpolate_image_by_cuda - calculation time: {time.time() - start:.5f} s')
    return output_image


def show_image(image, title: str = ''):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def calculate_image(image: np.ndarray, scale: int = 2):
    image_by_cuda = interpolate_image_by_cuda(image, scale)
    image_interpolated_manually = interpolate_image_manually(image, scale)
    np.testing.assert_array_equal(image_by_cuda, image_interpolated_manually)
    show_image(image, 'Original image')
    show_image(image_by_cuda, 'CUDA interpolation')
    show_image(image_interpolated_manually, 'Manual interpolation')


def get_images(folder: str = './data'):
    images = []
    for root, dirs, files in os.walk(folder):
        images += [imageio.imread(f'{folder}/{file}') for file in files if file.endswith(".png")]
    return images


if __name__ == "__main__":
    for img in get_images():
        calculate_image(img)
