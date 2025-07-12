#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_padding').convolve_grayscale_same


if __name__ == '__main__':

    dataset = np.load('/Users/mizeroreine/Desktop/docs/ALU/alu-machine_learning/math/convolutions_and_pooling/animals_1.npz')
    images = dataset['data']
    print(images.shape)

    images_gray = (
        0.2989 * images[:, :, :, 0] +
        0.5870 * images[:, :, :, 1] +
        0.1140 * images[:, :, :, 2]
    )
    print(images_gray.shape)  # (10000, 32, 32)

    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_same(images_gray, kernel)
    print(images_conv.shape)

    plt.imshow(images_gray[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
