#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    Args:
        images: numpy.ndarray (m, h, w, c) containing multiple images
        kernel_shape: tuple (kh, kw) - kernel shape
        stride: tuple (sh, sw) - strides
        mode: str - either 'max' or 'avg'

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1

    pooled = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            img_slice = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(img_slice, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(img_slice, axis=(1, 2))

    return pooled
