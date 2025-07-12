#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (np.ndarray): shape (m, h, w) multiple grayscale images.
        kernel (np.ndarray): shape (kh, kw), the convolution kernel.

    Returns:
        np.ndarray: the convolved images with shape (m, h-kh+1, w-kw+1).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                images[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )

    return output