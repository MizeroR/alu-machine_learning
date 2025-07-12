#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad the images with 0s
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    # Prepare the output
    output = np.zeros((m, h, w))

    # Perform convolution using only 2 loops
    for i in range(h):
        for j in range(w):
            # Extract the current region for all m images
            region = padded_images[:, i:i + kh, j:j + kw]
            # Perform element-wise multiplication and sum
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
