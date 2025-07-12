#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
        images (np.ndarray): shape (m, h, w), grayscale images
        kernel (np.ndarray): shape (kh, kw), convolution kernel
        padding (tuple): (ph, pw), padding for height and width

    Returns:
        np.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images with zeros
    padded_images = np.pad(images,((0, 0),(ph, ph),(pw, pw)),mode='constant')

    # Compute output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )

    return output
