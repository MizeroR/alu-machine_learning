#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images."""

import numpy as np

def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs convolution on grayscale images with optional padding and stride.
    
    Args:
        images (np.ndarray): shape (m, h, w), multiple grayscale images
        kernel (np.ndarray): shape (kh, kw), convolution kernel
        padding (str or tuple): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw), strides for height and width

    Returns:
        np.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if type(padding) == tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + ((h - 1) * sh + kh - h) % 2
        pw = ((w - 1) * sw + kw - w) // 2 + ((w - 1) * sw + kw - w) % 2
    else:  # 'valid'
        ph, pw = 0, 0

    # Pad images
    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant'
        )

    # Compute output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, out_h, out_w))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            w_start = j * sw
            region = images_padded[:, h_start:h_start+kh, w_start:w_start+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
