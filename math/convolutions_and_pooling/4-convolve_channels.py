#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels using a single kernel.

    Args:
        images (np.ndarray): shape (m, h, w, c)
        kernel (np.ndarray): shape (kh, kw, c)
        padding (str or tuple): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        np.ndarray: convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if type(padding) == tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + ((h - 1) * sh + kh - h) % 2
        pw = ((w - 1) * sw + kw - w) // 2 + ((w - 1) * sw + kw - w) % 2
    else:  # 'valid'
        ph = pw = 0

    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            hs = i * sh
            ws = j * sw
            region = padded_images[:, hs:hs + kh, ws:ws + kw, :]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
