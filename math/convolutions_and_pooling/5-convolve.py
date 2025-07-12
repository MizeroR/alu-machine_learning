#!/usr/bin/env python3
import numpy as np

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if c != kc:
        raise ValueError("The number of channels in the image and kernel must match.")

    # Calculate padding
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + (1 if ((h - 1) * sh + kh - h) % 2 else 0)
        pw = int(((w - 1) * sw + kw - w) / 2) + (1 if ((w - 1) * sw + kw - w) % 2 else 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Apply padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Calculate output dimensions
    oh = (padded.shape[1] - kh) // sh + 1
    ow = (padded.shape[2] - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, oh, ow, nc))

    # Perform convolution
    for i in range(oh):
        for j in range(ow):
            img_slice = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            for k in range(nc):
                output[:, i, j, k] = np.sum(img_slice * kernels[:, :, :, k], axis=(1, 2, 3))

    return output
