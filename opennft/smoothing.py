import sys
import numpy as np
import pyspm as spm
from scipy.special import erf


def spm_smooth(input_vol, kernel):
    if kernel.size == 1:
        kernel = np.array([kernel, kernel, kernel])

    smoothed_vol = smooth1(input_vol, kernel)

    return smoothed_vol


def smooth1(input_vol, kernel):
    vox = np.array([1, 1, 1])

    kernel = kernel / vox
    kernel1 = kernel / ((8 * np.log(2)) ** .5)

    x = int(np.round(6 * kernel1[0]))
    x = np.array(range(-x, x + 1), ndmin=2)
    x = spm_smoothkern(kernel[0], x, 1)
    x = x / np.sum(x)

    y = int(np.round(6 * kernel1[1]))
    y = np.array(range(-y, y + 1), ndmin=2)
    y = spm_smoothkern(kernel[1], y, 1)
    y = y / np.sum(y)

    z = int(np.round(6 * kernel1[2]))
    z = np.array(range(-z, z + 1), ndmin=2)
    z = spm_smoothkern(kernel[2], z, 1)
    z = z / np.sum(z)

    i = (x.size - 1) / 2
    j = (y.size - 1) / 2
    k = (z.size - 1) / 2

    smoothed_vol = np.zeros(input_vol.shape, order='F')
    smoothed_vol = spm.conv_vol(input_vol, smoothed_vol, x, y, z, np.array([-i, -j, -k], ndmin=2))

    return smoothed_vol


def spm_smoothkern(fwhm, x, t=1):
    eps = sys.float_info.epsilon
    s = (fwhm / (8 * np.log(2)) ** .5) ** 2 + eps

    if t == 0:

        w1 = 1 / ((2 * s) ** .5)
        krn = 0.5 * (erf(w1 * (x + 0.5)) - erf(w1 * (x - 0.5)))
        krn[krn < 0] = 0

    else:

        w1 = 0.5 * (2 / s) ** .5
        w2 = -0.5 / s
        w3 = (s / 2 / np.pi) ** .5
        krn = 0.5 * (erf(w1 * (x + 1)) * (x + 1) + erf(w1 * (x - 1)) * (x - 1) - 2 * erf(w1 * x) * x) + \
              w3 * (np.exp(w2 * (x + 1) ** 2) + np.exp(w2 * (x - 1) ** 2) - 2 * np.exp(w2 * x ** 2))

        krn[krn < 0] = 0

    return krn
