# Estimation of within modality rigid body movement parameters
# FORMAT P = realign(P,flags)
#
# P     - char array of filenames
#         All operations are performed relative to the first image.
#         ie. Coregistration is to the first image, and resampling
#         of images is into the space of the first image.
#         For multiple sessions, P should be a cell array, where each
#         cell should be a matrix of filenames.
#
# flags - a structure containing various options.  The fields are:
#         quality  - Quality versus speed trade-off.  Highest quality (1)
#                    gives most precise results, whereas lower qualities
#                    gives faster realignment.
#                    The idea is that some voxels contribute little to
#                    the estimation of the realignment parameters.
#                    This parameter is involved in selecting the number
#                    of voxels that are used.
#
#         fwhm     - The FWHM of the Gaussian smoothing kernel (mm) applied
#                    to the images before estimating the realignment
#                    parameters.
#
#         sep      - the default separation (mm) to sample the images.
#
#         rtm      - Register to mean.  If field exists then a two pass
#                    procedure is to be used in order to register the
#                    images to the mean of the images after the first
#                    realignment.
#
#         wrap     - Directions in the volume whose values should wrap
#                    around in. For example, in MRI scans, the images wrap
#                    around in the phase encode direction, so (e.g.) the
#                    subject's nose may poke into the back of the subject's
#                    head.
#
#         PW       -  a filename of a weighting image (reciprocal of
#                    standard deviation).  If field does not exist, then
#                    no weighting is done.
#
#         interp   - B-spline degree used for interpolation
#
#         graphics - display coregistration outputs
#                    default: ~spm('CmdLine')

import numpy as np
import pyspm as spm
from opennft import utils


def spm_realign(r, flags, ind_vol, ind_first_vol, a0, x1, x2, x3, wt, deg, b):

    f_nfb = True

    lkp = flags["lkp"]
    # Note: Matlab lkp starts from 1, check input flags
    if lkp[0] == 1:
        lkp -= 1

    if ind_vol == ind_first_vol:

        template_mat = r[0]["mat"][0:3, 0:3]
        skip = np.sqrt(np.sum(template_mat ** 2, axis=0)) ** (-1) * flags["sep"]
        d = r[0]["dim"][0:3]

        # Note: Python random number generation process differs from matlab
        # This is why the result is not identical to Matlab SPM
        rng = np.random.default_rng(0)
        if d[2] < 3:
            lkp = np.array([0, 1, 5])
            x1, x2, x3 = np.mgrid[1:d[0] - 0.5:skip[0], 1:d[1] - 0.5:skip[1], 1:d[2]:skip[2]]
            x1 = x1 + rng.random(x1.shape) * 0.5
            x2 = x2 + rng.random(x2.shape) * 0.5
        else:
            x1, x2, x3 = np.mgrid[1:d[0] - 0.5:skip[0], 1:d[1] - 0.5:skip[1], 1:d[2] - 0.5:skip[2]]
            x1 = x1 + rng.random(x1.shape) * 0.5
            x2 = x2 + rng.random(x2.shape) * 0.5
            x3 = x3 + rng.random(x3.shape) * 0.5

        x1 = x1.reshape([x1.size, 1])
        x2 = x2.reshape([x2.size, 1])
        x3 = x3.reshape([x3.size, 1])

        wt = np.array([])

        # Compute rate of change of chi2 w.r.t changes in parameters(matrix A)
        # ----------------------------------------------------------------------
        v, r[0]["C"] = smooth_vol(r[0], flags["interp"], flags["wrap"], flags["fwhm"])
        temp_d = np.array([1, 1, 1]) * int(flags['interp'])
        deg = np.hstack((temp_d.T, np.squeeze(flags['wrap'])))
        deg = np.array(deg, ndmin=2).T

        g, d_g1, d_g2, d_g3 = spm.bsplins_multi(v, x1, x2, x3, deg)
        a0 = make_a(r[0]["mat"], x1, x2, x3, d_g1, d_g2, d_g3, wt, lkp)

        b = g
        if wt.size > 0:
            b = b * wt

    # -Loop over images
    # --------------------------------------------------------------------------
    # control over accuracy and number of iterations
    if f_nfb:
        th_acc = 0.01
        nr_iter = 10
    else:
        # SPM defualt
        th_acc = 1e-8
        nr_iter = 64

    v, r[1]["C"] = smooth_vol(r[1], flags["interp"], flags["wrap"], flags["fwhm"])

    d = np.array(v.shape)
    ss = np.inf
    countdown = -1

    fix_a0 = []
    iteration = 1
    for iteration in range(1, nr_iter + 1):
        y1, y2, y3 = coords(np.zeros((6, 1)), r[0]["mat"], r[1]["mat"], x1, x2, x3)
        msk = np.nonzero((y1 >= 1) & (y1 <= d[0]) & (y2 >= 1) & (y2 <= d[1]) & (y3 >= 1) & (y3 <= d[2]))
        msk = msk[0]
        if msk.size < 32:
            error_message(r[1])
        f = spm.bsplins(v, y1[msk], y2[msk], y3[msk], deg)
        if wt.size > 0:
            f = f * wt[msk]

        if f_nfb:
            if iteration == 1:
                fix_a0 = a0.T @ a0

        a = a0[msk, :]
        b1 = b[msk]
        sc = np.sum(b1) / np.sum(f)
        b1 = b1 - np.squeeze(f * sc)
        if not f_nfb:
            soln = np.linalg.solve((a.T @ a), (a.T @ b1))
        else:
            soln = np.linalg.solve(fix_a0, (a.T @ b1))

        p = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=float)
        p[lkp] += soln.T
        r[1]["mat"] = np.linalg.solve(utils.spm_matrix(p), r[1]["mat"])

        pss = ss
        ss = np.sum(b1 ** 2) / b1.size
        if ((pss - ss) / pss < th_acc) and (countdown == -1):
            countdown = 2
        if not countdown == -1:
            if countdown == 0:
                break
            countdown -= 1

    nr_iter = iteration

    return r, a0, x1, x2, x3, wt, deg, b, nr_iter


def coords(p, m1, m2, x1, x2, x3):

    m = np.linalg.inv(m2) @ np.linalg.inv(utils.spm_matrix(p)) @ m1
    y1 = m[0, 0] * x1 + m[0, 1] * x2 + m[0, 2] * x3 + m[0, 3]
    y2 = m[1, 0] * x1 + m[1, 1] * x2 + m[1, 2] * x3 + m[1, 3]
    y3 = m[2, 0] * x1 + m[2, 1] * x2 + m[2, 2] * x3 + m[2, 3]

    return y1, y2, y3


def smooth_vol(r, hld, wrp, fwhm):

    s = np.sqrt(np.sum(r["mat"][0:3, 0:3] ** 2, axis=0)) ** (-1) * (fwhm / np.sqrt(8 * np.log(2)))

    x = round(6 * s[0])
    x = np.array(range(-x, x + 1), ndmin=2)

    y = round(6 * s[1])
    y = np.array(range(-y, y + 1), ndmin=2)

    z = round(6 * s[2])
    z = np.array(range(-z, z + 1), ndmin=2)

    x = np.exp(-x ** 2 / (2 * s[0] ** 2))
    y = np.exp(-y ** 2 / (2 * s[1] ** 2))
    z = np.exp(-z ** 2 / (2 * s[2] ** 2))

    x = x / np.sum(x)
    y = y / np.sum(y)
    z = z / np.sum(z)

    i = (x.size - 1) / 2
    j = (y.size - 1) / 2
    k = (z.size - 1) / 2

    temp_d = np.array([1, 1, 1], ndmin=2) * int(hld)
    d = np.hstack((temp_d.T, np.array(wrp, ndmin=2)))

    coef = spm.bsplinc(r["Vol"], d)
    coef = coef.reshape(r["dim"])

    v = np.zeros(coef.shape, order='F')
    v = spm.conv_vol(coef, v, x, y, z, np.array([-i, -j, -k], ndmin=2))

    return v, coef


def make_a(m, x1, x2, x3, dg1, dg2, dg3, wt, lkp):

    # Matrix of rate of change of weighted difference w.r.t.parameter changes
    p0 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=float)
    a = np.zeros((x1.size, lkp.size))
    for i in range(0, lkp.size):
        pt = p0.copy()
        pt[lkp[i]] = pt[i] + 1e-6
        y1, y2, y3 = coords(pt, m, m, x1, x2, x3)
        tmp = np.sum(np.array([y1 - x1, y2 - x2, y3 - x3]).squeeze() * np.array([dg1, dg2, dg3]), 0,
                     keepdims=True).T / (-1e-6)
        if wt.size > 0:
            a[:, i] = tmp * wt
        else:
            a[:, i] = tmp.T

    return a


def error_message(r):
    print('There is not enough overlap in the images to obtain a solution. Offending image: {:} \n'.format(r.fname))
