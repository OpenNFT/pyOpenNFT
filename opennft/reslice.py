# Rigid body reslicing of images
# FORMAT spm_reslice(P,flags)
#
# P      - matrix or cell array of filenames {one string per row}
#          All operations are performed relative to the first image.
#          ie. Coregistration is to the first image, and resampling
#          of images is into the space of the first image.
#
# flags  - a structure containing various options.  The fields are:
#
#         mask   - mask output images (true/false) [default: true]
#                  To avoid artifactual movement-related variance the
#                  realigned set of images can be internally masked, within
#                  the set (i.e. if any image has a zero value at a voxel
#                  than all images have zero values at that voxel). Zero
#                  values occur when regions 'outside' the image are moved
#                  'inside' the image during realignment.
#
#         mean   - write mean image (true/false) [default: true]
#                  The average of all the realigned scans is written to
#                  an image file with 'mean' prefix.
#
#         interp - the B-spline interpolation method [default: 1]
#                  Non-finite values result in Fourier interpolation. Note
#                  that Fourier interpolation only works for purely rigid
#                  body transformations. Voxel sizes must all be identical
#                  and isotropic.
#
#         which  - values of 0, 1 or 2 are allowed [default: 2]
#                  0   - don't create any resliced images.
#                        Useful if you only want a mean resliced image.
#                  1   - don't reslice the first image.
#                        The first image is not actually moved, so it may
#                        not be necessary to resample it.
#                  2   - reslice all the images.
#                  If which is a 2-element vector, flags.mean will be set
#                  to flags.which(2).
#
#         wrap   - three values of either 0 or 1, representing wrapping in
#                  each of the dimensions. For fMRI, [1 1 0] would be used.
#                  For PET, it would be [0 0 0]. [default: [0 0 0]]
#
#         prefix - prefix for resliced images [default: 'r']
#
#__________________________________________________________________________
#
# The spatially realigned images are written to the original subdirectory
# with the same (prefixed) filename. They are all aligned with the first.
#
# Inputs:
# A series of images conforming to SPM data format (see 'Data Format'). The
# relative displacement of the images is stored in their header.
#
# Outputs:
# The routine uses information in their headers and writes the realigned
# image files to the same subdirectory with a prefix.
#__________________________________________________________________________
# Copyright (C) 1999-2011 Wellcome Trust Centre for Neuroimaging
#
# John Ashburner
# $Id: spm_reslice.m 5929 2014-03-27 14:47:40Z guillaume $
#__________________________________________________________________________
# Adopted for OpenNFT by Yury Koush and John Ashburner.
# Copyright (C) 2016-2019 OpenNFT.org
#
#__________________________________________________________________________
#
# The headers of the images contain a 4x4 affine transformation matrix 'M',
# usually affected by the `realignment' and `coregistration' modules.
# What these matrices contain is a mapping from the voxel coordinates
# (x1,x2,x3) (where the first voxel is at coordinate (1,1,1)), to
# coordinates in millimeters (y1,y2,y3).
#
# y1 = m[0][0] * x1 + m[0][1] * x2 + (m[0][2] * x3 + m[0][3])
# y2 = m[1][0] * x1 + m[1][1] * x2 + (m[1][2] * x3 + m[1][3])
# y3 = m[2][0] * x1 + m[2][1] * x2 + (m[2][2] * x3 + m[2][3])
#
# Assuming that image1 has a transformation matrix M1, and image2 has a
# transformation matrix M2, the mapping from image1 to image2 is: M2\M1
# (ie. from the coordinate system of image1 into millimeters, followed
# by a mapping from millimeters into the space of image2).
#
# Several spatial transformations (realignment, coregistration,
# normalisation) can be combined into a single operation (without the
# necessity of resampling the images several times).
#__________________________________________________________________________
#
# Refs:
#
# Friston KJ, Williams SR, Howard R Frackowiak RSJ and Turner R (1995)
# Movement-related effect in fMRI time-series.  Mag. Res. Med. 35:346-355
#
# W. F. Eddy, M. Fitzgerald and D. C. Noll (1996) Improved Image
# Registration by Using Fourier Interpolation. Mag. Res. Med. 36(6):923-931
#
# R. W. Cox and A. Jesmanowicz (1999)  Real-Time 3D Image Registration
# for Functional MRI. Mag. Res. Med. 42(6):1014-1018
#__________________________________________________________________________

import numpy as np
import pyspm as spm
from opennft.errors import SpmError


def spm_reslice(r, flags):

    msk = []
    count = []
    integral = []
    v0 = []
    r0_dim = np.array(r[0]['dim'])
    r0_mat = np.array(r[0]['mat'])

    if int(flags['mask']) or int(flags['mean']):

        temp_x1 = np.transpose(np.array(range(1, r0_dim[0] + 1), ndmin=2))
        x1 = np.tile(temp_x1, (1, r0_dim[1]))
        temp_x2 = np.transpose(np.array(range(1, r0_dim[1] + 1), ndmin=2))
        x2 = np.transpose(np.tile(temp_x2, (1, r0_dim[1])))

        if int(flags['mean']):
            count = np.zeros(r0_dim)
            integral = np.zeros(r0_dim)

        if int(flags['mask']):
            msk = [[] for _ in range(r0_dim[2])]  # [None]*P['dim'][0][2]

        for x3 in range(0, r0_dim[2]):
            tmp = np.zeros((r0_dim[0], r0_dim[1]))
            for i in range(0, len(r)):
                ri_dim = r[i]['dim'][:3]
                ri_mat = r[i]['mat']

                try:
                    tmp_division = np.linalg.solve(r0_mat, ri_mat)
                except np.linalg.LinAlgError as err:
                    raise SpmError("R0 and R1 division error in reslice") from err

                temp_tmp, y1, y2, y3 = get_mask(np.linalg.inv(tmp_division), x1, x2, x3 + 1, ri_dim, flags['wrap'])
                tmp += temp_tmp

            if int(flags['mask']):
                msk[x3] = np.argwhere(tmp.reshape(tmp.size, 1) != len(r))[:, 0]

            if int(flags['mean']):
                count[:, :, x3] = tmp

    x1, x2 = np.mgrid[1:r0_dim[0] + 1, 1:r0_dim[1] + 1]

    temp_d = np.array([1, 1, 1]) * int(flags['interp'])
    d = np.hstack((temp_d.T, np.squeeze(flags['wrap'])))
    d = np.array(d, ndmin=2).T
    r0_mat = np.array(r[0]['mat'])

    for i in range(1, len(r)):  # range(0,P.size)

        ri_dim = r[i]['dim'][:3]
        ri_mat = r[i]['mat']

        if (i > 1 and int(flags['which']) == 1) or int(flags['which']) == 2:
            write_vol = 1
        else:
            write_vol = 0
        if write_vol or int(flags['mean']):
            read_vol = 1
        else:
            read_vol = 0

        if read_vol:

            v = np.zeros(r0_dim)
            for x3 in range(0, r0_dim[2]):
                try:
                    tmp_division = np.linalg.solve(r0_mat, ri_mat)
                except np.linalg.LinAlgError as err:
                    raise SpmError("R0 and R1 division error in reslice") from err

                tmp, y1, y2, y3 = get_mask(np.linalg.inv(tmp_division), x1, x2, x3 + 1, ri_dim, flags['wrap'])

                out_vol = spm.bsplins(r[i]['C'], y1, y2, y3, d)

                if int(flags['mean']):
                    integral[:, :, x3] += nan_to_zero(v[:, :, x3])

                if int(flags['mask']):
                    tmp = out_vol
                    tmp[msk[x3]] = 0
                    out_vol = tmp

                v[:, :, x3] = out_vol.reshape(y1.shape)

            if write_vol:
                v0 = v

    return v0


def get_mask(m, x1, x2, x3, dim, wrp):

    tiny = 5e-2  # From spm_vol_utils.cpp
    y1 = m[0][0] * x1 + m[0][1] * x2 + (m[0][2] * x3 + m[0][3])
    y2 = m[1][0] * x1 + m[1][1] * x2 + (m[1][2] * x3 + m[1][3])
    y3 = m[2][0] * x1 + m[2][1] * x2 + (m[2][2] * x3 + m[2][3])
    mask = np.array([True] * y1.size).reshape(y1.shape)
    if wrp[0] == 0:
        mask = np.logical_and(np.logical_and(mask, (y1 >= (1 - tiny))), (y1 <= (dim[0] + tiny)))
    if wrp[1] == 0:
        mask = np.logical_and(np.logical_and(mask, (y2 >= (1 - tiny))), (y2 <= (dim[1] + tiny)))
    if wrp[2] == 0:
        mask = np.logical_and(np.logical_and(mask, (y3 >= (1 - tiny))), (y3 <= (dim[2] + tiny)))

    return mask, y1, y2, y3


def nan_to_zero(vi):
    return np.nan_to_num(vi, copy=True, nan=0, posinf=0, neginf=0)
