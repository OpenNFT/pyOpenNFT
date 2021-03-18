import numpy as np
import pyspm as spm


def spm_reslice(r, flags):

    msk = []
    count = []
    integral = []
    v0 = []
    r0_dim = np.array(r[0]['dim'], dtype='int16')
    r0_mat = np.array(r[0]['mat'], dtype='int16')

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
                ri_dim = r[i]['dim'][0:3]
                ri_mat = r[i]['mat']

                try:
                    tmp_division = np.linalg.solve(r0_mat, ri_mat)
                except np.linalg.LinAlgError as err:
                    # TODO: Something
                    print(err)
                    raise

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

    for i in range(1, len(r)):  # range(0,P.size)

        ri_dim = r[i]['dim'][0:3]
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
                    tmp_division = np.linalg.solve(r[0]['mat'], r[i]['mat'])
                except np.linalg.LinAlgError as err:
                    # TODO: Something
                    print(err)
                    raise
                tmp, y1, y2, y3 = get_mask(np.linalg.inv(tmp_division), x1, x2, x3 + 1, ri_dim, flags['wrap'])

                out_vol = spm.bsplins(r[i]['C'], y1, y2, y3, d)

                if int(flags['mean']):
                    integral[:, :, x3] += nan_to_zero(v[:, :, :x3])

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
