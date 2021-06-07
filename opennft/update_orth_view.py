import numpy as np
import pyspm as spm
from scipy import linalg


def update_orth_view(str_param, vol, mat):
    bb = str_param['bb']
    dims = np.squeeze(np.round(np.diff(bb, axis=0).T + 1))
    _is = np.linalg.inv(str_param['space'])
    cent = _is[0:3, 0:3] @ str_param['centre'] + _is[0:3, 3]

    m = np.array(np.linalg.solve(str_param['space'], str_param['premul']) @ mat, order='F')
    tm0 = np.array([
        [1, 0, 0, -bb[0, 0] + 1],
        [0, 1, 0, -bb[0, 1] + 1],
        [0, 0, 1, -cent[2]],
        [0, 0, 0, 1],

    ])
    td = np.array([dims[0], dims[1]], dtype=int, order='F')

    cm0 = np.array([
        [1, 0, 0, -bb[0, 0] + 1],
        [0, 0, 1, -bb[0, 2] + 1],
        [0, 1, 0, -cent[1]],
        [0, 0, 0, 1],
    ])
    cd = np.array([dims[0], dims[2]], dtype=int, order='F')

    if str_param['mode'] == 0:
        sm0 = np.array([
            [0, 0, 1, -bb[0, 2] + 1],
            [0, 1, 0, -bb[0, 1] + 1],
            [1, 0, 0, -cent[0]],
            [0, 0, 0, 1],
        ])
        sd = np.array([dims[2], dims[1]], dtype=int, order='F')
    else:
        sm0 = np.array([
            [0, -1, 0, +bb[1, 1] + 1],
            [0, 0, 1, -bb[0, 2] + 1],
            [1, 0, 0, -cent[0]],
            [0, 0, 0, 1],
        ])
        sd = np.array([dims[1], dims[2]], dtype=int, order='F')

    coord_param = {'tm0': tm0, 'cm0': cm0, 'sm0': sm0, 'td': td, 'cd': cd, 'sd': sd}

    imgt, imgc, imgs = get_orth_vol(coord_param, vol, m)

    imgt = np.nan_to_num(imgt)
    imgc = np.nan_to_num(imgc)
    imgs = np.nan_to_num(imgs)

    imgt = ((imgt / np.max(imgt)) * 255).astype(np.uint8)
    imgc = ((imgc / np.max(imgc)) * 255).astype(np.uint8)
    imgs = ((imgs / np.max(imgs)) * 255).astype(np.uint8)

    return imgt, imgc, imgs


def get_orth_vol(coord_param, vol, m):
    temp = np.array([0, np.nan], order='F')

    mat_t = np.array(linalg.inv(coord_param['tm0'] @ m), order='F')
    imgt = np.zeros((coord_param['td'][0], coord_param['td'][1]), order='F')
    spm.slice_vol(vol, imgt, mat_t, temp)
    imgt = imgt.T

    mat_c = np.array(linalg.inv(coord_param['cm0'] @ m), order='F')
    imgc = np.zeros((coord_param['cd'][0], coord_param['cd'][1]), order='F')
    spm.slice_vol(vol, imgc, mat_c, temp)
    imgc = imgc.T

    mat_s = np.array(linalg.inv(coord_param['sm0'] @ m), order='F')
    imgs = np.zeros((coord_param['sd'][0], coord_param['sd'][1]), order='F')
    spm.slice_vol(vol, imgs, mat_s, temp)
    imgs = np.fliplr(imgs.T)

    return imgt, imgc, imgs
