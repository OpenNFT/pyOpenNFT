import numpy as np
from rtspm import spm_matrix, spm_imatrix


def prepare_orth_view(mat, dim):
    # set structure for Display and draw a first overlay
    str_param = {'n': 0, 'bb': [], 'space': np.eye(4, 4), 'centre': np.zeros((1, 3)), 'mode': 1,
                 'area': np.array([0, 0, 1, 1]), 'premul': np.eye(4, 4), 'hld': 1, 'mode_displ': np.zeros((1, 3))}

    temp = np.array([0, 0, 0, 0, np.pi, -np.pi / 2])
    str_param['space'] = spm_matrix.spm_matrix(temp) @ str_param['space']

    # get bounding box and resolution
    if len(str_param['bb']) == 0:
        str_param['max_bb'] = max_bb(mat, dim, str_param['space'], str_param['premul'])
        str_param['bb'] = str_param['max_bb']

    str_param['space'], str_param['bb'] = resolution(mat, str_param['space'], str_param['bb'])

    # Draw at initial location, center of bounding box
    temp = np.vstack((str_param['max_bb'].T, [1, 1]))
    mmcentre = np.mean(str_param['space'] @ temp, 1)
    str_param['centre'] = mmcentre[0:3]
    # Display modes: [Background + Stat + ROIs, Background + Stat, Background + ROIs]
    str_param['mode_displ'] = np.array([0, 0, 1])

    return str_param


def max_bb(mat, dim, space, premul):

    mn = np.array([np.inf] * 3, ndmin=2)
    mx = -mn
    premul = np.linalg.solve(space, premul)
    bb, vx = get_bbox(mat, dim, premul)
    mx = np.vstack((bb, mx)).max(0)
    mn = np.vstack((bb, mx)).min(0)
    bb = np.vstack((mn, mx))

    return bb


def get_bbox(mat, dim, premul):
    p = spm_imatrix(mat)
    vx = p[6:9]

    corners = np.array([
        [1, 1, 1, 1],
        [1, 1, dim[2], 1],
        [1, dim[1], 1, 1],
        [1, dim[1], dim[2], 1],
        [dim[0], 1, 1, 1],
        [dim[0], 1, dim[2], 1],
        [dim[0], dim[1], 1, 1],
        [dim[0], dim[1], dim[2], 1],

    ]).T

    xyz = mat[0:3, :] @ corners

    xyz = premul[0:3, :] @ np.vstack((xyz, np.ones((1, xyz.shape[1]))))

    bb = np.array([
        np.min(xyz, axis=1).T,
        np.max(xyz, axis=1).T
    ])

    return bb, vx


def resolution(mat, space, bb):
    res_default = 1

    temp = (np.sum((mat[0:3, 0:3]) ** 2, axis=0) ** .5)
    res = np.min(np.hstack((res_default, temp)))

    u, s, v = np.linalg.svd(space[0:3, 0:3])
    temp = np.mean(s)
    res = res / temp

    mat = np.diag([res, res, res, 1])

    space = space @ mat
    bb = bb / res

    return space, bb
