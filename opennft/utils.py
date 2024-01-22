import numpy as np


def img2d_vol3d(img2d, xdim_img_number, ydim_img_number, dim3d):
    sl = 0
    vol3d = np.zeros(dim3d)
    for sy in range(0, ydim_img_number):
        for sx in range(0, xdim_img_number):
            if sl >= dim3d[2]:
                break
            else:
                vol3d[:, :, sl] = img2d[sy * dim3d[0]: (sy + 1) * dim3d[0], sx * dim3d[0]: (sx + 1) * dim3d[0]]
            vol3d[:, :, sl] = np.rot90(vol3d[:, :, sl], 3)
            sl += 1

    return vol3d


def vol3d_img2d(vol3d, sl_nr_img2d_dimx, sl_nr_img2d_dimy, xdim_img_number, ydim_img_number, dim3d):
    sl = 0
    img2d = np.zeros((ydim_img_number, xdim_img_number))

    for sy in range(0, sl_nr_img2d_dimy):
        for sx in range(0, sl_nr_img2d_dimx):
            if sl >= dim3d[2]:
                break
            else:
                img2d[sy * dim3d[1]:(sy + 1) * dim3d[1], sx * dim3d[0]:(sx + 1) * dim3d[0]] = np.rot90(vol3d[:, :, sl])
            sl += 1

    return img2d


def get_mosaic_dim(dim3d):
    xdim_img_number = round(np.sqrt(dim3d[2]))
    tmp_dim = dim3d[2] - xdim_img_number ** 2

    if tmp_dim == 0:
        ydim_img_number = xdim_img_number
    else:
        if tmp_dim > 0:
            ydim_img_number = xdim_img_number
            xdim_img_number += 1
        else:
            xdim_img_number = xdim_img_number
            ydim_img_number = xdim_img_number

    img2d_dimx = xdim_img_number * dim3d[0]
    img2d_dimy = ydim_img_number * dim3d[0]

    return xdim_img_number, ydim_img_number, img2d_dimx, img2d_dimy


def ar_regr(a, input):
    output = np.zeros(input.shape)

    for col in range(0, input.shape[1]):
        for t in range(0, input.shape[0]):
            if t == 0:
                output[t, col] = (1 - a) * input[t, col]
            else:
                output[t, col] = input[t, col] - a * output[t - 1, col]

    return output


def zscore(x):
    np.seterr(divide='ignore', invalid='ignore')
    dim = np.nonzero(np.array(x.shape) != 1)[0]

    if len(x) == 1:
        return np.array([0])

    if dim.size:
        dim = dim[0]
    else:
        dim = 0

    mu = np.mean(x, axis=dim)
    sigma = np.std(x, ddof=1, axis=dim)
    if sigma.size == 1:
        if sigma == 0 or sigma != sigma:
            sigma = 1
    else:
        sigma[sigma == 0] = 1
    z = (x - mu) / sigma

    return z


def null_space(matrix, rcond=None):
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    m, n = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(m, n)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    nulls = vh[num:, :].T.conj()
    return nulls


def get_mat(dim, hdr):
    analyze_to_dicom = np.vstack(
        (np.hstack((np.diag([1, -1, 1]), np.array([0, (dim[1] - 1), 0], ndmin=2).T)), np.array([0, 0, 0, 1], ndmin=2)))
    analyze_to_dicom = analyze_to_dicom * np.hstack((np.eye(4, 3), np.array([-1, -1, -1, 1], ndmin=2).T))

    vox = np.array([hdr.PixelSpacing[0], hdr.PixelSpacing[1], np.float32(hdr.SpacingBetweenSlices)], ndmin=2)
    pos = np.array(hdr.ImagePositionPatient, ndmin=2)
    orient = np.reshape(np.array(hdr.ImageOrientationPatient), [2, 3]).T
    orient = np.hstack((orient, null_space(orient.T)))
    if np.linalg.det(orient) < 0:
        orient[:, 2] = -orient[:, 2]

    # The image position vector is not correct. In dicom this vector points to
    # the upper left corner of the image. Perhaps it is unlucky that this is
    # calculated in the syngo software from the vector pointing to the center
    # of the slice (keep in mind: upper left slice) with the enlarged FoV.
    dicom_to_patient = np.vstack((np.hstack((orient * np.diagflat(vox), pos.T)), np.array([0, 0, 0, 1], ndmin=2)))
    truepos = dicom_to_patient @ np.append((np.array([hdr.Columns, hdr.Rows]) - dim[0:2]) / 2, [0, 1])

    dicom_to_patient = np.vstack(
        (np.hstack((orient * np.diag(vox), np.array(truepos[0:3], ndmin=2).T)), np.array([0, 0, 0, 1], ndmin=2)))

    patient_to_tal = np.diag([-1, -1, 1, 1])
    mat = patient_to_tal @ dicom_to_patient @ analyze_to_dicom

    return mat
