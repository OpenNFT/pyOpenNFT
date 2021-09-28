import numpy as np
from scipy.special import betainc


def img_2d_to_3d(img2d, xdim_img_number, ydim_img_number, dim3d):
    sl = 0
    vol3d = np.zeros(dim3d)
    for sy in range(0, ydim_img_number):
        for sx in range(0, xdim_img_number):
            if sl > dim3d[2]:
                break
            else:
                vol3d[:, :, sl] = img2d[sy * dim3d[0]: (sy + 1) * dim3d[0], sx * dim3d[0]: (sx + 1) * dim3d[0]]
            vol3d[:, :, sl] = np.rot90(vol3d[:, :, sl], 3)
            sl += 1

    return vol3d


def vol_3d_to_2d(vol3d, sl_nr_img2d_dimx, sl_nr_img2d_dimy, xdim_img_number, ydim_img_number, dim3d):
    sl = 0
    img_2d = np.zeros((ydim_img_number, xdim_img_number))

    for sy in range(0, sl_nr_img2d_dimy):
        for sx in range(0, sl_nr_img2d_dimx):
            if sl > dim3d[2]:
                break
            else:
                img_2d[sy * dim3d[1]:(sy + 1) * dim3d[1], sx * dim3d[0]:(sx + 1) * dim3d[0]] = np.rot90(vol3d[:, :, sl])
            sl += 1

    return img_2d


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


def spm_matrix(p):
    if p.size == 3:
        a = np.eye(4)
        a[0:3, 3] = p[:]
        return a

    q = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    p = np.append(p, q[p.size:12])

    t = np.array([[1, 0, 0, p[0]],
                  [0, 1, 0, p[1]],
                  [0, 0, 1, p[2]],
                  [0, 0, 0, 1]])

    r1 = np.array([[1, 0, 0, 0],
                   [0, np.cos(p[3]), np.sin(p[3]), 0],
                   [0, -np.sin(p[3]), np.cos(p[3]), 0],
                   [0, 0, 0, 1]])

    r2 = np.array([[np.cos(p[4]), 0, np.sin(p[4]), 0],
                   [0, 1, 0, 0],
                   [-np.sin(p[4]), 0, np.cos(p[4]), 0],
                   [0, 0, 0, 1]])

    r3 = np.array([[np.cos(p[5]), np.sin(p[5]), 0, 0],
                   [-np.sin(p[5]), np.cos(p[5]), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    r = r1 @ r2 @ r3

    z = np.array([[p[6], 0, 0, 0],
                  [0, p[7], 0, 0],
                  [0, 0, p[8], 0],
                  [0, 0, 0, 1]])

    s = np.array([[1, p[9], p[10], 0],
                  [0, 1, p[11], 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    a = t @ r @ z @ s

    return a


def spm_imatrix(m):
    r = m[0:3, 0:3]
    c = np.linalg.cholesky(r.T @ r)

    p = np.append(m[0:3, 3].T, np.zeros(3, ))
    p = np.append(p, np.diag(c).T)
    p = np.append(p, np.zeros(3, ))

    if np.linalg.det(r) < 0:
        p[6] = -p[6]

    c = np.linalg.solve(np.diag(np.diag(c)), c)
    p[9:12] = c.flatten()[[3, 6, 7]]
    r0 = spm_matrix(np.append(np.zeros(6, ), p[6:12]))
    r0 = r0[0:3, 0:3]
    r1 = np.linalg.solve(r0.T, r.T).T

    def rang(x):
        return np.minimum(np.maximum(x, -1), 1)

    p[4] = np.arcsin(rang(r1[0, 2]))
    if (np.abs(p[4]) - np.pi / 2) ** 2 < 1e-9:
        p[3] = 0
        p[5] = np.arctan2(-rang(r1[1, 0]), rang(-r1[2, 0] / r1[0, 2]))
    else:
        c = np.cos(p[4])
        p[3] = np.arctan2(rang(r1[1, 2] / c), rang(r1[2, 2] / c))
        p[5] = np.arctan2(rang(r1[0, 1] / c), rang(r1[0, 0] / c))

    return p


def ar_regr(a, input):
    output = np.zeros(input.shape)

    for col in range(0, input.shape[1]):
        for t in range(0, input.shape[0]):
            if t == 0:
                output[t, col] = (1 - a) * input[t, col]
            else:
                output[t, col] = input[t, col] - a * output[t - 1, col]

    return output


def spm_inv_t_cdf(f, v):
    ad = np.array([2, 2])
    rd = np.max(ad)
    az = np.array([[1, 1],
                   [1, 1]])
    rs = np.max(az)
    xa = np.prod(az, 1) > 1

    x = np.zeros(rs)

    md = 0 <= f <= 1 and v > 0

    x[md and f == 0] = -np.inf
    x[md and f == 1] = +np.inf

    ml = (md and v == 1)
    if xa[0]:
        mlf = ml
    else:
        mlf = 1

    x[ml] = np.tan(np.pi * (f - 0.5))

    q = np.nonzero(md and f != 0.5 and 1 > f > 0 != v)[0]
    if q.size == 0:
        return x

    xqpos = f > 0.5
    bq = spm_inv_b_cdf(2 * (xqpos - (xqpos * 2 - 1) * f), v / 2, 0.5)
    x[q] = (xqpos * 2 - 1) * np.sqrt(v / bq - v)

    return x


def spm_inv_b_cdf(f, v, w, tol=1e-8):
    max_it = 1e4

    ad = np.array([2, 2, 2])
    rd = np.max(ad)
    az = np.array([[1, 1],
                   [1, 1],
                   [1, 1]])
    rs = np.max(az)
    xa = np.prod(az, 1) > 1

    x = np.zeros(rs)

    md = 0 <= f <= 1 and v > 0 and w > 0

    x[md and f == 1] = 1

    q = np.nonzero(md and 0 < f < 1)[0]
    if q.size == 0:
        return x

    if xa[0]:
        fq = f
        fq = fq.flatten()
    else:
        fq = f * np.ones((np.max(q.shape), 1))

    if xa[1]:
        vq = v
        vq = vq.flatten()
    else:
        vq = v * np.ones((np.max(q.shape), 1))

    if xa[2]:
        wq = w
        wq = wq.flatten()
    else:
        wq = w * np.ones((np.max(q.shape), 1))

    a = np.zeros((np.max(q.shape), 1))
    fa = a - fq
    b = np.ones((np.max(q.shape), 1))
    fb = b - fq
    i = 0
    xq = a + 0.5
    qq = np.array(range(0, (np.max(q.shape))))

    while qq.size > 0 and i < max_it:
        fxqq = betainc(vq[qq], wq[qq], xq[qq]) - fq[qq]
        mqq = fa[qq] * fxqq > 0

        a[qq[mqq[0]]] = xq[qq[mqq[0]]]
        fa[qq[mqq[0]]] = fxqq[mqq[0]]
        b[qq[not mqq]] = xq[qq[not mqq]]
        fb[qq[not mqq]] = fxqq[not mqq]
        xq[qq] = a[qq] + (b[qq] - a[qq]) / 2
        qq = qq[((b[qq] - a[qq]) > tol)[0]]

        i += 1

    x[q] = xq

    return x


def zscore(x):
    dim = np.nonzero(np.array(x.shape) != 1)[0][0]

    mu = np.mean(x, axis=dim)
    sigma = np.std(x, ddof=1, axis=dim)
    sigma[sigma == 0] = 1
    z = (x - mu) / sigma

    return z

