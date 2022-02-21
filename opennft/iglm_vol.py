import numpy as np
from rtspm import spm_inv_t_cdf
from scipy import linalg


def iglm_vol(cn, dn, sigma2n, tn, yn, n, nr_bas_fct, contr, bas_fct, p_val, rec_th, t_th, spm_mask_th):
    yn = np.array(yn, ndmin=2).T

    p_th = 1 - p_val
    df = n - nr_bas_fct
    if n > nr_bas_fct + 2:
        t_th[n-1] = spm_inv_t_cdf(p_th, df)

    ft = np.array(bas_fct[n-1, :], ndmin=2).T
    dn = dn + yn @ ft.T

    fn = ft @ ft.T
    cn = ((n - 1) / n) * cn + fn / n
    sigma2n += yn * yn
    bn = np.zeros(dn.shape)
    e2n = np.zeros(sigma2n.shape)

    try:
        nn = linalg.cholesky(cn)
        p = 0
    except np.linalg.LinAlgError:
        nn = []
        p = 1

    if p == 0 and n > nr_bas_fct + 2:

        inv_nn = linalg.inv(nn.T)
        an = (dn @ inv_nn.T) / n
        bn = an @ inv_nn
        e2n = n / df * (sigma2n / n - np.array(np.sum(an * an, 1), ndmin=2).T)

        neg_e2n = np.nonzero(e2n < 0)[0]
        if neg_e2n.size > 0:
            e2n[neg_e2n] = np.abs(e2n[neg_e2n])

        zero_e2n = np.nonzero(e2n == 0)[0]
        if zero_e2n.size > 0:
            e2n[zero_e2n] = 1e10

        eq_contr = inv_nn @ contr["pos"]
        tn["pos"] = (an @ eq_contr) / np.sqrt(e2n / n * (eq_contr.T @ eq_contr))
        eq_contr = inv_nn @ contr["neg"]
        tn["neg"] = (an @ eq_contr) / np.sqrt(e2n / n * (eq_contr.T @ eq_contr))

    else:
        neg_e2n = np.array([])

    iglm_act_vox = {"pos": np.nonzero(tn["pos"] > t_th[n-1])[0],
                    "neg": np.nonzero(tn["neg"] > t_th[n-1])[0]}

    rec_th += spm_mask_th[n-1] ** 2
    spm_act_vox = sigma2n > rec_th
    idx_spm_act = np.nonzero(sigma2n > rec_th)[0]

    tn["pos"] = tn["pos"] * spm_act_vox
    tn["neg"] = tn["neg"] * spm_act_vox
    idx_act_vox = {"pos": np.intersect1d(iglm_act_vox["pos"], idx_spm_act),
                   "neg": np.intersect1d(iglm_act_vox["neg"], idx_spm_act)}

    return idx_act_vox, rec_th, t_th, cn, dn, sigma2n, tn, neg_e2n, bn, e2n
