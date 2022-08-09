import pytest
import numpy as np
from rtspm import spm_smooth
from scipy.io import savemat

from opennft.iglm_vol import iglm_vol
from opennft.utils import ar_regr, vol_3d_to_2d, zscore


@pytest.mark.fourth
def test_iglm(data_path, nii_image_1, mc_param, prepr_data, sum_vols, stat_map_2d_matlab):

    mat = nii_image_1.affine
    dicom_info_vox = (np.sum(mat[0:3, 0:3] ** 2, axis=0)) ** .5
    gkernel = np.array([5, 5, 5]) / dicom_info_vox

    main_loop_data = {
        "dimVol": prepr_data["dimVol"],
        "slNrImg2DdimX": prepr_data["slNrImg2DdimX"],
        "slNrImg2DdimY": prepr_data["slNrImg2DdimY"],
        "img2DdimX": prepr_data["img2DdimX"],
        "img2DdimY": prepr_data["img2DdimY"],
        "nrVoxInVol": prepr_data["nrVoxInVol"],
        "nrBasFct": prepr_data["nrBasFct"],
        "numscan": prepr_data["numscan"],
        "spmMaskTh": np.array(prepr_data["spmMaskTh"], ndmin=2).T,
        "basFct": np.array(prepr_data["basFct"], ndmin=2).T,
        "pVal": prepr_data["pVal"],
        "tContr": {
            "pos": int(prepr_data["tContr"]["pos"]),
            "neg": int(prepr_data["tContr"]["neg"])
        },
        "highPass": prepr_data["high_pass"],
        "idx_act_vox_iglm_pos": {},
        "idx_act_vox_iglm_neg": {}
    }

    p = {
        "linRegr": np.array(prepr_data["lin_regr"], ndmin=2).T,
        "mcParam": mc_param["mc_python"],
        "aAR1": 0.2
    }

    stat_map_2d_pos = np.zeros((main_loop_data["img2DdimX"], main_loop_data["img2DdimY"]))
    stat_map_2d_all_pos = np.zeros((150, main_loop_data["img2DdimX"], main_loop_data["img2DdimY"]))

    for i in range(5, 155):

        resl_vol = np.array(np.squeeze(sum_vols["sumVols"][i, :, :, :]), order='F')
        sm_resl_vol = spm_smooth(resl_vol, gkernel)

        ind_vol_norm = i-5

        if ind_vol_norm == 0:

            main_loop_data["smReslVolAR1_1"] = (1 - p["aAR1"]) * sm_resl_vol

        else:

            main_loop_data["smReslVolAR1_1"] = sm_resl_vol - p["aAR1"] * main_loop_data["smReslVolAR1_1"]

        sm_resl_vol = main_loop_data["smReslVolAR1_1"]
        
        main_loop_data, stat_map_2d_pos = prepr_vol(main_loop_data, p, sm_resl_vol, ind_vol_norm+1)
        stat_map_2d_all_pos[ind_vol_norm,:,:] = stat_map_2d_pos
        # tn[ind_vol_norm,:,:] = main_loop_data["tn"] "tn": tn

    savemat(data_path / "stat_map_2d_all_python.mat", {"stat_map_2d_all_python": stat_map_2d_all_pos,
                                                       "stat_map_2d_python": stat_map_2d_pos,})

    np.testing.assert_almost_equal(stat_map_2d_pos, stat_map_2d_matlab['statMap2D'], decimal=7, err_msg="Not equal")


def prepr_vol(main_loop_data, p, sm_resl_vol, ind_iglm):

    dim = main_loop_data["dimVol"]
    img2d_dim_x = main_loop_data["img2DdimX"]
    img2d_dim_y = main_loop_data["img2DdimY"]
    sl_nr_img_2d_dimx = main_loop_data["slNrImg2DdimX"]
    sl_nr_img_2d_dimy = main_loop_data["slNrImg2DdimY"]
    stat_map_2d_pos = np.zeros((img2d_dim_x, img2d_dim_y))

    nr_vox_in_vol = main_loop_data["nrVoxInVol"]
    nr_bas_fct = main_loop_data["nrBasFct"]
    numscan = main_loop_data["numscan"]
    spm_mask_th = main_loop_data["spmMaskTh"]
    bas_fct = main_loop_data["basFct"]

    if ind_iglm != 1:

        p_val = main_loop_data["pVal"]
        t_contr = main_loop_data["tContr"].copy()
        nr_bas_fct_regr = main_loop_data["nrBasFctRegr"]
        cn = main_loop_data["Cn"]
        dn = main_loop_data["Dn"]
        s2n = main_loop_data["s2n"]
        tn = main_loop_data["tn"]
        t_th = main_loop_data["tTh"]
        dynt_th = main_loop_data["dyntTh"]
        stat_map_3d_pos = np.zeros(dim)

    else:

        p_val = main_loop_data["pVal"]
        t_contr = main_loop_data["tContr"].copy()

        nr_high_pass_regr = main_loop_data["highPass"].shape[1]
        nr_mot_regr = 6
        nr_bas_fct_regr = nr_mot_regr + nr_high_pass_regr + 2

        cn = np.zeros((nr_bas_fct + nr_bas_fct_regr, nr_bas_fct + nr_bas_fct_regr))
        dn = np.zeros((nr_vox_in_vol, nr_bas_fct + nr_bas_fct_regr))
        s2n = np.zeros((nr_vox_in_vol, 1))
        tn = {"pos": np.zeros((nr_vox_in_vol, 1)), "neg": np.zeros((nr_vox_in_vol, 1))}
        t_th = np.zeros((numscan, 1))
        dynt_th = 0

        stat_map_vect = np.zeros((nr_vox_in_vol, 1))
        stat_map_3d_pos = np.zeros(dim)
        main_loop_data["statMapVect"] = stat_map_vect
        main_loop_data["statMap3D_pos"] = stat_map_3d_pos

    tmp_regr = zscore(p["mcParam"][0:ind_iglm, :])
    # tmp_regr = p["mcParam"][0:ind_iglm,:]
    tmp_regr = np.hstack((tmp_regr, p["linRegr"][0:ind_iglm]))
    tmp_regr = np.hstack((tmp_regr, main_loop_data["highPass"][0:ind_iglm]))
    tmp_regr = np.hstack((tmp_regr, np.ones((ind_iglm, 1))))

    tmp_regr = ar_regr(p["aAR1"], tmp_regr)

    bas_fct_regr = np.hstack((bas_fct[0:ind_iglm], tmp_regr))

    t_contr["pos"] = np.vstack((t_contr["pos"], np.zeros((nr_bas_fct_regr, 1))))
    t_contr["neg"] = np.vstack((t_contr["neg"], np.zeros((nr_bas_fct_regr, 1))))

    idx_act_vox, dynt_th, t_th, cn, dn, sigma2n, tn, neg_e2n, bn, e2n = iglm_vol(cn, dn, s2n, tn,
                                                                                 sm_resl_vol.flatten(order="F"),
                                                                                 ind_iglm, nr_bas_fct+nr_bas_fct_regr,
                                                                                 t_contr, bas_fct_regr, p_val, dynt_th,
                                                                                 t_th, spm_mask_th)

    main_loop_data["nrBasFctRegr"] = nr_bas_fct_regr
    main_loop_data["Cn"] = cn
    main_loop_data["Dn"] = dn
    main_loop_data["s2n"] = s2n
    main_loop_data["tn"] = tn
    main_loop_data["tTh"] = t_th
    main_loop_data["dyntTh"] = dynt_th

    main_loop_data["idx_act_vox_iglm_pos"].update({ind_iglm: idx_act_vox["pos"]})
    main_loop_data["idx_act_vox_iglm_neg"].update({ind_iglm: idx_act_vox["neg"]})

    if idx_act_vox["pos"].size > 0 and np.max(tn["pos"]) > 0:

        masked_stat_map_vect_pos = tn["pos"][idx_act_vox["pos"]]
        max_t_val_pos = np.max(masked_stat_map_vect_pos)
        stat_map_vect = np.squeeze(masked_stat_map_vect_pos)
        temp_map = stat_map_3d_pos.flatten(order='F')
        temp_map[idx_act_vox["pos"]] = stat_map_vect
        stat_map_3d_pos = np.reshape(temp_map, dim, order='F')

        stat_map_2d_pos = vol_3d_to_2d(stat_map_3d_pos, sl_nr_img_2d_dimx, sl_nr_img_2d_dimy,
                                       img2d_dim_x, img2d_dim_y, dim) / max_t_val_pos
        stat_map_2d_pos *= 255

    return main_loop_data, stat_map_2d_pos
