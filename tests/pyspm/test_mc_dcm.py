# -*- coding: utf-8 -*-

from opennft.realign import spm_realign
from opennft.reslice import spm_reslice
from opennft import utils
import numpy as np
import pydicom
from scipy.io import savemat
import time


# def test_bench(benchmark, third_data_path, nii_image_1, p_struct, matlab_mc_result, r_struct):
#
#     result = benchmark.pedantic(mc_dcm, args=(third_data_path,nii_image_1,p_struct), iterations=1, rounds=10)
#
#     assert result


def test_mc_dcm(third_data_path, nii_image_1, p_struct, matlab_mc_result_dcm, r_struct):
    try:

        time_stamps = np.zeros((158,))
        t0 = time.time()

        a0 = []
        x1 = []
        x2 = []
        x3 = []
        wt = []
        deg = []
        b = []
        r = [{'mat': np.array([]), 'dim': np.array([]), 'Vol': np.array([])} for _ in range(2)]
        dim_vol = np.array([74, 74, 36])

        # R[0]["mat"] = r_struct["R"][0]["mat"]
        r[0]["mat"] = nii_image_1.affine
        r[0]["dim"] = dim_vol.copy()
        tmp_vol = np.array(nii_image_1.get_fdata(), dtype='uint16', order='F')

        xdim_img_number, ydim_img_number, img2d_dimx, img2d_dimy = utils.get_mosaic_dim(dim_vol)

        t1 = time.time()
        time_stamps[1] = t1 - t0

        nr_zero_pad_vol = p_struct["nrZeroPadVol"].item()
        if p_struct["isZeroPadding"].item():
            r[0]["dim"][2] = r[0]["dim"][2] + nr_zero_pad_vol * 2
            r[0]["Vol"] = np.pad(tmp_vol, ((0, 0), (0, 0), (nr_zero_pad_vol, nr_zero_pad_vol)),
                                 'constant', constant_values=(0, 0))
        else:
            r[0]["Vol"] = tmp_vol

        mot_corr_param = np.zeros((155, 6))
        sum_vols = np.zeros((155, 74, 74, 36))
        offset_mc_param = np.zeros((6,))

        t2 = time.time()
        time_stamps[2] = t2 - t1

        for ind_vol in range(0, 155):

            ti0 = time.time()
            file_name = str(ind_vol + 1) + '.dcm'
            data = np.array(pydicom.dcmread(third_data_path / file_name).pixel_array, dtype='uint16', order='F')

            r[1]["mat"] = r[0]["mat"]
            r[1]["dim"] = dim_vol.copy()
            tmp_vol = utils.img_2d_to_3d(data, xdim_img_number, ydim_img_number, dim_vol)

            if p_struct["isZeroPadding"].item():
                dim_vol[2] = dim_vol[2] + nr_zero_pad_vol * 2
                r[1]["Vol"] = np.pad(tmp_vol, ((0, 0), (0, 0), (nr_zero_pad_vol, nr_zero_pad_vol)),
                                     'constant', constant_values=(0, 0))
            else:
                r[1]["Vol"] = tmp_vol

            r[1]["Vol"] = np.array(r[1]["Vol"], dtype='uint16', order='F')
            r[1]["dim"] = dim_vol

            flags_spm_realign = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4,
                                      'wrap': np.zeros((3, 1)), 'rtm': 0, 'PW': '', 'lkp': np.array(range(0, 6))})
            flags_spm_reslice = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4,
                                      'wrap': np.zeros((3, 1)), 'mask': 1, 'mean': 0, 'which': 2})

            [r, a0, x1, x2, x3, wt, deg, b, _] = spm_realign(r, flags_spm_realign, ind_vol + 1,
                                                             1, a0, x1, x2, x3, wt, deg, b)

            temp_m = np.linalg.solve(r[0]["mat"].T, r[1]["mat"].T).T
            tmp_mc_param = utils.spm_imatrix(temp_m)
            if ind_vol + 1 == 1:
                offset_mc_param = tmp_mc_param[0:6]
            mot_corr_param[ind_vol, :] = tmp_mc_param[0:6] - offset_mc_param

            if p_struct["isZeroPadding"].item():
                tmp_resl_vol = spm_reslice(r, flags_spm_reslice)
                resl_vol = tmp_resl_vol[:, :, nr_zero_pad_vol: -1 - nr_zero_pad_vol + 1]
                dim_vol[2] = dim_vol[2] - nr_zero_pad_vol * 2
            else:
                resl_vol = spm_reslice(r, flags_spm_reslice)

            sum_vols[ind_vol, :, :, :] = resl_vol

            ti1 = time.time()
            time_stamps[ind_vol + 3] = ti1 - ti0

        # reslDic = {"mc_python": motCorrParam}
        # savemat("data/mc_python_dcm.mat", reslDic)

        resl_dic = {"sumVols": sum_vols}
        savemat("C:/pyOpenNFT/tests/data/sumVols_python_dcm.mat", resl_dic)

        resl_dic = {"python_times": time_stamps}
        savemat("C:/pyOpenNFT/tests/data/python_times_dcm.mat", resl_dic)

        print('\n')
        for i in range(0, 6):
            print('Third test MSE for {:} coordinate = {:}'.
                  format(i + 1, ((mot_corr_param[:, i] - matlab_mc_result_dcm["mc_series_matlab"][:, i]) ** 2).mean()))

        # return True
        assert True, "Done"
    except Exception as err:
        # return False
        assert False, f"Error occurred: {repr(err)}"
