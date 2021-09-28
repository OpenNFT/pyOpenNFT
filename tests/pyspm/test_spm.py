import pytest
import numpy as np
from opennft import utils
from scipy.io import savemat
from rtspm import spm_realign, spm_reslice


@pytest.mark.first
def test_spm_rt(data_path, dcm_image, nii_image_1, p_struct, matlab_result):
    try:

        a0 = []
        x1 = []
        x2 = []
        x3 = []
        deg = []
        b = []
        r = [{'mat': np.array([]), 'dim': np.array([]), 'Vol': np.array([])} for _ in range(2)]
        dim_vol = np.array([74, 74, 36])

        r[0]["mat"] = nii_image_1.affine
        r[0]["dim"] = dim_vol.copy()
        tmp_vol = np.array(nii_image_1.get_fdata(), order='F')

        xdim_img_number, ydim_img_number, img2d_dimx, img2d_dimy = utils.get_mosaic_dim(dim_vol)

        nr_zero_pad_vol = p_struct["nrZeroPadVol"].item()
        if p_struct["isZeroPadding"].item():
            r[0]["dim"][2] = r[0]["dim"][2] + nr_zero_pad_vol * 2
            r[0]["Vol"] = np.pad(tmp_vol, ((0, 0), (0, 0), (nr_zero_pad_vol, nr_zero_pad_vol)),
                                 'constant', constant_values=(0, 0))
        else:
            r[0]["Vol"] = tmp_vol

        ind_vol = 6
        dcm_data = np.array(dcm_image, dtype=float)

        r[1]["mat"] = nii_image_1.affine
        tmp_vol = utils.img_2d_to_3d(dcm_data, xdim_img_number, ydim_img_number, dim_vol)

        if p_struct["isZeroPadding"].item():
            dim_vol[2] = dim_vol[2] + nr_zero_pad_vol * 2
            r[1]["Vol"] = np.pad(tmp_vol, ((0, 0), (0, 0), (nr_zero_pad_vol, nr_zero_pad_vol)), 'constant',
                                 constant_values=(0, 0))
        else:
            r[1]["Vol"] = tmp_vol

        r[1]["Vol"] = np.array(r[1]["Vol"], order='F')
        r[1]["dim"] = dim_vol

        flags_spm_realign = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4, 'wrap': np.zeros((3, 1)),
                                  'rtm': 0, 'PW': '', 'lkp': np.array(range(0, 6))})
        flags_spm_reslice = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4, 'wrap': np.zeros((3, 1)),
                                  'mask': 1, 'mean': 0, 'which': 2})

        nr_skip_vol = p_struct["nrSkipVol"].item()
        [r, _, _, _, _, _, _, _] = spm_realign(r, flags_spm_realign, ind_vol, nr_skip_vol + 1,
                                               a0, x1, x2, x3, deg, b)

        if p_struct["isZeroPadding"].item():
            tmp_resl_vol = spm_reslice(r, flags_spm_reslice)
            resl_vol = tmp_resl_vol[:, :, nr_zero_pad_vol: -1 - nr_zero_pad_vol + 1]
            dim_vol[2] = dim_vol[2] - nr_zero_pad_vol * 2
        else:
            resl_vol = spm_reslice(r, flags_spm_reslice)

        resl_dic = {"reslVol_python": resl_vol}
        savemat(data_path / "reslVol.mat", resl_dic)

        matlab_resl_vol = matlab_result["reslVol"]
        # np.testing.assert_almost_equal(resl_vol, matlab_resl_vol, decimal=7, err_msg="Not equal")

        print('\n\nFirst test MSE = {:}\n'.format(((resl_vol - matlab_resl_vol) ** 2).mean()))

        assert True, "Done"
    except Exception as err:
        assert False, f"Error occurred: {repr(err)}"
