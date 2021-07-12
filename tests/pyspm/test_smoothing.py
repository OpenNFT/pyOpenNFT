import numpy as np
from opennft.smoothing import spm_smooth
from scipy.io import savemat


def test_smoothing(data_path, resl_vol, nii_image_1, smoothed_matlab):

    resl_vol = resl_vol['reslVol_python']
    mat = nii_image_1.affine

    dicom_info_vox = (np.sum(mat[0:3, 0:3] ** 2, axis=0)) ** .5

    gkernel = np.array([5, 5, 5]) / dicom_info_vox
    sm_resl_vol = spm_smooth(resl_vol, gkernel)

    resl_dic = {"smoothed_python": sm_resl_vol}
    savemat(data_path / "smoothed_python.mat", resl_dic)

    np.testing.assert_almost_equal(sm_resl_vol, smoothed_matlab['smReslVol'], decimal=7, err_msg="Not equal")

