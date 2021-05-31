import numpy as np
from opennft.smoothing import spm_smooth
from scipy.io import savemat


def test_smoothing(data_path, resl_vol, nii_image_1, smoothed_matlab):

    resl_vol = resl_vol['reslVol_python']
    matTemplMotCorr = nii_image_1.affine

    dicomInfoVox = (np.sum(matTemplMotCorr[0:3,0:3] ** 2,axis=0)) ** .5

    gKernel = np.array([5, 5, 5]) / dicomInfoVox
    smReslVol = spm_smooth(resl_vol, gKernel)

    resl_dic = {"smoothed_python": smReslVol}
    savemat(data_path / "smoothed_python.mat", resl_dic)

    np.testing.assert_almost_equal(smReslVol, smoothed_matlab['smReslVol'], decimal=7, err_msg="Not equal")

