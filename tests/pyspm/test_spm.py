# -*- coding: utf-8 -*-

from opennft import reslice as rs
from opennft import realign as ra
from opennft.utils import Utils as utils
import numpy as np
import nibabel as nib
from scipy.io import savemat


def test_spm_rt(dcm_image: np.array, nii_image_1: nib.nifti1.Nifti1Image, p_struct: dict, matlab_result: np.array, xs: dict):

    try:

        A0 = []
        x1 = []
        x2 = []
        x3 = []
        wt = []
        deg = []
        b = []
        R = [{'mat': None, 'dim': None, 'Vol': None} for i in range(2)]
        dimVol = np.array([74, 74, 36])

        R[0]["mat"] = nii_image_1.affine
        R[0]["dim"] = dimVol.copy()
        tmpVol = np.array(nii_image_1.get_fdata(), dtype='uint16', order='F')

        slNrImg2DdimX, slNrImg2DdimY, img2DdimX, img2DdimY = utils().getMosaicDim(dimVol)

        nrZeroPadVol = p_struct["nrZeroPadVol"].item()
        if p_struct["isZeroPadding"].item():
            R[0]["dim"][2] = R[0]["dim"][2]+nrZeroPadVol*2
            R[0]["Vol"] = np.pad(tmpVol, ((0,0),(0,0),(nrZeroPadVol,nrZeroPadVol)), 'constant', constant_values=(0, 0))
        else:
            R[0]["Vol"] = tmpVol

        indVol = 6
        dcmData = np.array(dcm_image, dtype=float)

        R[1]["mat"] = nii_image_1.affine
        tmpVol = utils().img2Dvol3D(dcmData, slNrImg2DdimX, slNrImg2DdimY, dimVol)

        if p_struct["isZeroPadding"].item():
            dimVol[2] = dimVol[2]+nrZeroPadVol*2
            R[1]["Vol"] = np.pad(tmpVol, ((0,0),(0,0),(nrZeroPadVol,nrZeroPadVol)), 'constant', constant_values=(0, 0))
        else:
            R[1]["Vol"] = tmpVol

        R[1]["Vol"] = np.array(R[1]["Vol"], dtype='uint16', order='F')
        R[1]["dim"] = dimVol

        flagsSpmRealign = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4, 'wrap': np.zeros((3,1)), 'rtm': 0, 'PW': '', 'lkp': np.array(range(0,6))})
        flagsSpmReslice = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4, 'wrap': np.zeros((3,1)), 'mask': 1, 'mean': 0, 'which': 2})

        # x1 = np.array(xs["x1"], order='F')
        # x2 = np.array(xs["x2"], order='F')
        # x3 = np.array(xs["x3"], order='F')

        nrSkipVol = p_struct["nrSkipVol"].item()
        [R, A0, x1, x2, x3, wt, deg, b, nrIter] = ra.Realign().spm_realign(R, flagsSpmRealign, indVol, nrSkipVol + 1, A0, x1, x2, x3, wt, deg, b)

        if p_struct["isZeroPadding"].item():
            tmp_reslVol = rs.Reslicing().spm_reslice(R, flagsSpmReslice)
            reslVol = tmp_reslVol[:,:, nrZeroPadVol : -1 - nrZeroPadVol +1]
            dimVol[2] = dimVol[2] - nrZeroPadVol * 2
        else:
            reslVol = rs.Reslicing().spm_reslice(R, flagsSpmReslice)

        reslDic = {"reslVol_python": reslVol}
        savemat("data/reslVol.mat", reslDic)

        matlab_reslVol = matlab_result["reslVol"]
        np.testing.assert_almost_equal(reslVol, matlab_reslVol, decimal=7, err_msg="Not equal")

        print('\n\nFirst test MSE = {:}\n'.format(((reslVol - matlab_reslVol)**2).mean()))

        assert True, "Done"
    except Exception as err:
        assert False, f"Error occurred: {repr(err)}"
