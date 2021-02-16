# -*- coding: utf-8 -*-

from opennft import reslice as rs
from opennft import realign as ra
from opennft import img2Dvol3D as i2v3
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat


def test_spm_rt(dcm_image: np.array, main_loop_data: dict, p_struct: dict, matlab_result: np.array, r_struct : dict, dcmData_struct : np.array, tmpVol_struct : np.array, flags_realign : dict, flags_reslice : dict):

    try:

        A0 = []
        x1 = []
        x2 = []
        x3 = []
        wt = []
        deg = []
        b = []
        R = [{'mat': None, 'dim': None, 'Vol': None} for i in range(2)]
        R[0]["mat"] = main_loop_data["matTemplMotCorr"].item()
        R[0]["dim"] = main_loop_data["dimTemplMotCorr"].item()
        R[0]["Vol"] = np.array(main_loop_data["imgVolTempl"].item(), dtype=float)

        matVol = main_loop_data["matVol"].item()
        dimVol = main_loop_data["dimVol"].item()
        slNrImg2DdimX = main_loop_data["slNrImg2DdimX"].item()
        slNrImg2DdimY = main_loop_data["slNrImg2DdimY"].item()

        indVol = 6

        dcmData = np.array(dcm_image, dtype=float)
        np.testing.assert_almost_equal(dcmData, dcmData_struct["dcmData"], decimal=7, err_msg="Not equal")

        R[1]["mat"] = matVol
        tmpVol = i2v3.img2Dvol3D().img2Dvol3D(dcmData, slNrImg2DdimX, slNrImg2DdimY, dimVol)
        np.testing.assert_almost_equal(tmpVol, tmpVol_struct["tmpVol"], decimal=7, err_msg="Not equal")

        nrZeroPadVol = p_struct["nrZeroPadVol"].item()
        if p_struct["isZeroPadding"].item():
            dimVol[2] = dimVol[2]+nrZeroPadVol*2
            R[1]["Vol"] = np.pad(tmpVol, ((0,0),(0,0),(nrZeroPadVol,nrZeroPadVol)), 'constant', constant_values=(0, 0))
        else:
            R[1]["Vol"] = tmpVol

        R[1]["dim"] = dimVol

        # R = r_struct["R"]
        # R[0]["C"] = np.array([])
        # R[1]["C"] = np.array([])

        np.testing.assert_almost_equal(R[0]["mat"], r_struct["R"][0]["mat"], decimal=7, err_msg="Not equal")
        np.testing.assert_almost_equal(R[1]["mat"], r_struct["R"][1]["mat"], decimal=7, err_msg="Not equal")
        np.testing.assert_almost_equal(R[0]["Vol"], r_struct["R"][0]["Vol"], decimal=7, err_msg="Not equal")
        np.testing.assert_almost_equal(R[1]["Vol"], r_struct["R"][1]["Vol"], decimal=7, err_msg="Not equal")
        np.testing.assert_almost_equal(R[0]["dim"], r_struct["R"][0]["dim"], decimal=7, err_msg="Not equal")
        np.testing.assert_almost_equal(R[1]["dim"], r_struct["R"][1]["dim"], decimal=7, err_msg="Not equal")

        # flagsSpmRealign = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4, 'wrap': np.zeros((1,3)), 'rtm': 0, 'PW': '', 'lkp': np.array(range(0,6))})
        # flagsSpmReslice = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4, 'wrap': np.zeros((1,3)), 'mask': 1, 'mean': 0, 'which': 2})

        flagsSpmReslice = flags_reslice
        flagsSpmRealign = flags_realign

        nrSkipVol = p_struct["nrSkipVol"].item()
        [R, A0, x1, x2, x3, wt, deg, b, nrIter] = ra.Realign().spm_realign(R, flagsSpmRealign, indVol, nrSkipVol + 1, A0, x1, x2, x3, wt, deg, b)

        np.testing.assert_almost_equal(R[0]["C"], r_struct["R"][0]["C"], decimal=7, err_msg="Not equal")
        np.testing.assert_almost_equal(R[1]["C"], r_struct["R"][1]["C"], decimal=7, err_msg="Not equal")

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

        assert True, "Done"
    except Exception as err:
        assert False, f"Error occurred: {repr(err)}"
