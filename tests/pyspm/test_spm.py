# -*- coding: utf-8 -*-

from opennft import reslice as rs
from opennft import realign as ra
from opennft import img2Dvol3D as i2v3
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat


def test_spm_rt(dcm_image: np.array, main_loop_data: dict, p_struct: dict, matlab_result: np.array, rVol_struct : dict, dcmData_struct : np.array, tmpVol_struct : np.array):

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
        R[1]["mat"] = matVol
        tmpVol = i2v3.img2Dvol3D().img2Dvol3D(dcmData, slNrImg2DdimX, slNrImg2DdimY, dimVol)
        # tmpVol = np.array(tmpVol_struct["tmpVol"], dtype=float)

        nrZeroPadVol = p_struct["nrZeroPadVol"].item()
        if p_struct["isZeroPadding"].item():
            zeroPadVol = np.zeros((dimVol[0],dimVol[1],nrZeroPadVol))
            dimVol[2] = dimVol[2]+nrZeroPadVol*2
            # R[1]["Vol"] = np.concatenate((zeroPadVol,tmpVol,zeroPadVol),2)
            R[1]["Vol"] = np.pad(tmpVol, ((0,0),(0,0),(nrZeroPadVol,nrZeroPadVol)), 'constant', constant_values=(0, 0))
            # R[1]["Vol"] = np.array(rVol_struct["R"][1]["Vol"],dtype=float)
        else:
            R[1]["Vol"] = tmpVol

        R[1]["dim"] = dimVol

        flagsSpmRealign = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4, 'wrap': np.zeros((3,1)), 'rtm': 0, 'PW': '', 'lkp': np.array(range(0,6))})
        flagsSpmReslice = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4, 'wrap': np.zeros((3,1)), 'mask': 1, 'mean': 0, 'which': 2})

        nrSkipVol = p_struct["nrSkipVol"].item()
        [R, A0, x1, x2, x3, wt, deg, b, nrIter] = ra.Realign().spm_realign(R, flagsSpmRealign, indVol, nrSkipVol + 1, A0, x1, x2, x3, wt, deg, b)

        if p_struct["isZeroPadding"].item():
            tmp_reslVol = rs.Reslicing().spm_reslice(R, flagsSpmReslice)
            reslVol = tmp_reslVol[:,:, nrZeroPadVol : -1 - nrZeroPadVol +1]
            dimVol[2] = dimVol[2] - nrZeroPadVol * 2
        else:
            reslVol = rs.Reslicing().spm_reslice(R, flagsSpmReslice)

        reslDic = {"reslVol_python": reslVol, "label": "reslVol_python"}
        savemat("data/reslVol.mat", reslDic)

        matlab_reslVol = matlab_result["reslVol"]
        np.testing.assert_almost_equal(reslVol, matlab_reslVol, decimal=7, err_msg="Not equal")

        assert True, "Done"
    except Exception as err:
        assert False, f"Error occurred: {repr(err)}"
