# -*- coding: utf-8 -*-

from opennft import reslice as rs
from opennft import realign as ra
from opennft.utils import Utils as utils
from pathlib import Path
import numpy as np
import nibabel as nib
import pydicom
from scipy.io import savemat
import time

def test_mc_dcm(third_data_path: Path, nii_image_1: nib.nifti1.Nifti1Image, p_struct: dict, matlab_MCResult_dcm: np.array, r_struct: dict, xs: dict):

    try:

        timeStamps = np.zeros((158,))
        t0 = time.time()

        A0 = []
        x1 = []
        x2 = []
        x3 = []
        wt = []
        deg = []
        b = []
        R = [{'mat': None, 'dim': None, 'Vol': None} for i in range(2)]
        dimVol = np.array([74, 74, 36])

        # R[0]["mat"] = r_struct["R"][0]["mat"]
        R[0]["mat"] = nii_image_1.affine
        R[0]["dim"] = dimVol.copy()
        tmpVol = np.array(nii_image_1.get_fdata(), dtype='uint16', order='F')

        slNrImg2DdimX, slNrImg2DdimY, img2DdimX, img2DdimY = utils().getMosaicDim(dimVol)

        t1 = time.time()
        timeStamps[1] = t1-t0

        nrZeroPadVol = p_struct["nrZeroPadVol"].item()
        if p_struct["isZeroPadding"].item():
            R[0]["dim"][2] = R[0]["dim"][2]+nrZeroPadVol*2
            R[0]["Vol"] = np.pad(tmpVol, ((0,0),(0,0),(nrZeroPadVol,nrZeroPadVol)), 'constant', constant_values=(0, 0))
        else:
            R[0]["Vol"] = tmpVol

        motCorrParam = np.zeros((155,6))
        sumVols = np.zeros((155,74,74,36))
        offsetMCParam = np.zeros((6,))

        t2 = time.time()
        timeStamps[2] = t2-t1

        for indVol in range(0,155):

            ti0 = time.time()
            fileName = str(indVol+1)+'.dcm'
            data = np.array(pydicom.dcmread(third_data_path / fileName).pixel_array, dtype='uint16', order='F')

            R[1]["mat"] = R[0]["mat"]
            R[1]["dim"] = dimVol.copy()
            tmpVol = utils().img2Dvol3D(data, slNrImg2DdimX, slNrImg2DdimY, dimVol)

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
            [R, A0, x1, x2, x3, wt, deg, b, nrIter] = ra.Realign().spm_realign(R, flagsSpmRealign, indVol+1, 1, A0, x1, x2, x3, wt, deg, b)

            tempM = np.linalg.solve(R[0]["mat"].T,R[1]["mat"].T).T
            tmpMCParam = utils().spm_imatrix(tempM)
            if indVol+1 == 1:
                offsetMCParam = tmpMCParam[0:6]
            motCorrParam[indVol,:] = tmpMCParam[0:6]-offsetMCParam

            if p_struct["isZeroPadding"].item():
                tmp_reslVol = rs.Reslicing().spm_reslice(R, flagsSpmReslice)
                reslVol = tmp_reslVol[:,:, nrZeroPadVol : -1 - nrZeroPadVol +1]
                dimVol[2] = dimVol[2] - nrZeroPadVol * 2
            else:
                reslVol = rs.Reslicing().spm_reslice(R, flagsSpmReslice)

            sumVols[indVol,:,:,:] = reslVol

            ti1 = time.time()
            timeStamps[indVol+3] = ti1-ti0

        # reslDic = {"mc_python": motCorrParam}
        # savemat("data/mc_python_dcm.mat", reslDic)
        #
        # reslDic = {"sumVols": sumVols}
        # savemat("data/sumVols_python_dcm.mat", reslDic)

        # print('\n')
        # for i in range(0,6):
        #     print('Third test MSE for {:} coordinate = {:}'.format(i+1, ((motCorrParam[:,i] - matlab_MCResult_dcm["mc_series_matlab"][:,i])**2).mean()))

        reslDic = {"python_times": timeStamps}
        savemat("python_times_dcm.mat", reslDic)

        assert True, "Done"
    except Exception as err:
        assert False, f"Error occurred: {repr(err)}"
