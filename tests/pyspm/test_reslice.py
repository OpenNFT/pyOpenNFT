# -*- coding: utf-8 -*-

from opennft import reslice as rs
import numpy as np
from scipy.io import savemat


def test_reslice_rt(r_struct: dict, flags_reslice: dict, matlab_result: np.array):

    try:
        resl = rs.Reslicing()
        tmpReslVol = resl.spm_reslice(r_struct['R'], flags_reslice)
        reslVol = tmpReslVol[:,:,3:-1-2]
        # reslDic = {"reslVol_python": reslVol, "label": "reslVol_python"}
        # savemat("data/reslVol.mat", reslDic)
        matlab_reslVol = matlab_result["reslVol"]
        np.testing.assert_almost_equal(reslVol, matlab_reslVol, decimal=7,err_msg="Not equal")
        assert True, "Done"
    except Exception as err:
        assert False, f"Error occurred: {repr(err)}"
