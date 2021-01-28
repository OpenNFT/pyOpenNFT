# -*- coding: utf-8 -*-

import numpy as np
from opennft import reslice as rs
import pyspm


def test_reslice_rt(r_struct: dict, flags_reslice: dict):

    try:
        resl = rs.Reslicing()
        resl.spm_reslice(r_struct['R'], flags_reslice)
        assert True, "Done"
    except:
        assert False, "Error acquired"
