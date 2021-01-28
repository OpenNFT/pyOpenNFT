# -*- coding: utf-8 -*-

from opennft import reslice as rs


def test_reslice_rt(r_struct: dict, flags_reslice: dict):

    try:
        resl = rs.Reslicing()
        resl.spm_reslice(r_struct['R'], flags_reslice)
        assert True, "Done"
    except Exception as err:
        assert False, f"Error occurred: {repr(err)}"
