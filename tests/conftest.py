# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
import nibabel as nib
import numpy as np
import pydicom
from scipy.io import loadmat


# main paths
@pytest.fixture(scope='session')
def root_path() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope='session')
def data_path(root_path) -> Path:
    return root_path / 'data'


@pytest.fixture(scope='session')
def second_data_path(data_path: Path) -> Path:
    return data_path / 'second_test'


@pytest.fixture(scope='session')
def third_data_path(data_path: Path) -> Path:
    return data_path / 'third_test'


# epi template
@pytest.fixture(scope='session')
def nii_image_1(data_path: Path) -> nib.nifti1.Nifti1Image:
    fp = data_path / 'fanon-0007-00006-000006-01.nii'
    return nib.load(fp, mmap=False)


# structural
@pytest.fixture(scope='session')
def struct_image(data_path: Path) -> nib.nifti1.Nifti1Image:
    fp = data_path / 'structScan_PSC.nii'
    return nib.load(fp, mmap=False)


# first test dcm
@pytest.fixture(scope='session')
def dcm_image(data_path: Path) -> np.array:
    fp = data_path / '001_000007_000006.dcm'
    return pydicom.dcmread(fp).pixel_array


# matlab resuls and settings
@pytest.fixture(scope='session')
def main_loop_data(data_path: Path) -> dict:
    fp = str(data_path / 'mainLoopData.mat')
    return loadmat(fp, squeeze_me=True)["mainLoopData"]


@pytest.fixture(scope='session')
def p_struct(data_path: Path) -> dict:
    fp = str(data_path / 'P.mat')
    return loadmat(fp, squeeze_me=True)["P"]


@pytest.fixture(scope='session')
def matlab_result(data_path: Path) -> np.array:
    fp = str(data_path / 'reslVol_matlab.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def smoothed_matlab(data_path: Path) -> np.array:
    fp = str(data_path / 'smoothed_matlab.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def resl_vol(data_path: Path) -> np.array:
    fp = str(data_path / 'reslVol.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def r_struct(data_path: Path) -> dict:
    fp = str(data_path / 'R.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def matlab_mc_result(data_path: Path) -> np.array:
    fp = str(data_path / 'mc_matlab_nii.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def matlab_mc_result_dcm(data_path: Path) -> np.array:
    fp = str(data_path / 'mc_matlab_dcm.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def check_data(data_path: Path) -> dict:
    fp = str(data_path / 'check_data.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def orth_matlab(data_path: Path) -> dict:
    fp = str(data_path / 'orth_matlab.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def orth_matlab_struct(data_path: Path) -> dict:
    fp = str(data_path / 'orth_matlab_struct.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def str_param(data_path: Path) -> dict:
    fp = str(data_path / 'strParam.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def xs(data_path: Path) -> dict:
    fp = str(data_path / 'xs.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def smoothed_python(data_path: Path) -> np.array:
    fp = str(data_path / 'smoothed_python.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def sum_vols(data_path: Path) -> np.array:
    fp = str(data_path / 'sumVols_python_dcm.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def mc_param(data_path: Path) -> np.array:
    fp = str(data_path / 'mc_python_dcm.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def prepr_data(data_path: Path) -> dict:
    fp = str(data_path / 'prepr_data.mat')
    return loadmat(fp, squeeze_me=True)


@pytest.fixture(scope='session')
def stat_map_2d_matlab(data_path: Path) -> dict:
    fp = str(data_path / 'stat_map_2d_matlab.mat')
    return loadmat(fp, squeeze_me=True)


# misc
@pytest.fixture(scope='session')
def flags_reslice(data_path: Path) -> dict:
    fp = str(data_path / 'flagsSpmReslice.mat')
    non_squeezed_struct = loadmat(fp, squeeze_me=False)
    quality = non_squeezed_struct["flagsSpmReslice"]["quality"][0][0][0][0]
    fwhm = non_squeezed_struct["flagsSpmReslice"]["fwhm"][0][0][0][0]
    sep = non_squeezed_struct["flagsSpmReslice"]["sep"][0][0][0][0]
    interp = non_squeezed_struct["flagsSpmReslice"]["interp"][0][0][0][0]
    wrap = non_squeezed_struct["flagsSpmReslice"]["wrap"][0][0][0]
    mask = non_squeezed_struct["flagsSpmReslice"]["mask"][0][0][0][0]
    mean = non_squeezed_struct["flagsSpmReslice"]["mean"][0][0][0][0]
    which = non_squeezed_struct["flagsSpmReslice"]["which"][0][0][0][0]

    flags_spm_reslice = {
        "quality": quality,
        "fwhm": fwhm,
        "sep": sep,
        "interp": interp,
        "wrap": wrap,
        "mask": mask,
        "mean": mean,
        "which": which
    }

    return flags_spm_reslice


@pytest.fixture(scope='session')
def flags_realign(data_path: Path) -> dict:
    fp = str(data_path / 'flagsSpmRealign.mat')
    non_squeezed_struct = loadmat(fp, squeeze_me=False)
    quality = non_squeezed_struct["flagsSpmRealign"]["quality"][0][0][0][0]
    fwhm = non_squeezed_struct["flagsSpmRealign"]["fwhm"][0][0][0][0]
    sep = non_squeezed_struct["flagsSpmRealign"]["sep"][0][0][0][0]
    interp = non_squeezed_struct["flagsSpmRealign"]["interp"][0][0][0][0]
    wrap = non_squeezed_struct["flagsSpmRealign"]["wrap"][0][0][0]
    rtm = non_squeezed_struct["flagsSpmRealign"]["rtm"][0][0][0][0]
    pw = non_squeezed_struct["flagsSpmRealign"]["PW"][0][0]
    lkp = non_squeezed_struct["flagsSpmRealign"]["lkp"][0][0][0]

    flags_spm_realign = {
        "quality": quality,
        "fwhm": fwhm,
        "sep": sep,
        "interp": interp,
        "wrap": wrap,
        "rtm": rtm,
        "PW": pw,
        "lkp": lkp
    }

    return flags_spm_realign
