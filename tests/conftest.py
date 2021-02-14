# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
import pydicom
import numpy as np
from scipy.io import loadmat


@pytest.fixture(scope='session')
def root_path() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope='session')
def data_path(root_path) -> Path:
    return root_path / 'data'


@pytest.fixture(scope='session')
def dcm_image(data_path: Path) -> np.array:
    fp = data_path / '001_000007_000006.dcm'
    return pydicom.dcmread(fp).pixel_array


@pytest.fixture(scope='session')
def main_loop_data(data_path: Path) -> dict:
    fp = str(data_path / 'mainLoopData.mat')
    return loadmat(fp,squeeze_me=True)["mainLoopData"]


@pytest.fixture(scope='session')
def p_struct(data_path: Path) -> dict:
    fp = str(data_path / 'P.mat')
    return loadmat(fp,squeeze_me=True)["P"]

@pytest.fixture(scope='session')
def rVol_struct(data_path: Path) -> dict:
    fp = str(data_path / 'rVol.mat')
    return loadmat(fp, squeeze_me=True)

@pytest.fixture(scope='session')
def r_struct(data_path: Path) -> dict:
    fp = str(data_path / 'R.mat')
    return loadmat(fp, squeeze_me=True)

@pytest.fixture(scope='session')
def matlab_result(data_path: Path) -> np.array:
    fp = str(data_path / 'reslVol_matlab.mat')
    return loadmat(fp, squeeze_me=True)

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
