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
    return loadmat(fp)


@pytest.fixture(scope='session')
def p_struct(data_path: Path) -> dict:
    fp = str(data_path / 'P.mat')
    return loadmat(fp)
