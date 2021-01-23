from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("python_spm",
        ["src/bsplines.c", "src/bsplines.h", "src/spm_bsplinc.c", "src/spm_bsplins.c", "src/spm_datatypes.h",
         "src/spm_mapping.c", "src/spm_mapping.h", "src/spm_vol_access.c", "src/spm_vol_access.h", "src/spm_vol_utils.c" ],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        ),
]

setup(
    name="python_spm",
    version=__version__,
    author="Nikita Davydov",
    author_email="nikita.davydov.ssau@gmail.com",
    url="https://github.com/OpenNFT/pyOpenNFT",
    description="Python adaptation of SPM for OpenNFT project",
    long_description="",
    ext_modules=ext_modules,
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)