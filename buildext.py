"""
Building Python extension modules via pybind11

The script can be used from Poetry pyproject.toml:

[tool.poetry]
build = "buildext.py"
...

poetry install

Or directly via distutils 'build' command:

python buildext.py build
python buildext.py build --debug  # for debug symbols

"""

import sys
from pathlib import Path
import shutil

from pybind11.setup_helpers import Pybind11Extension, build_ext

from setuptools import setup
from distutils.core import Distribution


MODULE_NAME = 'pyspm'

ROOT_PATH = Path(__name__).parent
SOURCE_DIR = ROOT_PATH / 'src'
SOURCES = sorted(map(str, SOURCE_DIR.glob('*.cpp')))

DEFINE_MACROS = []

if sys.platform == 'win32':
    DEFINE_MACROS.append(
        ('SPM_WIN32', None),
    )

ext_modules = [
    Pybind11Extension(MODULE_NAME, SOURCES, define_macros=DEFINE_MACROS),
]


def build(setup_kwargs: dict):
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": build_ext}
    })


def copy_files(dist: Distribution):
    build_cmd = dist.get_command_obj('build')
    src_dir = ROOT_PATH / build_cmd.build_lib

    for fpath in src_dir.iterdir():
        shutil.copy2(fpath, ROOT_PATH)


def main():
    """Minimal setup for build pybind11 extension modules with debug symbols

    Usage::

        python buildext.py build --debug

    """

    setup_kwargs = {
        'name': MODULE_NAME,
    }

    build(setup_kwargs)
    dist = setup(**setup_kwargs)
    copy_files(dist)


if __name__ == '__main__':
    main()
