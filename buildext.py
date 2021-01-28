
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


MODULE_NAME = 'pyspm'

SOURCE_DIR = Path(__name__).parent / 'src'
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


def main():
    """Minimal setup for build pybind11 extension modules

    Usage::

        python buildext.py build

    """

    setup_kwargs = {
        'name': MODULE_NAME,
    }

    build(setup_kwargs)
    setup(**setup_kwargs)


if __name__ == '__main__':
    main()
