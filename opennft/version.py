# -*- coding: utf-8 -*-

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version('pyopennft')
except PackageNotFoundError:  # pragma: no cover
    __version__ = '0.0.0.dev'
