"""Compatibility shim; all package metadata lives in pyproject.toml.

pip older than 21.3 (JetPack 5 / Ubuntu 20.04 ships pip 20.0.2) can do ``pip install -e .``
only through ``setup.py develop`` and refuses a project without this file. It still runs that
command in an isolated build environment with the setuptools >= 61 that pyproject's
[build-system] asks for, and that setuptools reads the [project] table, so ``setup()`` needs no
arguments. Keep metadata out of here (tests/test_packaging.py checks that). Newer pip and uv
build through pyproject.toml (PEP 517/660); setuptools still runs this file, which adds nothing.
"""
from setuptools import setup

setup()
