"""
Minimal setup.py shim — pyproject.toml is the source of truth.
Needed for: older pip (<21.3), conda envs, editable installs on some platforms.

Install:
    pip install -e .
    # or
    pip install -e . --no-build-isolation
"""
from setuptools import setup

setup()
