from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    py_modules=[p.stem for p in Path(__file__).parent.glob('*.py') if p.stem != 'setup'],
    ext_modules=[Pybind11Extension('_tiemu_native', ['cpp/tiemu_native.cpp'], cxx_std=17)],
    cmdclass={'build_ext': build_ext},
)
