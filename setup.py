from pathlib import Path
import os
import sys
if sys.platform == 'darwin':
    os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '11.0')
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

compile_args = ['/fp:strict'] if sys.platform == 'win32' else ['-ffp-contract=off']
link_args = []
if sys.platform.startswith('linux'):
    compile_args.append('-pthread')
    link_args.append('-pthread')

setup(
    py_modules=[p.stem for p in Path(__file__).parent.glob('*.py') if p.stem != 'setup'],
    ext_modules=[Pybind11Extension('_tiemu_native', ['cpp/tiemu_native.cpp'], cxx_std=17,
                                  extra_compile_args=compile_args, extra_link_args=link_args)],
    cmdclass={'build_ext': build_ext},
)
