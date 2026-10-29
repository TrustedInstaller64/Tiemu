from pathlib import Path
import os
import sys
if sys.platform == 'darwin':
    os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '11.0')
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CoreBuild(build_ext):
    def get_export_symbols(self, extension):
        # This is a regular shared library loaded with ctypes, with no PyInit
        # entry point or Python C API dependency. Windows exports use the header.
        return []

compile_args = (['/std:c++17', '/fp:strict', '/O2'] if sys.platform == 'win32'
                else ['-std=c++17', '-ffp-contract=off', '-fno-fast-math', '-O3'])
link_args = []
if sys.platform.startswith('linux'):
    compile_args.append('-pthread')
    link_args.append('-pthread')

setup(
    py_modules=[p.stem for p in Path(__file__).parent.glob('*.py') if p.stem != 'setup'],
    ext_modules=[Extension('_tiemu_core',
                           ['cpp/tiemu_core.cpp', 'cpp/tiemu_inorganic.cpp'],
                           language='c++', define_macros=[('TIEMU_CORE_BUILD', '1')],
                           extra_compile_args=compile_args, extra_link_args=link_args)],
    cmdclass={'build_ext': CoreBuild},
)
