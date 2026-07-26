"""Build the Cython/C++ FDIR native extension."""
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name               = "fdir_native",
    sources            = ["fdir_native.pyx", "rolling_buffer.cpp"],
    include_dirs       = [numpy.get_include(), "."],
    language           = "c++",
    extra_compile_args = ["-O3", "-march=native", "-std=c++17", "-ffast-math"],
    extra_link_args    = ["-std=c++17"],
)

setup(
    name        = "fdir_native",
    ext_modules = cythonize(
        ext,
        compiler_directives = {
            "language_level": "3",
            "boundscheck":    False,
            "wraparound":     False,
            "cdivision":      True,
            "nonecheck":      False,
        },
    ),
)
