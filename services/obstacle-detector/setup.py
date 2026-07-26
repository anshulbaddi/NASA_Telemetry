"""Build the Cython LIDAR clustering extension."""
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name    = "lidar_cluster",
    sources = ["lidar_cluster.pyx"],
    extra_compile_args = ["-O3", "-march=native", "-ffast-math"],
)

setup(
    name       = "lidar_cluster",
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
