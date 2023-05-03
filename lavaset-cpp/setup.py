from setuptools import setup, Extension
# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np



ext = Extension(
    "cython_wrapper",
    sources=["cython_wrapper.pyx", "GBCP.cpp"],
    language="c++",
    extra_compile_args=["-std=c++11"],
)

setup(
    ext_modules=cythonize(ext, language_level=3),
    include_dirs=[np.get_include(), ["."]]
)

