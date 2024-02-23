from setuptools import setup, Extension
# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="lavaset.cython_wrapper",
    sources=["src/lavaset/cython_wrapper.pyx", "src/lavaset/GBCP.cpp"],
    language="c++",
    extra_compile_args=["-std=c++11"],
)

setup(
    ext_modules=cythonize(ext, language_level=3),
    include_dirs=[np.get_include(), ["."]]
)

setup(
    name="LAVASET",
    version="0.1.0",
    author="Melpomeni Kasapi",
    author_email="mk218@ic.ac.uk",
    description="LAVASET: Latent Variable Stochastic Ensemble of Trees. An ensemble method for correlated datasets with spatial, spectral, and temporal dependencies ",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/melkasapi/LAVASET",
    packages=["lavaset"],
    package_dir={'':'src'},
    ext_modules=cythonize([ext], language_level=3),
    include_dirs=[np.get_include()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
