try:
    from Cython.Build import cythonize
except ImportError as e:
    raise ImportError(f"{e}. Please install Cython and NumPy before installing this package.") from e

from setuptools import setup, find_packages, Extension
from pathlib import Path

# Define the directory containing this script
this_directory = Path(__file__).parent
# Read the contents of your README file
long_description = (this_directory / "README.md").read_text()

def numpy_include():
    import numpy
    return numpy.get_include()
# Define the extension module
ext = Extension(
    name="lavaset.cython_wrapper",  # Extension name
    sources=["lavaset/cython_wrapper.pyx", "lavaset/GBCP.cpp"],  # Source files
    language="c++",  # Specify the language
    extra_compile_args=["-std=c++11"],  # Additional flags for the compiler
)

# Configuration for the package
setup(
    name="LAVASET",
    version="1.0.0",
    author="Melpomeni Kasapi",
    author_email="mk218@ic.ac.uk",
    description="LAVASET: Latent Variable Stochastic Ensemble of Trees. An ensemble method for correlated datasets with spatial, spectral, and temporal dependencies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/melkasapi/LAVASET",
    packages=['lavaset'],
    ext_modules=cythonize([ext], language_level=3),
    include_dirs=[numpy_include()],  # Include the NumPy headers
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy',
        'setuptools',
        'scipy',
        'cython',
        'joblib',
    ],
    # setup_requires=['numpy', 'cython']
)
