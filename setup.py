from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="lavaset", ext_modules=cythonize('splitting.pyx'), include_dirs=[numpy.get_include()])

# from setuptools import setup, Extension
# from Cython.Distutils import build_ext
# import numpy as np

# NAME = "lavaset"

# REQUIRES = ["numpy", "cython", "joblib"]

# SRC_DIR = "LAVASET"
# PACKAGES = [SRC_DIR]

# ext_1 = Extension(SRC_DIR + "lavaset",
#                   [SRC_DIR + "/splitting.pyx"],
#                   libraries=[],
#                   include_dirs=[np.get_include()])
# EXTENSIONS = [ext_1]

# if __name__ == "__main__":
#     setup(install_requires=REQUIRES,
#           packages=PACKAGES,
#           zip_safe=False,
#           name=NAME,
#           cmdclass={"build_ext": build_ext},
#           ext_modules=EXTENSIONS
#           )