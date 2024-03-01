# LAVASET

LAVASET (Latent Variable Stochastic Ensemble of Trees) is a Python package designed for ensemble learning in datasets with complex spatial, spectral, and temporal dependencies. The main method is described in our Oxford Bioinformatics paper: https://doi.org/10.1093/bioinformatics/btae101. 

## Features

- **Efficient Handling of Correlated Data**: Optimized for datasets where traditional models struggle.
- **Cython-Powered Performance**: Critical computations are implemented in Cython for efficiency.
- **Cross-Platform Compatibility**: Tested and deployable across Linux, macOS, and Windows.

## Installation

You can install LAVASET directly from PyPI:

```bash
pip install lavaset
```

## Requirements
- Python >= 3.7
- NumPy
- pandas
- scikit-learn
- scipy
- Cython
- joblib

Cython and NumPy are incorporated as build dependencies for LAVASET and are pre-installed before the package setup. If you encounter any issues during installation, especially regarding Cython or NumPy, consider installing these packages manually before proceeding with the LAVASET installation.

# For macOS or Windows users

LAVASET is built on a Linux architecture that is compatible with various linux platforms via a Docker image, however if you want to install directly to a MacOS or Windows environment using the conda would be the easiest way to do it.

## Step 1: Create a Conda environment

First, create and activate the conda environment where you'll install the Linux-built packages.

```bash
conda create -n lavaset-env python=3.x
conda activate lavaset-env
```
## Step 2: Add the Conda-forge channel
Add the Conda-forge channel, which provides many pre-built packages for various platforms.

```bash 
conda config --add channels conda-forge
```
## Step 3: Install linux-built LAVASET

```bash 
conda install LAVASET=0.1.0=linux-64
```

## Contributing

Contributions to LAVASET are always welcome.

### Issues 
Please submit any issues or bugs via the GitHub [issues](https://github.com/melkasapi/LAVASET/issues) page. Please include details about the LAVASET minor version used (`lavaset.__version__`) as well as any relevant input data.

### Contributions
Please submit any changes via a pull request. These will be reviewed by the LAVASET team and merged in due course.


## License

LAVASET is released under the MIT License. See the LICENSE file for more details.

## Contact

For questions or feedback, please contact Melpi Kasapi at mk218@ic.ac.uk.

Visit our [GitHub repository](https://github.com/melkasapi/LAVASET) for more information and updates.
