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

## For macOS or Windows users

LAVASET is built on a Linux architecture that is compatible with various linux platforms via a Docker image, however if you want to install directly to a MacOS or Windows environment using the conda would be the easiest way to do it.

### Step 1: Create a Conda environment

First, create and activate the conda environment where you'll install the Linux-built packages.

```bash
conda create -n lavaset-env python=3.x
conda activate lavaset-env
```
### Step 2: Add the Conda-forge channel
Add the Conda-forge channel, which provides many pre-built packages for various platforms.

```bash 
conda config --add channels conda-forge
```
### Step 3: Install linux-built LAVASET

```bash 
conda install LAVASET=0.1.0=linux-64
```
## Example Usage

A jupyter notebook with examples on how to import and use the LAVASET package can be found [here]. Briefly, the LAVASET model can be called as below

```bash
model = LAVASET(ntrees=100, n_neigh=10, distance=False, nvartosample='sqrt', nsamtosample=0.5, oobe=True) 
```

- ntrees: number of trees (or estimators) for the ensemble (int)
- n_neigh: number of neighbors to take for the calculation of the latent variable; this excludes the feature that has been selected for split, therefore the latent variable is calculated by the total of n+1 features (int)
- distance: parameter indicating whether the input for neighbor calculation is a distance matrix, default is False; if True, then n_neigh should be 0 (boolean)
- nvartosample: the number of features picked for each split, 'sqrt' indicates the squared root of total number of features, if int then takes that specific number of features (string or int)
- nsamtosample: the number of sample to consider for each tree, if float (like 0.5) then it considers `float * total number of samples`, if int then takes that specific number of samples (float or int)
- oobe: parameter for calcualting the out-of-bag score, default=True (boolean)

If the input to the `knn_calculation` function is a distance matrix then:
```bash
model = LAVASET(ntrees=100, n_neigh=0, distance=True, nvartosample='sqrt', nsamtosample=0.5, oobe=True) 

knn = model.knn_calculation(distance_matrix, data_type='distance_matrix')
```
If the neighbors need to be calculated from the 1D spectrum ppm values of an HNMR dataset, then the input is the 1D array with the ppm values. Here the model parameters should be set as `distance=False` and `n_neigh=k`. The `data_type` parameter for the `knn_calculation` in this case will be set to `1D`. All options include:
- 'distance_matrix' is used for distance matrix input, 
- '1D' is used for 1D data like signals or spectra, 
- 'VCG' is used for VCG data, 
- 'other' is used for any other type of data, where it calculates the nearest neighbors based on the 2D data input. 

```bash 
knn = model.knn_calculation(mtbls1.columns[1:], data_type='1D') ### this is the input for the knn calculation 
```

## Contributing

Contributions to LAVASET are always welcome.

### Issues 
Please submit any issues or bugs via the GitHub [issues](https://github.com/melkasapi/LAVASET/issues) page. Please include details about the LAVASET minor version used (`lavaset.__version__`) as well as any relevant input data.

### Contributions
Please submit any changes via a pull request. These will be reviewed by the LAVASET team and merged in due course.


## License

LAVASET is released under the GNU License. See the LICENSE file for more details.

## Contact

For questions or feedback, please contact Melpi Kasapi at mk218@ic.ac.uk.

Visit our [GitHub repository](https://github.com/melkasapi/LAVASET) for more information and updates.
