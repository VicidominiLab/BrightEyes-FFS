# BrightEyes-FFS

A toolbox for analysing Fluorescence Correlation Spectroscopy (FCS) and Fluorescence Fluctuation Spectroscopy (FFS) data with array detectors.
The fcs module contains libraries for:

* Calculating autocorrelations and cross-correlations of raw FCS/FFS data (i.e. photon counts vs. time or photon arrival time traces). Supported file types include .h5, .ptu, and .czi.
* Fitting correlations to various 2D and 3D diffusion models
* Calibration-free FCS/FFS analysis such as circular-scanning FCS and pair-correlation analysis
* Miscellaneous tools

The fcs_gui module contains libraries for:

* Storing and loading FCS/FFS analysis sessions, as used in the GUI

The pch module contains libraries for:

* Calculating photon counting histograms
* Fitting histograms with Fluorescence Intensity Distribution Analysis (FIDA)

The tools module contains libraries for:

* Fitting various models to data (polynomial, Gaussian, power law, etc.)
* Stokes-Einstein relation
* Save/load 2D arrays to/from .csv files
* Save data to .tiff file
* Miscellaneous tools

----------------------------------

## Installation

You can install `brighteyes-ffs` via [pip] directly from [PyPI]:

    pip install brighteyes-ffs

or using the version on GitHub:

    pip install git+https://github.com/VicidominiLab/BrightEyes-FFS

It requires the following Python packages

    h5py
	joblib
	matplotlib>=3.3.2
	multipletau>=0.3.3
	numpy>=1.19.4
	pandas>=1.1.4
	scipy
	tifffile>=2020.9.29
	seaborn
	imutils
	PyQt5
	qdarkstyle
	nbformat
	ome_types
	czifile
	brighteyes_ism
	notebook
	ptufile



### GUI

For quick and common types of analysis, you can use the GUI (https://github.com/VicidominiLab/BrightEyes-FFS-GUI), which contains most of the basic features. In addition, there is an automatic Jupyter Notebook writing tool to convert an analysis session started in the GUI to a Notebook.

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"BrightEyes-FFS" is free and open source software

## Contributing

You want to contribute? Great!
Contributing works best if you creat a pull request with your changes.

1. Fork the project.
2. Create a branch for your feature: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'My new feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request!
