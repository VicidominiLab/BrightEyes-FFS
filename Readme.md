# BrightEyes-FFS

A toolbox for analysing Fluorescence Correlation Spectroscopy (FCS) and Fluorescence Fluctuation Spectroscopy (FFS) data with array detectors.
The fcs module contains libraries for:

* Calculating autocorrelations and cross-correlations of raw FCS/FFS data (i.e. photon counts vs. time)
* Fitting correlations to various 2D and 3D diffusion models
* Calibration-free FCS/FFS analysis such as circular-scanning FCS and iMSD analysis
* Miscellaneous tools

The fcs_gui module contains libraries for:

* Storing and loading FCS/FFS analysis sessions, as used in the GUI

The dataio module contains libraries for:

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

## Getting started 

### Supported file types

The current version of the package can read FFS data in .h5, .tiff, and .czi files. For .h5 files generated with BrightEyes-MCS, no pre-processing is needed. For custom .h5 files, make sure to use the keyword 'data' to store the [Nt x Nc] array of FFS data, with *Nt* the number of time points and *Nc* the number of channels (detector elements). For .tiff, make sure the data contains one page with a 2D array [Nt x Nc]. For .czi files, the data is automatically converted to .h5 upon calculating the correlation.

The only metadata that needs to be known is the time per data point (dwell time). For .h5 files, make sure the dwell time in microseconds can be read using

	f['configurationGUI_beforeStart'].attrs['time_resolution']

For .tiff files, make sure the dwell time in microseconds is stored as a tag called 'dwell_time'.

For .czi files, the dwell time cannot be automatically read. Instead, make sure a .txt file with the following syntax is stored in the same location as the .czi file. If your file is located at *my_path/my_file.czi*, then add a .txt file with the same name, i.e. *my_path/my_file.txt*, with exactly the following information:

	DWELL TIME [us]
	17

Replace *17* with your actual dwell time in microseconds. BrightEyes-FFS will automatically generate an .h5 file stored in the same location.

### Calculating correlations

The variable *file* contains the path to the data file (.h5 or .czi). The variable *accuracy* defines the number of points for which the correlation is calculated. The higher this number, the more points G contains. Note that accuracy does not (always) equal the number of points. The variable *split* contains the duration of a single chunk of data for which the correlation is calculated. E.g. for a 100 s time trace, split=20 will result in 5 correlations from a 20 s time trace each. The boolean *time_trace* defines whether or not the time trace should be returned. The variable *algorithm* defines the algorithm used to calculate the correlation. Valid options are 'multipletau' (time-based) and 'wiener-khinchin' (fft based).

The variable *list_of_g* contains the various correlations to be calculated. Acceptable entries are (i) integer numbers for calculating the autocorrelation of a given channel, (ii) a pre-defined string (such as 'central', 'sum3', and 'sum5') or (iii) a custom-made string with the following syntax 'Ca+b+...+cxd+...+e': starting with C (for Custom), then a list of channels to sum over (a+b+c), then x, then a second list of channels to sum over. E.g. 'C0+1x2+3' calculates tha cross-correlation between the sum of the channels 0 and 1 and the sum of channels 2 and 3. To calculate the autocorrelation, use simply 'Ca+b+c' (which is equivalent to 'Ca+b+cxa+b+c'). For cross-correlations of single channels, use 'xaabb' with aa and bb the two channel numbers (e.g. 'x0123' for the cross-correlation between channels 1 and 23).


	from brighteyes_ffs.fcs.fcs2corr import fcs_load_and_corr_split as correlate
	list_of_g = ['central', 'sum3', 'sum5']
	G, time_trace = correlate(file, list_of_g=list_of_g, accuracy=16, split=10, time_trace=True, algorithm='multipletau')

G is an object with 2D arrays (tau, G) for each correlation in list_of_g and each chunk of data, e.g. G.central_chunk0 contains the autocorrelation for the central detector element for the first chunk of data. *G.central_average* contains the average correlation for the central element over the whole data set.

To remove bad chunks of data, e.g. caused by a bright aggregate entering the focal volume, use the following code:

	good_chunks = [0, 1, 4, 5, 6, 7, 8, 9] # list with the good chunks of data, i.e., remove chunks 2 and 3 from further analysis
	G = fcs_av_chunks(G, good_chunks)
	
The output of *fcs_av_chunks* is the same object as before, but now containing attributes such as *G.central_averageX*, where the *X* stands for the custom average of the good chunks only.

### Fitting correlations

The library *fcs_fit* contains several fit functions. The variable *fitfun* can be 'fitfun_2c' (for 1 or 2-component free diffusion), 'fitfun_an' (for 1-component anomalous diffusion), 'fitfun_dualfocus' (for free diffusion with two focii displaced from each other), 'fitfun_circfcs' (for free diffusion with a circularly scanning excitation beam), 'mem_fit_free_diffusion' (for free diffusion with a large number of components), 'fitfun_free_diffusion_2d' (for 1 or 2-component free diffusion in 2D), or 'fcs_analytical_2c_anomalous' (for 2-component anomalous diffusion). The variable *fit_info* contains 0s and 1s for parameters that have to be kept fixed and fitted, respectively. The variable *param* contains the starting values for all parameters. Check the library for the order of parameters for each fit function.

	from brighteyes_ffs.fcs.fcs_fit import fcs_fit
	fitresults = []
	for corr in list_of_g:
		Gsingle = getattr(G, corr + '_averageX') # get the average correlation curve
		Gexp = Gsingle[1:,1]
		tau = Gsingle[1:,0]
		fitfun = 'fitfun_2c'
		fit_info = np.asarray([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]) # fit N, tauD, and offset (we fit with one component)
		param = np.asarray([1, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0]) # starting values for all parameters
		lBounds = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])*(-1e6) # lower bounds for all parameters
		uBounds = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])*(1e6) # upper bounds for all parameters
		fitresult = fcs_fit(Gexp, tau, fitfun, fit_info, param, lBounds, uBounds, plotInfo=-1)
		fitresults.append(fitresult)

The variable *fitresults* contains a list of *N* elements, with *N* = len(list_of_g), in which each element is the output of a scipy.optimize.least_squares call. E.g. fitresults[0].fun contains the residuals for the first fit (typically the detector central element in spot-variation FCS). Similarly, fitresults[0].x contains the fitted parameter values.

### Checking the diffusion law

In spot-variation FCS, the diffusion law tau_D vs w0^2 is plotted and fitted with a linear curve to derive the diffusion modality. Here, tau_D is the diffusion time and w0^2 the beam waist squared for each of the three curves, as found in a calibration measurement. For free diffusion, the fit must go through the origin. A positive intercept with the y-axis means anomalous diffusion caused by microdomain formation, a negative intercept anomalous diffusion caused by a meshwork. To check the diffusion law, use:

	taufit = np.asarray([fitresults[i].x[0] for i in range(3)]) # diffusion times in ms for the three curves
	w0 = np.asarray([220, 290, 364])*1e-3 # beam waists from calibration in micrometer
	fitresult = fit_curve(taufit, w0**2, 'linear', [1, 1], param=[1, 1], lBounds=[-1e6, -1e6], uBounds=[1e6, 1e6], savefig=0) # use 'linear' for y = a*x + b, param = starting values

Plotting the diffusion law:

	taufitres = np.zeros(len(w0) + 1)
	taufitres[0] = fitresult.x[1]
	taufitres[1:] = taufit - fitresult.fun

	w02fit = np.zeros(len(w0) + 1)
	w02fit[1:] = w0**2

	plt.figure()
	plt.scatter(w0**2, taufit)
	plt.plot(w02fit, taufitres)

### GUI

To get familiar with the BrightEyes-FFS package, we highly recommend using the GUI (https://github.com/VicidominiLab/BrightEyes-FFS-GUI) which contains an automatic Jupyter Notebook writing tool.

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
