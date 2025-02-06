#from functions.appearance import corr_long_label as cll
import numpy as np
import matplotlib.image as mpimg
import copy
from datetime import datetime

#current = os.path.dirname(os.path.realpath('analysis_settings.py'))
#parent = os.path.dirname(current)
#sys.path.append(parent)

from ..fcs.fcs2corr import fcs_av_chunks, fcs_crosscenter_av
from ..fcs.fcs_polar import g2polar, g2flow

"""
==============================================================================
FFSlib = [FFSexp1, FFSexp2, FFSexp3, etc.] with each element corresponding to 
one FFS experiment, i.e. 1 2d image and multiple FFS measurements
    
    FFSexp1 = [FFSfile1, FFSfile2, FFSfile3, etc.] with each element
    corresponding to one FFS measurement

FFSlib
     -> FFSexp: image
         -> FFSfile: single spot measurement
             -> metadata
             -> analysis
==============================================================================
"""

class FFSlib:
    """
    FFSlib is the main data object. It contains all images,
    each of which may contain multiple FFS experiments,
    and each experiment may contain multiple analyses.
    """
    def __init__(self,
                 lib = None,
                 active_image = None,
                 version = '1.1',
                 notes = '',
                 date_created = None,
                 date_modified = None):
        self.lib = lib if lib is not None else []
        self.active_image = active_image
        self.version = version
        self.notes = notes
        self.date_created = date_created or str(datetime.now())
        self.date_modified = date_modified or str(datetime.now())
    
    @property
    def num_images(self):
        """Returns the number of images in the library."""
        return len(self.lib)
    
    def add_image(self, image, fname, mdata=None):
        """Adds a new image object to the library."""
        if not fname:
            raise ValueError("Image must have a valid filename.")
        new_image_obj  = FFSimage()
        new_image_obj.image_name = fname
        new_image_obj.image = image
        if mdata is not None:
            new_image_obj.image_metadata = mdata
        new_image_obj.image_metadata.num_pixels = np.shape(image)[1]
        new_image_obj.image_metadata.num_lines = np.shape(image)[0]
        new_image_obj.imageRaw = image # may be deleted
        self.lib.append(new_image_obj)
    
    def add_random_image(self):
        self.add_image(np.zeros((5,5)), 'no_image_added')
    
    def get_image(self, image_nr='active'):
        # return the imageNr-th image object from the library
        # by default return current image object
        if isinstance(image_nr, str) and image_nr == 'active':
            image_nr = self.active_image
        if image_nr is None or self.num_images < 1:
            return None
        if image_nr > -1:
            image_nr = np.min((image_nr, self.num_images-1))
        image = self.lib[image_nr]
        return image
    
    def remove_image(self, image_nr='active'):
        if image_nr == 'active':
            image_nr = self.active_image
        if image_nr is not None and image_nr < self.num_images:
            del self.lib[image_nr]
            if self.num_images == 0:
                self.active_image = None
            else:
                self.active_image = 0
    
    def get_image_name(self, image_nr):
        image = self.get_image(image_nr)
        if image is None:
            return "No images added yet."
        return image.image_name
    
    def return_image(self, image_nr=-1):
        # return the image_nr-th image from the library
        image = self.get_image(image_nr)
        if image is None:
            return mpimg.imread('files/Cells_DEKegfp_75x75um.jpg')
        return image.image
    
    
class FFSimage:
    """
    FFSImage contains everything related to a single 2D image, 
    including metadata, image data, and associated FFS files.
    """
    def __init__(self):
        self.image_name  = "" # path to image
        self.image_metadata = FFSmetadata() # image metadata
        self.image = None
        self.image_raw = None
        self.ffs_list = [] # empty list that will be filled with FFSfile objects
        self.active_ffs = None
    
    @property
    def num_files(self):
        # return the number of FFS files for image
        return len(self.ffs_list)
    
    def change_image(self,  image, fname, mdata=None):
        self.image_name = fname
        self.image = image
        if mdata is not None:
            self.image_metadata = mdata
        self.image_metadata.num_pixels = np.shape(image)[1]
        self.image_metadata.num_lines = np.shape(image)[0]
        self.imageRaw = image # may be deleted
    
    def update(self, image_name=None, image=None, ffs_list=None):
        if image_name is not None:
            self.image_name = image_name
        if image is not None:
            self.image = image
        if ffs_list is not None:
            self.ffs_list = ffs_list
    
    def add_ffs_file(self, FFSfileObj):
        self.ffs_list.append(FFSfileObj)
    
    def remove_ffs_file(self, file_num):
        if file_num is not None and file_num < self.num_files:
            del self.ffs_list[file_num]
            
    def get_ffs_file(self, file_num='active'):
        if type(file_num) == str and file_num == 'active':
            file_num = self.active_ffs
        if file_num is not None and file_num < self.num_files:
            return self.ffs_list[file_num]
        return None
    
    def print_image_metadata(self):
        txt = ''
        for prop in ['num_pixels', 'num_lines', 'range_x', 'dwelltime', 'pxsize']:
            propvalue = getattr(self.image_metadata, prop)
            if propvalue is not None:
                if prop == 'num_pixels':
                    txt += str(int(propvalue))
                if prop == 'num_lines':
                    txt += ' x ' + str(int(propvalue)) + ' pixels\n'
                if prop == 'range_x':
                    txt += str(int(propvalue)) + ' x ' + str(int(propvalue)) + ' um\n'
                if prop == 'dwelltime':
                    txt += 'Dwelltime: ' + str(int(1e6*propvalue)) + ' us\n'
                if prop == 'pxsize':
                    txt += 'Pixel size: ' + "{:.3f}".format(propvalue) + ' um\n'
                
        return txt
        
        
class FFSfile:
    # each FFS file for each image contains a single FFSfile object
    def __init__(self,
                 fname="",  # path to file
                 label=None, # file nick name
                 metadata=None,
                 analysis_list=None, # list of correlations to calculate
                 timetrace=None,
                 airy=None,
                 active_analysis=None):
        self.fname = fname
        self.label = label 
        self.metadata = metadata if metadata is not None else FFSmetadata()
        self.analysis_list = analysis_list if analysis_list is not None else []
        self.timetrace = timetrace
        self.airy = airy
        self.active_analysis = active_analysis # choose which analysis is active by default
    
    @property
    def num_analyses(self):
        return len(self.analysis_list)
    
    @property
    def coords(self):
        return self.metadata.coords
    
    @property
    def duration(self):
        return self.metadata.duration
    
    @property
    def number_of_elements(self):
        if self.timetrace is None:
            return None
        if len(np.shape(self.timetrace)) > 1:
            # 2 dimensions
            return np.shape(self.timetrace)[1]
        return len(self.timetrace)
    
    def add_analysis(self, mode, resolution=10, chunksize=10, algorithm='multipletau', chunks_off="allon", active_analysis=-1):
        FFScorrObj = FFScorr()
        duration = self.metadata.duration
        if active_analysis == -1:
            self.active_analysis = self.num_analyses
        else:
            self.active_analysis = active_analysis
        if chunks_off == "allon":
            chunks_off = np.ones(np.clip(int(np.floor(duration / chunksize)), 1, None))
        FFScorrObj.analysis(mode, resolution, chunksize, chunks_off, algorithm)
        self.analysis_list.append(FFScorrObj)
    
    def remove_analysis(self, an_nr):
        if an_nr is not None and an_nr < self.num_analyses:
            del self.analysis_list[an_nr]
            if self.num_analyses == 0:
                self.active_analysis = None
            else:
                self.active_analysis = np.min((self.num_analyses-1, self.active_analysis))
    
    def copy_correlation(self, an_nr):
        # get analysis
        an_orig = self.get_analysis(analysis_num=an_nr)
        if an_orig is None:
            return
        # copy analysis and fit
        self.analysis_list.append(copy.deepcopy(an_orig))
        an = self.get_analysis(analysis_num=self.num_analyses-1)
        # remove all fits
        for i in range(an.num_fits):
            an.remove_fit(0)
    
    def use_fit_as_data(self, an_num, fit_num):
        # get analysis and fit object
        an_orig = self.get_analysis(analysis_num=an_num)
        if an_orig is None:
            return
        fits = copy.deepcopy(an_orig.return_fit_obj(fit_num=fit_num))
        if fits is None:
            return
        # copy analysis and fit
        self.analysis_list.append(copy.deepcopy(an_orig))
        an = self.get_analysis(analysis_num=self.num_analyses-1)
        # remove all fits
        for i in range(an.num_fits):
            an.remove_fit(0)
        # fill correlation analysis with fit as correlation data
        elements = an.settings.elements # central, sum3x3, sum5x5
        for element in elements:
            Gsingle = an.get_corr(element)
            fit = fits.fit_all_curves
            if Gsingle is not None and fit[0].fitfunction_label not in ['Model-free displacement analysis', 'Mean squared displacement']:
                for j in range(len(fit)):
                    if element == fit[j].data:
                        # fit found, add fitres to all chunks
                        for c in range(int(an.num_chunks(duration=self.metadata.duration))):
                            Gchunk = an.get_corr(element + '_chunk' + str(c))
                            fitres = fit[j].fitresult
                            fitrange = fit[j].fitrange
                            start = fitrange[0]
                            stop = fitrange[1]
                            xout = Gchunk[start:stop, 0]
                            yout = Gchunk[start:stop, 1] - fitres
                            Gout = np.zeros((len(xout), 2))
                            Gout[:,0] = xout
                            Gout[:,1] = yout
                            setattr(an.corrs, element + '_chunk' + str(c), Gout)
                            an.settings.update(chunks_off=an.settings.chunks_off, analysis=an)
                            
                        
    def get_analysis(self, analysis_num=-1):
        if analysis_num == -1:
            # return active analysis
            analysis_num = self.active_analysis
        if analysis_num is not None and analysis_num < self.num_analyses:
            return self.analysis_list[analysis_num]
        return None
    
    def update(self, fname=None, label=None, coords=None, timetrace=None, airy=None, active_analysis=None):
        if label is not None:
            self.label = label
        if coords is not None:
            self.metadata.coords = coords
        if timetrace is not None:
            self.timetrace = timetrace
        if airy is not None:
            self.airy= airy
        if active_analysis is not None:
            if active_analysis == 'None':
                active_analysis = None
            self.active_analysis = active_analysis


class FFSmetadata:
    def __init__(self,
                 num_pixels = None,
                 num_lines = None,
                 num_frames = None,
                 range_x = None,
                 range_y = None,
                 range_z = None,
                 num_datapoints = None,
                 hold_off_x5 = None,
                 hold_off = None,
                 time_resolution = None,
                 dwelltime = None,
                 duration = None,
                 pxsize = None,
                 coords = None
                 ):
        self.num_pixels = num_pixels
        self.num_lines = num_lines
        self.num_frames = num_frames
        self.range_x = range_x # µm
        self.range_y = range_y # µm
        self.range_z = range_z # µm
        self.num_datapoints = num_datapoints
        self.hold_off_x5 = hold_off_x5 # ns
        self.hold_off = hold_off # ns
        self.time_resolution = time_resolution # µs
        self.dwelltime = dwelltime # s
        self.duration = duration # s
        self.pxsize = pxsize # µm
        self.coords = coords # row, column number [y, x] of FFS position


class FFScorr:
    # for each file multiple FFScorr analysis objects can be added for different
    # types of analyses, e.g. spot-variation, iMSD, etc.
    def __init__(self,
                 mode=None,
                 settings=None,
                 corrs=None,
                 fits=None,
                 active_fit=None):
        self.mode = mode
        self.settings = settings if settings is not None else CorrSettings()
        self.corrs = corrs
        self.fits = fits if fits is not None else [] # list with fit results
        self.active_fit = active_fit # choose which fit to show by default
    
    @property
    def n_curves_mode(self):
        return len(self.settings.elements)
    
    @property
    def num_fits(self):
        return len(self.fits)
    
    def analysis(self, correlationObj, resolution, chunksize, chunks_off, algorithm, active_fit=-1, det_type='Genoa Instruments 5x5'):
        self.mode = correlationObj.mode
        
        self.settings.elements = correlationObj.elements
        self.settings.list_of_g = correlationObj.list_of_g
        self.settings.average = correlationObj.average
        
        self.settings.resolution = resolution
        self.settings.algorithm = algorithm
        self.settings.chunksize = chunksize # s
        self.settings.chunks_off = chunks_off
        if active_fit == -1:
            active_fit = self.num_fits
        self.active_fit = active_fit
    
    def num_chunks(self, duration):
        if duration is not None:
            N = int(np.floor(duration / self.settings.chunksize))
            return N
        return None
    
    def remove_fit(self, fit_num):
        if fit_num is not None and fit_num < self.num_fits:
            del self.fits[fit_num]
            if self.num_fits == 0 or self.active_fit is None:
                self.active_fit = None
            else:
                self.active_fit = np.min((self.num_fits-1, self.active_fit))
    
    def return_fit_obj(self, fit_num=-1):
        # by default return active fit object
        f = self.fits
        if fit_num == -1:
            fit_num = self.active_fit
        if fit_num is not None and len(f) > fit_num:
            return f[fit_num]
        return None
    
    def get_corr(self, corrtype="random"):
        # return average correlation central, sum3x3, sum5x5, etc.
        Gall = self.corrs
        if Gall is None:
            return None
        if corrtype == "random":
            excluded_keys = {"dwellTime", "crossCenterAv", "chunksOff", "chunks_off"}
            # Get the filtered list of keys
            keys = [key for key in list(Gall.__dict__.keys()) if key not in excluded_keys]
            Gsingle = getattr(Gall, keys[0])
        elif 'crossCenterAv' in corrtype:
            if 'chunk' in corrtype:
                chunk = corrtype[len('crossCenterAv'):]
                Gsingle = fcs_crosscenter_av(Gall, returnField=chunk, returnObj = False)
            else:
                Gsingle = fcs_crosscenter_av(Gall, returnObj = False)
        else:
            try:
                Gsingle = getattr(Gall, corrtype + "_averageX")
            except:
                try:
                    Gsingle = getattr(Gall, corrtype + "_average")
                except:
                    Gsingle = getattr(Gall, corrtype)
        return Gsingle
    
    def analysis_summary(self):
        algorithm = str(self.settings.algorithm) if self.settings.algorithm is not None else "unknown algorithm"
        return self.mode + " (settings: " + str(self.settings.resolution) + "/" + str(self.settings.chunksize) + "/" + algorithm + ")"
    
    def calc_corr(self):
        # calculate correlation only if not yet calculated before
        if self.corrs is None:
            return True
        return False
    
    def update(self, mode=None, settings=None, corrs=None, fits=None, active_fit=None):
        # mainly used to store calculated correlations in analysis object
        if mode is not None:
            self.mode = mode
        if settings is not None:
            self.settings = settings
        if corrs is not None:
            self.corrs = corrs
        if fits is not None:
            self.fits = fits
        if active_fit is not None:
            # use active_fit = 'None' to set active fit to None
            if active_fit == 'None':
                active_fit = None
            self.active_fit = active_fit
    
    def corr_param(self):
        # return all parameters needed to perform autocorrelation calculation
        return self.mode, self.settings.resolution, self.settings.chunksize
    
    def add_fit_analysis(self, fitfunctionmodel, fitarray, startvalues, fitrange=[1, -1]):
        # fitarray always has length 12 (11 parameters + weighted fit)
        mode = self.mode
        modelname = fitfunctionmodel.model
        if any(fitarray[:-1]) > 0 or modelname in ['Maximum entropy method free diffusion', 'Flow heat map', 'Asymmetry heat map', 'Model-free displacement analysis']:
            corrFitObj = CorrFit()
            corrFitObj.add_fit_analyses(mode, fitfunctionmodel, fitarray, startvalues, fitrange, self.settings.elements)
            self.fits.append(corrFitObj)
            self.active_fit = self.num_fits - 1
    
    def update_fit_analysis(self, fitfunctionmodel, fitarray, startvalues, fitrange=[1, -1], fit_num=-1):
        mode = self.mode
        if fit_num == -1:
            fit_num = self.active_fit
        if any(fitarray) > 0 or fitfunctionmodel.model in ['Maximum entropy method free diffusion', 'Asymmetry heat map', 'Flow heat map', 'Model-free displacement analysis']:
            corrFitObj = self.fits[fit_num]
            corrFitObj.fit_all_curves = [] # empty the object and fill it from scratch
            corrFitObj.add_fit_analyses(mode, fitfunctionmodel, fitarray, startvalues, fitrange, self.settings.elements)
                    
        
class CorrSettings():
    def __init__(self,
                 elements = None,
                 list_of_g = None,
                 resolution = None,
                 algorithm = None,
                 chunksize = None,
                 chunks_off = None,
                 average = None):
        self.elements   = elements # fields that are returned by fcs2corr
        self.list_of_g  = list_of_g # what is sent to fcs2corr
        self.resolution = resolution
        self.algorithm  = algorithm
        self.chunksize  = chunksize
        self.chunks_off = chunks_off
        self.average    = average # which cross-correlations should be averaged for flow analysis
    
    def update(self, elements=None, resolution=None, chunksize=None, chunks_off=None, analysis=None, algorithm=None, list_of_g=None):
        if elements is not None:
            self.elements = elements
        if list_of_g is not None:
            self.list_of_g = list_of_g
        if algorithm is not None:
            self.algorithm = algorithm
        if resolution is not None:
            self.resolution = resolution
        if chunksize is not None:
            self.chunksize = chunksize
        if chunks_off is not None:
            # if chunks_off is changed, also the average correlation is changed
            self.chunks_off = chunks_off
            G = analysis.corrs
            idx = np.nonzero(chunks_off)
            idx = list(idx[0]) # list of indices of good chunks
            try:
                G = fcs_av_chunks(G, idx)
                try:
                    G = fcs_crosscenter_av(G, returnField='_averageX')
                except:
                    pass
                analysis.update(corrs=G)
            except:
                pass
            

class CorrFit():
    def __init__(self):
        self.fit_all_curves = [] # list with each element consisting of 3 objects with fits for central, sum3, sum5
    
    @property
    def num_fitcurves(self):
        return len(self.fit_all_curves)

    def return_field(self, field):
        if field == "fitAllCurves":
            return self.fit_all_curves
        if field == "w0":
            w0 = []
            num_curves = len(self.fit_all_curves)
            for i in range(num_curves):
                w0.append(self.fit_all_curves[i].w0)
            return w0
        if field == "D":
            D = []
            num_curves = len(self.fit_all_curves)
            for i in range(num_curves):
                D.append(self.fit_all_curves[i].D)
            return D
        fits = self.fit_all_curves
        if len(fits) == 0:
            return None
        return fits[0].return_field(field)
    
    def return_all(self, field):
        fits = self.fit_all_curves
        if len(fits) == 0:
            return None
        data = []
        for i in range(len(fits)):
            data.append(getattr(fits[i], field))
        
        return data
    
    def fitrange(self):
        fits = self.fit_all_curves
        if len(fits) == 0:
            return None
        fit = fits[0]
        return fit.fitrange
    
    def fitresults(self, returntype="string"):
        # return 12 fit start values for the (3) fcs curves, either as strings or as 2d array
        num_curves = len(self.fit_all_curves)
        num_param = 12
        stv = ["NaN" for j in range(num_param)]
        bls = [False for j in range(num_param)]
        fitfunction = None
        fitres_array = np.zeros((num_param, num_curves))
        for i in range(num_curves):
            fit = self.fit_all_curves[i]
            fitfunction = fit.fitfunction_label
            power10 = fit.param_factors10
            for j in range(len(fit.paramidx)):
                fitabsv = fit.startvalues[fit.paramidx[j]] / power10[j]
                fitres_array[j, i] = fitabsv
                if np.abs(fitabsv) < 1e-2 or np.abs(fitabsv) > 999:
                    fitresString = str(fitabsv)[0:10]
                else:
                    fitresString = str(fitabsv)[0:4]
                if i == 0:
                    stv[j] = fitresString
                else:
                    stv[j] += fitresString
                if i < num_curves - 1:
                    stv[j] += ", "
                bls[j] = bool(fit.fitarray[fit.paramidx[j]])
        bls[-1] = bool(fit.fitarray[-1])
        if returntype == "string":
            return [stv, bls, fitfunction]
        else:
            return fitres_array
    
    def fitresults_mfda(self):
        # return scatter plot for model-free displacement analysis
        num_curves = len(self.fit_all_curves)
        N = 5 # pixels per dimension
        difftimesarray = np.zeros((N, N))
        corrvarray = np.ones((N, N))
        Nfound = 0
        for vert in range(N):
            for hor in range(N):
                # find the right fit
                for c in range(num_curves):
                    if self.fit_all_curves[c].data == 'V'+str(vert-int(np.floor(N/2)))+'_H'+str(hor-int(np.floor(N/2))):
                        fitr = self.fit_all_curves[c].fitresult
                        difftimesarray[vert, hor] = self.fit_all_curves[c].fitresult[0]
                        if len(fitr) > 1:
                            corrvarray[vert, hor] = self.fit_all_curves[c].fitresult[1]
                        Nfound += 1
        if Nfound > 0:
            return difftimesarray, corrvarray
        
        return None
    
    def fitresults_asymmetrymap(self):
        # return asymmetry heat map
        num_curves = len(self.fit_all_curves)
        fit = self.fit_all_curves[-1]
        
        if fit.fitresult is None:
            return
        
        N = int(len(fit.fitresult))
        
        if num_curves == 4:
            columnorder = ['Right', 'Up', 'Left', 'Down'] # square array detector
        elif num_curves == 6:
            columnorder = ['Right', 'UpRight', 'UpLeft', 'Left', 'DownLeft', 'DownRight'] # airy detector
        else:
            return
        
        allfits = np.zeros((N, len(columnorder)))
        columnNotFound = False
        for i in range(num_curves):
            try:
                c = columnorder.index(self.fit_all_curves[i].data)
                allfits[:, c] = self.fit_all_curves[i].fitresult
            except:
                columnNotFound = True
        
        if columnNotFound:
            return np.zeros((5,5)), columnNotFound
        
        z = g2polar(allfits)
    
        return z, columnNotFound
    
    def fitresults_flowmap(self):
        # return flow heat map
        num_curves = len(self.fit_all_curves)
        fit = self.fit_all_curves[-1] # color as a function of radius
        
        if fit.fitresult is None:
            return
        
        N = int(len(fit.fitresult))
        
        dets = ['square', 'airy']
        columnorder_square = ['Right', 'Up', 'Left', 'Down'] # square array detector
        columnorder_airy = ['UpRight', 'UpLeft', 'DownLeft', 'DownRight'] # airy detector
        
        for det, columnorder in enumerate([columnorder_square, columnorder_airy]):
            allfits = np.zeros((N, len(columnorder)))
            columnNotFound = False
            for i in range(num_curves):
                try:
                    c = columnorder.index(self.fit_all_curves[i].data)
                    allfits[:, c] = self.fit_all_curves[i].fitresult
                except:
                    columnNotFound = True
            if not columnNotFound:
                break
        
        if columnNotFound:
            return np.zeros((5,5)), [0,0], columnNotFound
        
        z, flow = g2flow(allfits, detector=dets[det])
        u = 2*flow[0]
        r = 2*flow[1]
    
        return z, [r, u], columnNotFound
    
    def fitresults_mem(self, tau, nparam=5):
        # return diffusion times distributions for MEM fit
        num_curves = len(self.fit_all_curves)
        fit0 = self.fit_all_curves[0]
        
        taumin = np.log10(tau[fit0.fitrange[0]])
        taumax = np.log10(tau[fit0.fitrange[1]-1])
        tauD = np.logspace(taumin, taumax, len(fit0.startvalues[0:-nparam]))
        
        fitresArray = np.zeros((len(tauD), num_curves))
        for i in range(num_curves):
            fit = self.fit_all_curves[i]
            fitresArray[:, i] = fit.startvalues[0:-nparam]
        
        fit = self.fit_all_curves[0]
        power10 = fit.param_factors10
        num_param = 12
        stv = ["NaN" for j in range(num_param)]
        bls = [False for j in range(num_param)]
        for j in range(len(fit.paramidx)):
            fitabsv = fit.startvalues[-1-nparam+fit.paramidx[j]] / power10[j]
            stv[j] = fitabsv
            bls[j] = bool(fit.fitarray[fit.paramidx[j]])
        stv[0] = len(tauD)
        
        return fitresArray, tauD, stv, bls
    
    def fitresults_msd(self):
        fit0 = self.fit_all_curves[0]
        D = fit0.startvalues[0]
        rho = 1e-3 * fit0.startvalues[3] # µm
        slope = 2 / rho**2 * D
        offset = fit0.startvalues[1]
        tauvar = fit0.fitresult
        tau = tauvar[0]
        var = tauvar[1]
        varfit = tau * slope + offset
        
        return tau, var, varfit
            
    def add_fit_analyses(self, mode, fitmodel, fitarray, startvalues, fitrange=[1, -1], data=[]):
        for i in range(len(data)):
            self.fit_all_curves.append(FitSingleObj(data[i], fitmodel, fitarray, startvalues[:,i], fitrange))
    

class FitSingleObj:
    def __init__(self, data, fitmodel, fitarray, startvalues, fitrange):
        # fitmodel is an object of the class fitModels
        
        Nparam = fitmodel.num_param + 1 # weighted fit
        self.minbound = np.array(fitmodel.param_minbound)
        self.maxbound = np.array(fitmodel.param_maxbound)
        self.param_factors10 = np.array(fitmodel.param_factors10)
        self.data = data # central, sum3, sum5, etc.
        self.fitfunction_label = fitmodel.model # more readable name of the fit function
        self.fitrange = fitrange
        self.fitresult = None # fit residuals
        
        paramind = fitmodel.fitfunction_param_used
        
        self.fitfunction = fitmodel.fitfunction_name # fit function used for the calculation (not a string but a function)
        fitarrayTemp = np.array([False for i in range(Nparam)])
        startvTemp = np.array(fitmodel.all_param_def_values).astype(float)
        if paramind is not None:
            for i in range(len(paramind)):
                fitarrayTemp[paramind[i]] = fitarray[i]
                startvTemp[paramind[i]] = float(startvalues[i]) * fitmodel.param_factors10[i]
            fitarrayTemp[Nparam-1] = fitarray[-1]
        else:
            fitarrayTemp = None
            startvTemp = None
        
        self.fitarray = fitarrayTemp
        self.startvalues = startvTemp
        self.paramidx = paramind # array indices with the parameters of interest (for startvalues and fitarray)
        self.w0 = None # beam waist (nm)
        self.D = None # diffusion coefficient (µm^2/s)
    
    def update(self, fitresult=None, startvalues=None, w0=None, D=None):
        # start values contains two lists [startv, fitv]
        # with startv containing the start values of both the fitted and unfitted parameters
        # fitv the 
        # start values are [M x 1] vector with M the 12 (or 7 for circFCS) fit parameters
        if fitresult is not None:
            self.fitresult = fitresult
        if startvalues is not None:
            self.startvalues = startvalues
        if w0 is not None:
            self.w0 = w0
        if D is not None:
            self.D = D