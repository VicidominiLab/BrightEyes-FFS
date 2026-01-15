from pathlib import Path
from os import path as pth
from ..tools.h5data import h5data
from .atimes_data import atimes_data_2_duration, load_atimes_data
from tifffile import TiffFile
import czifile
import h5py

class infoObj:
    pass


def get_metafile_from_file(fname, metastring='_info.txt'):
    """
    Get metadata file from fcs data file
    If data file is stored in .bin file --> then metadata is in info.txt file
    If data file is stored in .h5 file --> then metadata is in same .h5 file
    If data file is stored in .tiff file --> then metadata is in same .tiff file

    Parameters
    ----------
    fname : string
        File name.
    metastring : string, optional
        String of the metafile. The default is '_info.txt'.

    Returns
    -------
    TYPE
        File name of the metadata.

    """
    
    head_tail = pth.split(fname)
    folder = head_tail[0]
    fname_stripped = head_tail[1]
    root_ext = pth.splitext(fname_stripped)
    fname_stripped = root_ext[0]
    fname_ext = root_ext[1]
    if fname_ext != '.bin':
        return fname
    metafile = pth.join(folder, fname_stripped + metastring)
    return metafile


def get_fcs_info(fname, parameter=['all']):
    """
    Get info from fcs info file (old version)

    Parameters
    ----------
    fname : TYPE
        File name [.txt].
    parameter : list of strings, optional
        Name of the parameter to return
        E.g. "READING SPEED [kHz]" will return the reading speed
        Leave blank to return an object containing all info. The default is ['all'].

    Returns
    -------
    info : object or number
        Either an object will all information
        Or the requested parameter.

    """
    
    # PARSE INPUT
    fname = fname.replace("\\", "/")
    fname = Path(fname)
    
    f = open(fname, "r", encoding="ISO8859-1") 
    if f.mode == "r":
        contents = f.read()  # in the case of error remove the encoding ="ISO8859-1"
        if parameter == ['all']:
            output = infoObj()
            info = float(get_fcs_info_singleparam(contents, "DURATION [s]"))
            output.duration = info
            info = float(get_fcs_info_singleparam(contents, "TOTAL NUMBER OF DATA POINTS"))
            output.numberOfDataPoints = info
            info = float(get_fcs_info_singleparam(contents, "HOLD-OFF [x5 ns]"))
            output.holdOffx5 = info
            output.holdOff = info * 5
            info = float(get_fcs_info_singleparam(contents, "READING SPEED [kHz]"))
            output.readingSpeed = info
            output.dwellTime = 1e-3 / info # s
            output.timeResolution = 1e6 * output.dwellTime # µs
            return output
        else:
            output = get_fcs_info_singleparam(contents, parameter)
            return float(output)


def get_file_info(fname, parameter=['all']):
    """
    Get info from fcs/imaging info file

    Parameters
    ----------
    fname : TYPE
        File name [*_info.txt] or [*.h5].
    parameter : TYPE, optional
        Name of the parameter to return
        E.g. "READING SPEED [kHz]" will return the reading speed
        Leave blank to return an object containing all info
        For h5 files, it is always an object with all info. The default is ['all'].

    Returns
    -------
    finfo : object or number
        Either an object will all information
        Or the requested parameter.

    """
    
    head_tail = pth.split(fname)
    fname_stripped = head_tail[1]
    root_ext = pth.splitext(fname_stripped)
    fname_ext = root_ext[1]
    
    if fname_ext == '.txt':
        # read from text file
        finfo = get_file_info_txt(fname, parameter=['all'])
        return finfo
    elif fname_ext == '.h5':
        # read from h5 file
        finfo = get_file_info_h5(fname)
        return finfo
    elif fname_ext == '.tiff':
        # read from tiff file
        finfo = get_file_info_tiff(fname)
        return finfo
    elif fname_ext == '.czi':
        # read from czi file
        finfo = get_file_info_czi(fname)
        return finfo
    elif fname_ext == '.ptu':
        # read from ptu file
        finfo = get_file_info_ptu(fname)
        return finfo
    
def get_file_info_txt(fname, parameter=['all']):
    """
    Get fcs parameter info from metadata file

    Parameters
    ----------
    fname : string
        File name [.._info.txt].
    parameter : list of strings, optional
        Name of the parameter to return
        E.g. "READING SPEED [kHz]" will return the reading speed
        Leave blank to return an object containing all info. The default is ['all'].

    Returns
    -------
    info : object or number
        Either an object will all information
        Or the requested parameter.

    """
    
    # PARSE INPUT
    fname = fname.replace("\\", "/")
    fnameP = Path(fname)
    
    try:
        f = open(fnameP, "r", encoding="ISO8859-1") 
    except:
        f = None
            
    if f is not None and f.mode == "r":
        contents = f.read()
        
        if contents.find("READING SPEED") != -1:
            # info file is old, first version with reading speed
            output = get_fcs_info(fname)
            return output
        
        elif parameter == ['all']:
            output = infoObj()
            
            info = float(get_fcs_info_singleparam(contents, "NUMBER OF TIME BINS PER PIXEL"))
            output.numberOfTbinsPerPixel = info
            
            info = float(get_fcs_info_singleparam(contents, "NUMBER OF PIXELS"))
            output.numberOfPixels = info
            
            info = float(get_fcs_info_singleparam(contents, "NUMBER OF LINES"))
            output.numberOfLines = info
            
            info = float(get_fcs_info_singleparam(contents, "NUMBER OF FRAMES"))
            output.numberOfFrames = info
            
            info = float(get_fcs_info_singleparam(contents, "RANGE X [µm]"))
            output.rangeX = info
            
            info = float(get_fcs_info_singleparam(contents, "RANGE Y [µm]"))
            output.rangeY = info
            
            info = float(get_fcs_info_singleparam(contents, "RANGE Z [µm]"))
            output.rangeZ = info
            
            info = float(get_fcs_info_singleparam(contents, "TOTAL NUMBER OF DATA POINTS"))
            output.numberOfDataPoints = info
            
            info = float(get_fcs_info_singleparam(contents, "HOLD-OFF [x5 ns]"))
            output.holdOffx5 = info
            output.holdOff = info * 5
            
            info = float(get_fcs_info_singleparam(contents, "TIME RESOLUTION [µs]"))
            output.timeResolution = info
            
            output.dwellTime = output.timeResolution * output.numberOfTbinsPerPixel * 1e-6 # s
            output.duration = output.dwellTime * output.numberOfPixels * output.numberOfLines * output.numberOfFrames # s
            output.pxsize = output.rangeX / output.numberOfPixels
            
            return output
        else:
            output = get_fcs_info_singleparam(contents, parameter)
            return float(output)
    else:
        return f


def get_file_info_h5(fname):
    """
    Get metadata from fcs/imaging .h5 file
    
    Parameters
    ----------
    fname : string
        File name [..h5].

    Returns
    -------
    output : object or number
        An object will all information
        Or the requested parameter.

    """
    
    file = h5data(fname)
    
    output = infoObj()
    
    try:
        output.numberOfTbinsPerPixel = file.tbinpx()
    except:
        output.numberOfTbinsPerPixel = 1
    try:
        output.numberOfPixels = file.nx()
    except:
        output.numberOfPixels = 1
    try:
        output.numberOfLines = file.ny()
    except:
        output.numberOfLines = 1
    try:
        output.numberOfFrames = file.nz()
    except:
        output.numberOfFrames = 1
    try:
        output.rangeX = file.rangex()
    except:
        output.rangeX = 0
    try:
        output.rangeY = file.rangey()
    except:
        output.rangeY = 0
    try:
        output.rangeZ = file.rangez()
    except:
        output.rangeZ = 0
    try:
        output.numberOfDataPoints = file.ndatapoints()
    except:
        output.numberOfDataPoints = 0
    try:
        output.timeResolution = file.tres()
    except:
        output.timeResolution = 1e-6 # us
    try:
        output.dwellTime = file.tres() * 1e-6 # s
    except:
        output.dwellTime = 1e-12 # s
    try:
        output.duration = file.duration() # s
    except:
        output.duration = get_duration_from_h5_timetagging_data(fname)
    try:
        output.pxsize = file.pxsize()
    except:
        output.pxsize = 0
    
    return output


def get_file_info_ptu(fname):
    """
    Get metadata from ptu file (under development)
    
    Parameters
    ----------
    fname : string
        File name [*.ptu].

    Returns
    -------
    output : object
        An object will all information

    """
    output = infoObj()
    raw_data = load_atimes_data(fname, channels='auto', perform_calib=False)
    raw_data.macrotime = 1e-12
    raw_data.microtime = 1e-12
    output.duration = atimes_data_2_duration(raw_data, macrotime=raw_data.macrotime, subtract_start_time=False)
    output.numberOfTbinsPerPixel = 1
    output.numberOfPixels = 1
    output.numberOfLines = 1
    output.numberOfFrames = 1
    output.rangeX = 0
    output.rangeY = 0
    output.rangeZ = 0
    output.numberOfDataPoints = 0
    output.timeResolution = 1e-6 # us
    output.dwellTime = 1e-12 # s
    output.pxsize = 0
    
    return output


def get_file_info_czi(fname):
    """
    Get metadata from czi file (under development)
    
    Parameters
    ----------
    fname : string
        File name [*.czi].

    Returns
    -------
    output : object
        An object will all information

    """
    
    with czifile.CziFile(fname) as czi:
        # Access dimensions and shape
        dims_shape = czi.shape  # Shape of the image data
    
    output = infoObj()
    
    output.numberOfTbinsPerPixel = dims_shape[4] * dims_shape[7]
    output.numberOfPixels = 1
    output.numberOfLines = 1
    output.numberOfFrames = 1
    
    output.timeResolution = get_file_info_txt(fname[:-4] + '.txt', parameter="DWELL TIME [us]") # us
    if output.timeResolution is None:
        output.timeResolution = 1
    
    output.dwellTime = get_file_info_txt(fname[:-4] + '.txt', parameter="DWELL TIME [us]") # us
    if output.dwellTime is None:
        output.dwellTime = 1.2e-6 # s
        output.timeResolution = 1.2 # us
    else:
        output.dwellTime *= 1e-6 # s
   
    output.duration = output.dwellTime * dims_shape[4]*dims_shape[7] # s
    
    return output

def get_file_info_tiff(fname):
    """
    Get metadata from fcs/imaging .tiff file
    
    Parameters
    ----------
    fname : string
        File name [*.tiff].

    Returns
    -------
    output : object
        An object will all information

    """
    output = infoObj()
    
    with TiffFile(fname) as tif:
        # Access the first image (page) and its tags
        tags = tif.pages[0].tags  # Retrieve metadata tags for the first image
        
        # check time resolution
        for tag in ["timeResolution", "time_resolution", "timeresolution", "dwellTime", "dwelltime", "dwell_time"]: # in us
            custom_tag = tags.get(tag)
            if custom_tag:
                output.timeResolution = custom_tag.value
                output.dwellTime = custom_tag.value * 1e-6 # s
            else:
                output.timeResolution = 1
                output.dwellTime = 1e-6 # s
            
        output.duration = max(tif.pages[0].shape) * output.dwellTime
    
    return output

def get_fcs_info_singleparam(contents, parameter):
    # get single parameter from fcs info file
    start = contents.find(parameter)
    start = start + len(parameter) + 1
    stop = contents.find("\n", start)
    if stop == -1:
        stop = len(contents)
    return contents[start:stop]

def get_duration_from_h5_timetagging_data(fname):
    max_values = {}
    with h5py.File(fname, 'r') as f:
        for name in f.keys():
            if name.startswith('det'):
                data = f[name][()]  # read the dataset as a NumPy array
                if data.size > 0:
                    max_values[name] = data.max()
                else:
                    max_values[name] = None
    abs_max = 1e-12 * max(abs(v) for v in max_values.values() if v is not None) # s
    return abs_max