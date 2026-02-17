import numpy as np
import pandas as pd
import h5py
import os
import copy

from ..tools.list_files import list_files
from ..tools.closefile import closefile
from ..tools.moving_average import moving_average

import libttp.ttp as ttp
import ptufile


class ATimesData:
    def __init__(self):
        self.macrotime = 1e-12
        self.microtime = 1e-12
    
    @property
    def all_channels(self):
        all_ch = [int(k[3:]) for k in dir(self) if k.startswith('det')]
        all_ch = ['det'+str(i) for i in sorted(all_ch)]
        return all_ch
    
    @property
    def num_channels(self):
        return len(self.all_channels)
    
    def get_channels(self, startswith='det'):
        all_ch = [int(k[len(startswith):]) for k in dir(self) if k.startswith(startswith)]
        all_ch = [startswith+str(i) for i in sorted(all_ch)]
        return all_ch
    
    def calc_hist(self, bintime='auto', ttm_correction=False, laser_freq=40e6):
        """
        Calculate histograms from ttm data

        Parameters
        ----------
        bintime : float, optional
            Bin time in ps. The default is 'auto'.
        ttm_correction : boolean, optional
            Only True for ttm data. The default is False.
        n_bins : TYPE, optional
            DESCRIPTION. The default is 576.
        laser_freq : TYPE, optional
            DESCRIPTION. The default is 40e6.

        Returns
        -------
        None.

        """
        all_ch = self.all_channels
        
        if bintime == 'auto':
            data_single_ch = getattr(self, all_ch[0])
            bintime = np.min(data_single_ch[data_single_ch>0])
            
        max_microtime = 1e12 * atimes_data_2_duration(self, 1e-12, subtract_start_time=False, return_period=True) # ps
        
        for det in all_ch:
            microtime = getattr(self, det)[:,1]
            if ttm_correction:
                macrotime = getattr(self, det)[:,0] # ps
                microtime = np.mod(microtime, 1e12 / laser_freq)
                microtime = -microtime + np.max(microtime)
                setattr(self, det, np.transpose([macrotime, microtime]))
            tcspc_bins = np.arange(0, max_microtime, bintime)
            [Ihist, lifetime_bins] = np.histogram(microtime, tcspc_bins)
            setattr(self, "hist" + det[3:], np.transpose(np.stack((lifetime_bins[0:-1], Ihist))))
        
    
    def align_hist(self, n=15):
        """
        Align lifetime histograms of different channels. Calculate moving averaged
        histograms, caculate derivative and find peak. Then shift everything
        so that the peaks are aligned.

        Parameters
        ----------
        n : int, optional
            width of the window for the calculation of the moving average. The default is 15.

        """
        
        histList = self.get_channels('hist')
        
        hshifts = np.zeros(len(histList))
        
        i = 0
        for hist in histList:
            # get histogram
            histD = copy.deepcopy(getattr(self, hist))
            # calculate moving average
            IAv = moving_average(histD[:,1], n=n)
            # calculate derivative
            derivative = IAv[1:] - IAv[0:-1]
            # find maximum of derivative
            maxInd = np.argmax(derivative)
            # shift histogram
            histD[:,1] = np.roll(histD[:,1], -maxInd)
            # store shifted histogram in data object
            setattr(self, 'A' + hist, histD)
            # store shift value
            hshifts[i] = maxInd
            i += 1
        setattr(self, 'hshifts', hshifts)


def load_atimes_data(fname, channels='auto', sysclk_MHz=240, perform_calib=True):
    """
    Load multichannel arrival times data from .hdf5 file and store in data object
    for further processing. Used for data collected by the BrightEyes-TTM.

    Parameters
    ----------
    fname : string
        Filename [.hdf5]
        containing the tables ['det0', 'det1', 'det2', ..., 'det20']
        with each table containing the macro and microtimes.
    channels : int or list of int, optional
        Either number describing the number of channels (typically 21 or 25)
        or list of channels that have to be loaded, e.g. [15, 17, 18].
        The default is all channels, 'auto'.
    sysclk_MHz : int, optional
        System clock frequency [MHz]. The default is 240.
        If data is already in [macrotime, microtime] format, then this is not used
    perform_calib : boolean
        If data is already in [macrotime, microtime] format, then False
        If True: the calibration to convert to the right format is performed

    Returns
    -------
    data : object
        Data object with a field for each channel
        Each field contains a [Np x 2] np.array
        with Np the number of photons,
        column 1 the absolute macrotimes,
        column 2 the absolute microtimes.

    """
    
    data = ATimesData()
    
    # -------------------- HDF5 file --------------------
    if fname[-4:] == "hdf5" or (fname[-3:] == ".h5" and perform_calib == False):
        if isinstance(channels, int):
            # total number of channels is given, e.g. 21
            Nchannels = channels
            channels = [str(x) for x in range(channels)]
        elif isinstance(channels, str) and channels == 'auto':
            # load all channels
            Nchannels = atimes_file_2_nch(fname)
            channels = [str(x) for x in range(Nchannels)]
        else:
            # individual channel numbers are given, e.g. [15, 17, 18]
            channels = [str(x) for x in channels]
            Nchannels = len(channels)
            
        with h5py.File(fname, 'r') as f:
            for ch in channels:
                setattr(data, 'det' + ch, f['det' + ch][()])
        f.close()
        
    # -------------------- PTU file --------------------
    elif fname[-4:] == ".ptu":
        ptu = ptufile.PtuFile(fname)
        decoded = ptu.decode_records()
        macrotime = decoded['time'].astype(float)
        channel = decoded['channel'].astype(float)
        microtime = decoded['dtime'].astype(float)
        
        if np.any(channel < 0):
            valid = channel >= 0
            macrotime = macrotime[valid]
            channel = channel[valid]
            microtime = microtime[valid]
        
        channel += 1
        channels_all = np.unique(channel)
        macrotime_res = ptu.global_resolution
        microtime_res = ptu.tcspc_resolution # in s
        macrotime *= macrotime_res * 1e12 # in ps
        microtime *= microtime_res * 1e12 # in ps
        
        if isinstance(channels, str) and channels == 'auto':
            # load all channels
            channels = channels_all
        
        for ch in channels:
            data_single_ch = np.column_stack((macrotime[channel==ch], microtime[channel==ch]))
            setattr(data, 'det' + str(int(ch)), data_single_ch)
    
    # -------------------- Raw TTM file --------------------
    elif fname[-2:] == "h5":
        
        if isinstance(channels, int):
            # total number of channels is given, e.g. 21
            Nchannels = channels
            channels = [str(x) for x in range(channels)]
        elif isinstance(channels, str) and channels == 'auto':
            # load all channels
            Nchannels = atimes_file_2_nch(fname)
            channels = [str(x) for x in range(Nchannels)]
        else:
            # individual channel numbers are given, e.g. [15, 17, 18]
            channels = [str(x) for x in channels]
            Nchannels = len(channels)
        
        calibDict=ttp.calculateCalibFromH5(filenameH5=fname, listChannel=range(0,Nchannels))
        data = ATimesData()
        for ch in channels:
            df = ttp.applyCalibDict(fname, channel=int(ch), calibDict=calibDict)
            macrotime = df['cumulative_step'] / (1e6 * sysclk_MHz) * 1e12
            microtime = df['dt_' + ch]            
            setattr(data, 'det' + ch, np.transpose([macrotime, microtime]))
        closefile(fname)
    
    return data


def write_atimes_data(data, channels, fname):
    """
    Write multichannel arrival times data to .hdf5

    Parameters
    ----------
    data : object
        Arrival times data, i.e. output from load_atimes_data_pandas().
    channels : int or list of int
        Either number describing the number of channels (typically 21)
        or list of channels that have to be loaded, e.g. [15, 17, 18].
    fname : str
        Filename [.hdf5].

    Returns
    -------
    

    """
    
    if isinstance(channels, int):
        # total number of channels is given, e.g. 21
        channels = [str(x) for x in range(channels)]
    else:
        # individual channel numbers are given, e.g. [15, 17, 18]
        channels = [str(x) for x in channels]
    
    with h5py.File(fname, 'w') as f:
        for ch in channels:
            f.create_dataset('det' + str(ch), data=getattr(data, 'det' + str(ch)))
    

def atimes_data_2_duration(data, macrotime=1.0, subtract_start_time=False, return_period=False):
    """
    Return duration of a TCSPC measurement or duration of a pulse period

    Parameters
    ----------
    data : Object
        With field det0, det1, etc. with each field a 2D np.array()
        with the first column the macrotimes, and the second column the microtimes.
    macrotime : float, optional
        Macrotime units in s. Typically the macrotimes are in ps, hence 1e-12.
        The default is 1.
    subtract_start_time : boolean
        Subtract the start time, so duration = last photon time - first photon time
    return_period : boolean
        If False: return duration of the measurement
        If True: return maximum microtime found

    Returns
    -------
    duration : float
        Measurement time in s. Calculated from the very last photon
        Or Maximum microtime

    """
    all_ch = atimes_data_2_channels(data)
    
    column = 0
    if return_period:
        column = 1
    
    # check for last photon
    atime_last_photon = 0
    for idx, ch in enumerate(all_ch):
        data_single_ch = getattr(data, ch)
        atime_last_photon = np.max([atime_last_photon, np.max(data_single_ch[:,column], 0)])
    
    # check for first photon
    atime_first_photon = atime_last_photon
    for idx, ch in enumerate(all_ch):
        data_single_ch = getattr(data, ch)
        atime_first_photon = np.min([atime_first_photon, np.min(data_single_ch[:,column], 0)])
    
    # duration in arbitrary units
    if subtract_start_time:
        atime_last_photon -= atime_first_photon
    
    # duration in s
    duration = macrotime * atime_last_photon
    
    return duration


def atimes_data_2_channels(data):
    all_ch = [int(k[3:]) for k in dir(data) if k.startswith('det')]
    all_ch = ['det'+str(i) for i in sorted(all_ch)]
    return all_ch


def atimes_data_attr_2_ch(attribute_name):
    # convert attribute name to channel number
    # e.g. 'det31' returns 31
    return int(attribute_name[3:])


def atimes_file_2_nch(fname):
    """
    Get number of channels from h5 TCSPC file

    Parameters
    ----------
    fname : string
        Path to h5 file or ptu file.

    Returns
    -------
    n_ch : int
        Number of channels, typically 25 for a 5x5 array.

    """
    
    if fname[-3:] == ".h5" or fname[-5:] == ".hdf5":
        with h5py.File(fname, "r") as f:
            datasets = []
            f.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
            n_ch = 0
            for d in datasets:
                if d.startswith('det'):
                    n_ch += 1
        return n_ch

    if fname[-4:] == ".ptu":
        ptu = ptufile.PtuFile(fname)
        decoded = ptu.decode_records()
        channel = decoded['channel'].astype(float)
        
        if np.any(channel < 0):
            valid = channel >= 0
            channel = channel[valid]
        
        channels_all = np.unique(channel)
        return len(channels_all)
    

def load_atimes_data_pandas(fname, chunksize=1000000, macro_freq=240e6):
    """
    Load multichannel arrival times data from h5 file and store in data object
    for further processing

    Parameters
    ----------
    fname : str
        Filename.
    chunksize : int, optional
        Number of rows to read in a single chunk. The default is 1000000.
    macro_freq : float, optional
        Conversion factor to go from relative macrotimes to absolute
        macrotimes. The default is 240e6.

    Returns
    -------
    data : object
        Data object with a field for each channel
        Each field contains a [Np x 2] np.array
        with Np the number of photons,
        column 1 the absolute macrotimes,
        column 2 the absolute microtimes.

    """
    
    # convert macrotime frequency to macrotime step size
    macroStep = 1 / macro_freq
    
    # read hdf file
    dataR = pd.read_hdf(fname, iterator=True, chunksize=chunksize)
    
    # number of data chunks to read
    Nchunks = int(np.ceil(len(dataR.coordinates) / chunksize))
    
    myIter = iter(dataR)
    
    chunk = 1
    for dataChunk in myIter:
        print('Loading data chunk ' + str(chunk) + '/' + str(Nchunks))
        if chunk == 1:
            # initialize data object
            data = ATimesData()
            listOfChannels = [name[12:] for name in dataChunk.columns if name.startswith('microtime_ch')]
            for chNr in listOfChannels:
                setattr(data, "det" + chNr, np.array([]))
        # go through each channel
        cumstep = dataChunk['cumulative_step']
        for chNr in listOfChannels:
            dataSingleCh = dataChunk['microtime_ch' + chNr]
            microtime = dataSingleCh[dataSingleCh.notna()]
            macrotime = macroStep * cumstep[dataSingleCh.notna()]
            dataSingleCh = np.transpose([macrotime, microtime])
            dataSingleChTot = getattr(data, "det" + chNr)
            setattr(data, "det" + chNr, np.vstack([dataSingleChTot, dataSingleCh]) if dataSingleChTot.size else dataSingleCh)
        chunk += 1
    
    return data


def load_atimes_data_newprot(filename, sysclk_MHz=240.0, laser_MHz= 80.0, nchannel = 26, kC4=43, channels=25):
    data_head, data_filename = os.path.split(filename)
    fname = data_filename[:-4]
    
    df = ttp.readNewProtocolFileToPandas(filenameIn=filename, CHANNELS=channels, reorder_channels=25)
    
    myReturn=ttp.convertFromPandasDataFrame(df, filenameOutputHDF5=data_head+fname+'.h5',
                    sysclk_MHz = sysclk_MHz,
                    laser_MHz=laser_MHz,
                    dwell_time_us=100.,
                    list_of_channels=np.arange(0,channels),
                    autoCalibration=True,
                    kC4=45.,
                    textInPlot=None,
                    compressionLevel=1,
                    makePlots=True,
                    ignorePixelLineFrame = False,
                    coincidence_analysis = True)
    
    fnameOut = myReturn['filenameH5']
    #h_main=pd.read_hdf(fnameOut, key="main")
    #h_valid_L = h_main[h_main["valid_tdc_L"]==1]
    data=load_atimes_data(fnameOut, channels=channels)
    data.macrotime = 1e-12
    data.microtime = 1e-12
    
    return data


def atimes_h5_to_hdf5(fname, chunksize=1000000, macro_freq=240e6, channels=21):
    """
    Load multichannel arrival times data from .h5 file, remove NaN and
    store as .hdf5 file

    Parameters
    ----------
    fname : str
        .h5 filename.
    chunksize : int, optional
        Number of rows to read in a single chunk. The default is 1000000.
    macro_freq : float, optional
        Conversion factor to go from relative macrotimes to absolute
        macrotimes. The default is 240e6.
    channels : int or list of int, optional
        Either number describing the number of channels (typically 21 or 25)
        or list of channels that have to be loaded, e.g. [15, 17, 18].
        The default is 21.

    Returns
    -------
    None.

    """
    
    data = load_atimes_data_pandas(fname, chunksize, macro_freq)
    write_atimes_data(data, channels, fname[:-3] + '.hdf5')


def atimes_raw_2_h5all(folder, sysclk_MHz=240, laser_MHz=40, n_ch=21, destination_folder=""):
    """
    Load multichannel arrival times data from raw data files and store as .h5
    files. Repeat this for all raw data files in a given folder.
    
    +------------------------------------------------------------------------+
    | > THIS IS USUALLY THE FIRST FUNCTION TO CAlL AFTER A TTM MEASUREMENT < |
    +------------------------------------------------------------------------+
    
    Note: this function expects the files to have the .ttm file extension
    Change this manually if not the case

    Parameters
    ----------
    folder : path
        Folder to look into.
    sysclk_MHz : float, optional
        System clock frequency [MHz]. The default is 240.
    laser_MHz : float, optional
        Laser clock frequency [MHz]. The default is 40.
    n_ch : int, optional
        Number of channels. The default is 21.
    destination_folder : path to folder, optional
        Location to store the .h5 file
        If left empty, the current working directory will be taken.
        The default is "".

    Returns
    -------
    Each file will be converted to a .h5 file.

    """
    
    files = list_files(directory=folder, filetype='ttm', substr=False)
    
    for file in files:
        atimes_raw_2_h5(file, sysclk_MHz=sysclk_MHz, laser_MHz=laser_MHz, Nch=n_ch, destinationFolder=destination_folder)
    print("All files converted.")


def atimes_raw_2_h5(fname, sysclk_MHz=240, laser_MHz=40, n_ch=21, destination_folder=""):
    """
    Load multichannel arrival times data from raw data file and store as .h5
    file.

    Parameters
    ----------
    fname : str
        Raw data file name.
    sysclk_MHz : float, optional
        System clock frequency [MHz]. The default is 240.
    laser_MHz : float, optional
        Laser clock frequency [MHz]. The default is 40.
    Nch : int, optional
        Number of channels. The default is 21.
    destination_folder : path to folder, optional
        Location to store the .h5 file
        If left empty, the current working directory will be taken.
        The default is "".

    Returns
    -------
    None.

    """
    
    
    list_of_channels=np.arange(0,n_ch)
    
    if destination_folder == "":
        destination_folder = os.getcwd()
    
    ttp.convertDataRAW(filenameToRead = fname,
                                sysclk_MHz = sysclk_MHz,
                                laser_MHz = laser_MHz,
                                list_of_channels = list_of_channels, # list of channel [0,1,2,3]
                                compressionLevel = 1,                # Compression HDF5 file
                                #metadata=metadataDict,              # If present append metadata to the HDF5
                                destinationFolder = "",              # If not selected the default output folder is filenameFolder/output/
                                ignorePixelLineFrame = True,         # Does not calculate x,y,frame
                                )
    print('Done.')


def h5_atimes_2_data(fname, n_ch, sysclk_MHz=240):
    """
    Load multichannel arrival times data from h5 file and store in data object
    for further processing - newer version

    Parameters
    ----------
    fname : str
        File name.
    Nch : int
        Number of channels.
    sysclk_MHz : float, optional
        System clock frequency [MHz]. The default is 240.

    Returns
    -------
    data : object
        Data object with a field for each channel
        Each field contains a [Np x 2] np.array
        with Np the number of photons,
        column 1 the absolute macrotimes,
        column 2 the absolute microtimes.

    """
    
    calibDict=ttp.calculateCalibFromH5(filenameH5=fname, listChannel=range(0,n_ch))
    data = ATimesData()
    for ch in range(n_ch):
        print('Loading channel ' + str(ch))
        df = ttp.applyCalibDict(fname, channel=ch, calibDict=calibDict)
        
        macrotime = df['cumulative_step'] / (sysclk_MHz * 1e6)
        microtime = df['dt_' + str(ch)]
        
        setattr(data, 'det' + str(ch), np.transpose([macrotime, microtime]))
    
    return data
