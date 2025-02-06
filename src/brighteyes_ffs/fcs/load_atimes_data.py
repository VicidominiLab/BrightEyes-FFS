import numpy as np
import pandas as pd
import h5py
import os

from .fcs2arrivaltimes import ATimesData
from ..tools.list_files import list_files
from ..tools.closefile import closefile

import libttp.ttp as ttp


def load_atimes_data(fname, channels=21, sysclk_MHz=240):
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
        The default is 21.
    sysclk_MHz : int, optional
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
    
    if isinstance(channels, int):
        # total number of channels is given, e.g. 21
        Nchannels = channels
        channels = [str(x) for x in range(channels)]
    else:
        # individual channel numbers are given, e.g. [15, 17, 18]
        channels = [str(x) for x in channels]
        Nchannels = len(channels)
    
    data = ATimesData()
    
    if fname[-4:] == "hdf5":
        with h5py.File(fname, 'r') as f:
            for ch in channels:
                print('Loading channel ' + ch)
                setattr(data, 'det' + ch, f['det' + ch][()])
        f.close()
    elif fname[-2:] == "h5":
        calibDict=ttp.calculateCalibFromH5(filenameH5=fname, listChannel=range(0,Nchannels))
        data = ATimesData()
        for ch in channels:
            print('Loading channel ' + ch)
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
        for ch in range(21):
            f.create_dataset('det' + str(ch), data=getattr(data, 'det' + str(ch)))


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
