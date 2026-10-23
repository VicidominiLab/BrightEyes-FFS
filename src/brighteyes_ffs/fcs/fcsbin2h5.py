import h5py
import numpy as np
from .meas_to_count import file_to_fcs_count
from .get_fcs_info import get_file_info, get_metafile_from_file
from .atimes_data import ATimesData

def fcsbin2h5(binfile, h5file):
    data = file_to_fcs_count(binfile)

    h5f = h5py.File(h5file, 'w')
    h5f.create_dataset('dataset_1', data=data, compression="gzip")
    
    h5f.close()


def fcsbin2h5_atimes(binfile, h5file=None):
    photon_data = fcsbin2atimes(binfile)
    if h5file is None:
        h5file = binfile[:-4] + '.h5'

    with h5py.File(h5file, "w") as f:
        for ch in photon_data.all_channels:
            f.create_dataset(ch, data=getattr(photon_data, ch), compression="gzip")

        # save version
        inf = f.create_group('info')
        inf.attrs['tcspc'] = False
        inf.attrs['macrotime'] = photon_data.macrotime
        inf.attrs['microtime'] = photon_data.microtime


def fcsbin2atimes(binfile):
    data = file_to_fcs_count(binfile)
    mdata = get_file_info(get_metafile_from_file(binfile))
    macro_time = mdata.timeResolution * 1e-6 # s
    photon_data = numpy_timetrace_2_atimes(data, macro_time)
    return photon_data


def numpy_timetrace_2_atimes(data, time_resolution):
    """
    Convert a numpy array into an ATimes data object
    Parameters
    ----------
    data : numpy array

    time_resolution

    Returns
    -------

    """
    photon_data = ATimesData()
    photon_data.tcspc = False
    photon_data.macrotime = time_resolution  # s
    photon_data.microtime = time_resolution  # s

    for i in range(np.shape(data)[1]):
        setattr(photon_data, 'det' + str(i), photon_arrivals(data, i))

    return photon_data


def photon_arrivals(data, channel):
    weights = data[:, channel]
    times = np.flatnonzero(weights)

    return np.column_stack((
        times,
        np.ones(times.size, dtype=int),
        weights[times],
    ))