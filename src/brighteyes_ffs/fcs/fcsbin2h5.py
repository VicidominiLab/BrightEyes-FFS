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
    photon_data_2_h5(photon_data, h5file)


def photon_data_2_h5(photon_data, h5file):
    with h5py.File(h5file, "w") as f:
        for ch in photon_data.all_channels:
            f.create_dataset(ch, data=getattr(photon_data, ch), compression="gzip")
        # save version
        inf = f.create_group('info')
        inf.attrs['tcspc'] = False
        inf.attrs['macrotime'] = photon_data.macrotime
        inf.attrs['microtime'] = photon_data.microtime
        inf.attrs['coords'] = photon_data.coords


def fcsbin2atimes(binfile, chunksize=-1):
    """
    Convert a .bin file with FCS data into an ATimes data object
    Parameters
    ----------
    binfile : path
        path to .bin file name
    chunksize : int, optional
        Choose -1 for a file that is not too large to read in memory
        Choose a number of points (e.g. 1000000) to read the data in chunks of this size
        The output is identical in both cases

    Returns
    -------
        ATimes data object
            Object with fields det0, det1, det2, det3, etc. with macrotimes, microtimes and weights
    """
    mdata = get_file_info(get_metafile_from_file(binfile))
    macro_time = mdata.timeResolution * 1e-6  # s

    if chunksize > 0:
        n_offset = 0
        photon_data_list = []
        while True:
            chunk = file_to_fcs_count(binfile, n_points=chunksize, n_offset=n_offset)
            if len(chunk) == 0:
                break
            photon_data_list.append(numpy_timetrace_2_atimes(chunk, macro_time, n_offset))
            n_offset += len(chunk)
        num_chunks = len(photon_data_list)
        for channel in photon_data_list[0].all_channels:
            photon_data = np.concatenate([getattr(photon_data_list[i], channel) for i in range(num_chunks)], axis=0, )
            setattr(photon_data_list[0], channel, photon_data)
        photon_data = photon_data_list[0]
    else:
        data = file_to_fcs_count(binfile)
        photon_data = numpy_timetrace_2_atimes(data, macro_time)

    photon_data.coords = mdata.coords
    return photon_data


def numpy_timetrace_2_atimes(data, time_resolution, arrival_time_offset=0):
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
        setattr(photon_data, 'det' + str(i), photon_arrivals(data, i, arrival_time_offset))

    return photon_data


def photon_arrivals(data, channel, arrival_time_offset=0):
    weights = data[:, channel]
    times = np.flatnonzero(weights)

    return np.column_stack((
        times + arrival_time_offset,
        np.ones(times.size, dtype=int),
        weights[times],
    ))