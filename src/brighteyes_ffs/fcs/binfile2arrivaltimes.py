import numpy as np
import os
from .read_binfile import read_binfile
from .u64_to_counts import u64_to_counts
from ..tools.savevar import savevar


class ArrivalTimesObject:
    pass


def binfile2arrivaltimes(fname, store_pickle=False, dwellTime=[]):
    """
    Read a binary fcs data file, containing a list of U64 numbers that
    represent the photon counts per microtime bin of about 1 µs and store the
    arrival times in units of dwellTime.
    Storing arrivaltimes in general reduces the file size significantly

    Parameters
    ----------
    fname : string
        Binary file name.
    store_pickle : Boolean, optional
        Store result in Pickle file. The default is False.
    dwellTime : float, optional
        Pixel dwell time [s]. The default is [].

    Returns
    -------
    arrival_times : list
        26 lists with the arrival times of the photon for each of
        the 25 detector elements + the sum.

    """
    
    readLength = 8 * 65536

    # Get file size [in bytes]
    fsize = os.path.getsize(fname)

    # Empty object to store photon arrival times
    arrival_times = ArrivalTimesObject()
    for i in range(25):
        setattr(arrival_times, 'det' + str(i), [])
    arrival_times.sum = []
    arrival_times.dwellTime = dwellTime

    startPos = 0  # position in the .bin file to start reading
    t = 0  # iterator used to indicate current time bin

    # Go through the file, read a chunk of data, write to the object, and repeat
    while startPos < fsize:
        # Read part of the binary file
        data = read_binfile(fname, startPos, readLength)
        startPos += readLength

        # Calculate photon counts
        for i in range(len(data)):
            counts = u64_to_counts(data[i])
            # go through each detector element
            for det in range(25):
                for c in range(counts[det]):
                    getattr(arrival_times,
                            'det' + str(det)).append(int(np.floor(t/2)))
            # sum of all detector elements
            for c in range(counts[25]):
                arrival_times.sum.append(int(np.floor(t/2)))
            t += 1

    
    if store_pickle:
        print("Storing .pickle file")
        savevar(arrival_times, fname[0:-4] + "_arrival_times.pickle")
        print(".pickle file stored")

    return arrival_times
