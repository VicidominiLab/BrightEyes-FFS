import numpy as np
from .get_fcs_info import get_file_info
from .meas_to_count import file_to_fcs_count


class ATimesData:
    pass


def fcs2atimes(fname, macrotime, split=10):
    """
    Load SPAD-fcs data in chunks (of 10 s) and convert to arrivaltimes vectors

    Parameters
    ----------
    fname : string
        File name with the .bin data.
    macrotime : float
        Macrotime multiplication factor / dwell time [s].
    split : float, optional
        Number of seconds of each chunk to split the data into
        E.g. split=10 will divide a 60 second stream in 6 ten-second
        traces and calculate G for each individual trace. The default is 10.

    Returns
    -------
    data : object
        Object with following fields:
        data.det0chunk0    arrival times chunk 0 detector 0
        ...
        data.det24chunk10  arrival times chunk 10 detector 24
        data.macrotime      macrotime
        data.duration       measurement duration [s]

    """
    
    info = get_file_info(fname[:-4] + "_info.txt")
    dwellTime = info.dwellTime
    duration = info.duration
    
    data = ATimesData()
    data.macrotime = dwellTime
    data.duration = duration
    
    N = np.int(np.floor(duration / split)) # number of chunks

    chunkSize = int(np.floor(split / dwellTime))
    for chunk in range(N):
        # ---------------- CALCULATE CORRELATIONS SINGLE CHUNK ----------------
        print("+-----------------------")
        print("| Loading chunk " + str(chunk))
        print("+-----------------------")
        dataChunk = file_to_fcs_count(fname, np.uint8, chunkSize, chunk*chunkSize)
        if np.max(dataChunk) > 1:
            print("Max too high.")
        else:
            for det in range(np.shape(dataChunk)[1]):
                setattr(data, "det" + str(det) + "chunk" + str(chunk), np.nonzero(dataChunk[:,det])[0])
    
    return data