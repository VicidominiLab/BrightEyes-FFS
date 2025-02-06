import numpy as np

def timetrace_end(tt):
    """
    Find index of last data plot of time trace to plot.
    Because of rounding errors in binning the data, the last couple of data
    points may be zeros. These should not be included in the plot
    ===========================================================================
    Input           Meaning
    ---------------------------------------------------------------------------
    tt              np.array((N x 25)) with time trace for 25 pixels
    ===========================================================================
    Output
    ---------------------------------------------------------------------------
    idx             inde of last nonzero element
                    (only 50 last points are checked)
    ===========================================================================
    """
    ttsum = np.sum(tt, 1)
    L = len(ttsum) - 1 # last index
    i = 0
    lastFound = False
    while i < 50 and lastFound == False:
        idx = L - i
        if ttsum[idx] == 0:
            i += 1
        else:
            lastFound = True
    return idx + 1