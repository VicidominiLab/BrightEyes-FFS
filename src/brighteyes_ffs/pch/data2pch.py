# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 10:59:51 2025

@author: eslenders
"""

import numpy as np
from ..tools.bindata import bindata_chunks
from ..fcs.fcs2corr import fcs_load_and_corr_split


def pch_load_and_hist_split(fname, list_of_g=['central', 'sum3', 'sum5'], binsize=100, maxcount=30, split=10, time_trace=False, metadata=None, root=0, list_of_g_out=None):
    """
    Load an ffs data file, split in chunks and calculate pch for each chunk

    Parameters
    ----------
    fname : str
        File name with ffs data.
    list_of_g : list, optional
        List for which elements to calculate the pch, same concept as fcs.
        The default is ['central', 'sum3', 'sum5'].
    binsize : int, optional
        Bin the time trace before calculating the histogram. The default is 100.
    maxcount : int, optional
        The maximum photon count value in the histogram. The default is 30.
    split : float, optional
        Split the data in chunks of this duration and calculate the PCH for 
        each chunk separately. The default is 10.
    time_trace : boolean, optional
        Return the intenstiy time trace. The default is False.
    metadata : metadata object, optional
        Metadata object. If None, the metadata is loaded from the file.
        The default is None.
    root : int, optional
        Used for the GUI. The default is 0.
    list_of_g_out : list, optional
        List of strings with the names of the pch fields.
        E.g. Use ['ch12', 'sum3', 'sum5'] to rename the central element PCH to
        'ch12'. By default the same names as the input list_of_g are used.
        The default is None.

    Returns
    -------
    pch : Object
        Object with each field the PCH for 'central', 'sum3', etc. for each chunk

    """
    pch, data = fcs_load_and_corr_split(fname,
                                        list_of_g=list_of_g,
                                        accuracy=maxcount,
                                        split=split,
                                        binsize=binsize,
                                        time_trace=time_trace,
                                        metadata=metadata,
                                        root=root,
                                        list_of_g_out=list_of_g_out,
                                        averaging=None,
                                        algorithm='pch')
    if time_trace:
        return pch, data
    return pch

def calc_pch(data, binsize=1, maxcount=30, norm=True):
    """
    Calculate PCH from a numpy array

    Parameters
    ----------
    data : np.array()
        2D array (Nt x Nc), with Nt the number of time points and Nc the number
        of channels.
    binsize : int, optional
        Bin size. The default is 1.
    maxcount : int, optional
        The maximum photon count value in the histogram. The default is 30.
    norm : boolean, optional
        Normalize the histogram by its sum. The default is True.

    Returns
    -------
    PCH : np.array()
        2D array with photon counts and histogram values.

    """
    bdata = bindata_chunks(data, binsize)
    bins = np.arange(0, maxcount+1, 1)
    counts, bin_edges = np.histogram(bdata, bins=bins)
    pch = np.stack([bin_edges[0:-1], counts], axis=1)
    pch = pch.astype(float)
    if np.sum(counts) == 0:
        pch[:,1] = 0
    elif norm:
        pch[:,1] /= np.sum(pch[:,1])
    return pch.astype(float)