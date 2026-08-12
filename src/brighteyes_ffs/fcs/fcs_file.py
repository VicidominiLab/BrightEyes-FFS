# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:00:11 2026

@author: eslenders
"""

from .atimes_data import load_atimes_data
from .meas_to_count import file_to_fcs_count

def load_file(fname, n_points=-1):
    # TCSPC:  .ptu, .h5, .hdf5
    # binned: .bin, .h5, .mat, .czi, .tiff, else
    
    if any([fname.endswith(".ptu"), fname.endswith(".h5"), fname.endswith(".hdf5"), fname.endswith(".t3r")]):
        try:
            data = load_atimes_data(fname, channels='auto', sysclk_MHz=240, perform_calib=False)
            return data
        except:
            pass
    
    if any([fname.endswith(".bin"), fname.endswith(".h5"), fname.endswith(".mat"), fname.endswith(".czi"), fname.endswith(".tiff")]):
        try:
            data = file_to_fcs_count(fname, n_points=-1)
            return data
        except:
            pass
        
    
        
        