# -*- coding: utf-8 -*-

"""
Get channel numbers from predefined keywords
E.g. "central" is a predefined keyword for the central channel number
of the 5x5 GI SPAD array detector, i.e. channel 12

Keywords MUST NOT start with capital C, as this is reserved for custom sums,
see extract_spad_photon_streams.py
"""

keyword_2_ch = {
    # SPAD Genoa Instruments
      "central": [12],
      "sum3": [6, 7, 8, 11, 12, 13, 16, 17, 18],
      "sum5": [i for i in range(25)],
      "chess0": list(range(0, 25, 2)),
      "chess1": list(range(1, 25, 2)),
      
    # PI23
      "picentral": [20],
      "piring1": [15, 16, 19, 20, 21, 24, 25],
      "piring2": [i+9 for i in range(23) if i not in [0, 4, 18, 22]],
      "piring3": [i+9 for i in range(23)],
      
    # airyscan
      "airycentral" : [0],
      "airyring1" : [0, 1, 2, 3, 4, 5, 6],
      "airyring2" : [i for i in range(19)],
      "airyring3" : [i for i in range(32)],
    
    }