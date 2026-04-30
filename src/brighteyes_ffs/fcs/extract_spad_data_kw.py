# -*- coding: utf-8 -*-

"""
Get channel numbers from predefined keywords
E.g. "central" is a predefined keyword for the central channel number
of the 5x5 GI SPAD array detector, i.e. channel 12

Keywords MUST NOT start with capital C, as this is reserved for custom sums,
see extract_spad_photon_streams.py
"""

keyword_2_ch = {
    # SPAD 5x5
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
    
    # PRISM 7x7 Genoa Instruments
      "prismcentral" : [24],
      "prismsum3" : [16,17,18,23,24,25,30,31,32],
      "prismsum5" : [i for i in range(49) if i not in [0,1,2,3,4,5,6,7,13,14,20,21,27,28,34,35,41,42,43,44,45,46,47,48]],
      "prismsum7" : [i for i in range(49)],
    
    }