# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:46:35 2019

@author: SPAD-fcs
"""

class Correlations:
    pass

def select_g(g, selection='average'):
    """
    Return a selection of the autocorrelations

    Parameters
    ----------
    g : TYPE
        Object with all autocorrelations, i.e. output of e.g.
        fcs2corrsplit.
    selection : TYPE, optional
        Default value 'average': select only the autocorrelations that 
        are averaged over multiple time traces.
        E.g. if fcs2corrsplit splits a time trace in 10 pieces,
        calculates G for each trace and then calculates the average G, 
        all autocorrelations are stored in G. This function removes all
        of them except for the average G. The default is 'average'.

    Returns
    -------
    Autocorrelation object with only the pixel dwell time and the 
    average autocorrelations stored. All other autocorrelations are
    removed.

    """
    
    # get all attributes of G
    Glist = list(g.__dict__.keys())
    
    if selection == 'average':
        # make a new list containing only 'average' attributes
        Glist2 = [s for s in Glist if "average" in s]
    else:
        Glist2 = Glist
    
    # make a new object with the average attributes
    Gout = Correlations()
    for i in Glist2:
        setattr(Gout, i, getattr(g, i))
        
    # add dwell time
    Gout.dwellTime = g.dwellTime
    
    return(Gout)
