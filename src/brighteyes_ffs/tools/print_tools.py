# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:18:22 2025

@author: eslenders
"""

def print_table(table, width=8):
    """
    Print 2D numeric table in a nice tabular way

    Parameters
    ----------
    table : np.array()
        2D array.
    width : int, optional
        Width of each cell. The default is 8.

    Returns
    -------
    None.

    """
    for row in table:
        print("".join(f"{v:{width}.2f}" for v in row))
