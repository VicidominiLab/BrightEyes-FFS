# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 14:02:39 2025

@author: eslenders
"""

import numpy as np

def generate_3d_gaussian(shape, w_xy, w_z, px_xy=1.0, px_z=1.0):
    """
    Generate a 3D Gaussian centered in the box.

    Parameters
    ----------
    shape : tuple of 3 ints
        The shape of the output array (nz, ny, nx).
    w_xy : float
        Lateral (xy) beam waist in physical units (nm).
    w_z : float
        Axial (z) beam waist in physical units.
    px_xy : float
        Pixel size in xy-plane.
    px_z : float
        Pixel size in z-direction.

    Returns
    -------
    g : 3D numpy array
        The Gaussian function sampled on the specified grid.
    """

    nz, ny, nx = shape
    z = (np.arange(nz) - nz // 2) * px_z
    y = (np.arange(ny) - ny // 2) * px_xy
    x = (np.arange(nx) - nx // 2) * px_xy

    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

    gaussian = np.exp(-2 * (xx**2 + yy**2) / w_xy**2 - 2 * (zz**2) / w_z**2)
    return gaussian / np.sum(gaussian)  # normalize