from .constants import constants
import numpy as np


def stokes_einstein(D, T=293, visc=1e-3):
    """
    Calculate the diameter of particles based on the Stokes-Einstein equation

    Parameters
    ----------
    D : float
        Diffusion coefficient of the particles [in m^2/s].
    T : float, optional
        Temperature of the suspension [in K]. The default is 293.
    visc : float, optional
        Viscosity of the solvent [in Pa.s]. The default is 1e-3.

    Returns
    -------
    d : float
        Diameter of the particles [in m].

    """
    
    kb = constants('boltzmann')
    r = kb * T / 6 / np.pi / visc / D
    d = 2 * r
    return d
