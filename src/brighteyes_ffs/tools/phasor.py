import numpy as np
import matplotlib.pyplot as plt


def phasor(H, normalize=True):
    """
    Caculate phasor values from given histogram
    g = 1 / N * sum(H * cos(f))
    s = 1 / N * sum(H * sin(f))

    Parameters
    ----------
    H : List or np.array
        List or np.array with histogram values.
    normalize : boolean, optional
        Normalize histogram. The default is True.

    Returns
    -------
    complex number describing the phasor.

    """
    
    Np = len(H)
    F = 2 * np.pi * np.linspace(0, Np-1, Np) / (Np - 1)
    
    norm = 1
    if normalize:
        norm = np.sum(H)
    
    g = np.sum(H * np.cos(F)) / norm
    s = np.sum(H * np.sin(F)) / norm
    
    z = complex(g, s)
    
    return(z)


def plotphasor(z):
    
    plt.figure()
    
    A = 1
    phi = np.arange(0, 2*np.pi, 2*np.pi/360)
    x1 = A * np.cos(phi)
    y1 = A * np.sin(phi)
    
    A = 0.5
    phi = np.arange(0, 2*np.pi, 2*np.pi/360)
    x2 = A * np.cos(phi) + 0.5
    y2 = A * np.sin(phi)
    
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.scatter(z.real, z.imag)
    plt.plot([-1, 1], [0, 0], color='k')
    plt.plot([0, 0], [-1, 1], color='k')
    
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.gca().set_aspect('equal', adjustable='box')    
    