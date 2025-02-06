from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from copy import deepcopy

def color_gradient(startColor, endColor, N):
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(startColor[0], endColor[0], N)
    vals[:, 1] = np.linspace(startColor[1], endColor[1], N)
    vals[:, 2] = np.linspace(startColor[2], endColor[2], N)
    return(ListedColormap(vals))
    
    
def cmaps(color, N=256):
    switcher = {
        "blue2black": color_gradient((0, 0, 0, 1), mcolors.to_rgba('#1f77b4'), N),
        "orange2black": color_gradient((0, 0, 0, 1), mcolors.to_rgba('#ff7f0e'), N),
        "green2black": color_gradient((0, 0, 0, 1), mcolors.to_rgba('#2ca02c'), N),
        "red2black": color_gradient((0, 0, 0, 1), mcolors.to_rgba('#d62728'), N),
        "purple2black": color_gradient((0, 0, 0, 1), mcolors.to_rgba('#9467bd'), N),
    }
    # Get the function from switcher dictionary
    func = switcher.get(color, "Invalid color")
    # Execute the function
    return func


def change_color_from_map(cmap, new_color, idx=0):
    """
    Change a single color from a color map

    Parameters
    ----------
    cmap : string
        Color map (e.g. 'inferno').
    new_color : list
        list or np.array with 3 values between 0-1 (r,g,b) for the new color
    idx : int, optional
        Index of the color maps that has to be changed. The default is 0.

    Returns
    -------
    cmap : color map
        New color map with the color at idx changed to new_color.

    """
    
    if cmap == 'seaborn-colorblind':
        return cmap
    cmap = deepcopy(plt.get_cmap(cmap))
    vals = cmap.colors
    vals[idx] = new_color
    return(ListedColormap(vals))