import numpy as np
from ..tools.find_nearest import find_nearest


def sig(x, x0, a):
 return 1/(1 + np.exp(-a*(x-x0)))


def g2polar(g, Nr=512):
    """
    Convert directional cross-correlation curves into a polar heatmap.

    Parameters
    ----------
    g : (n_lags, n_dirs) array_like
        Cross-correlation values. Columns correspond to directions.
        Supported n_dirs: 4 (R, U, L, D) or 6 (R, UR, UL, L, DL, DR).
        Rows correspond to lag/time (mapped radially).
    Nr : int
        Output image size in pixels (Nr x Nr).

    Returns
    -------
    z : (Nr, Nr) ndarray
        Polar heatmap (NaN outside the unit circle).
    """
    g = np.asarray(g)
    if g.ndim != 2:
        raise ValueError("g must be a 2D array with shape (n_lags, n_dirs).")

    n_lags, n_dirs = g.shape

    if n_dirs == 4:
        # R, U, L, D (angles in radians)
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    elif n_dirs == 6:
        # R, UR, UL, L, DL, DR
        angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    else:
        # Return empty map for unsupported number of directions
        return np.full((Nr, Nr), np.nan, dtype=float)

    # Grid in normalized coordinates [-1, 1]
    grid = np.linspace(-1.0, 1.0, Nr)
    xx, yy = np.meshgrid(grid, grid)
    yy = -yy  # keep your original convention (image y-axis downward)

    rr = np.sqrt(xx**2 + yy**2)                 # radius in [0, sqrt(2)]
    theta = (np.arctan2(yy, xx) + 2*np.pi) % (2*np.pi)  # angle in [0, 2pi)

    # Map radius -> lag index
    r_clipped = np.clip(rr, 0.0, 1.0)
    ri = np.floor(r_clipped * (n_lags - 1)).astype(np.int64)  # (Nr, Nr)

    # Map angle -> nearest direction index (circular distance!)
    # Compute wrapped angular difference to each direction
    d = np.angle(np.exp(1j * (theta[..., None] - angles[None, None, :])))  # (-pi, pi]
    ai = np.abs(d).argmin(axis=-1).astype(np.int64)  # (Nr, Nr)

    # Lookup
    z = g[ri, ai].astype(float)

    # Outside unit circle -> NaN
    z[rr > 1.0] = np.nan
    return z


def g2flow(g, Nr=512, detector='square'):
    """
    Function that converts x-correlations into a polar plot to indicate
    diffusion anisotropy such as flow
    
    Parameters
    ----------
    g : np.array()
        Array with 4 columns, each one the G values for right, up, left, down
        cross-correlations (only y values).
    Nr : int, optional
        Number of pixels in the polar plot. The default is 180.
    detector : str, optional
        Detector type used, either 'square' or 'airy' for the Zeiss airyscan

    Returns
    -------
    z : np.array()
        2D array with the flow plot.
    flow : list
        List with 2 values indicating the flow in the [up, right] direction.

    """
    
    g = np.asarray(g)
    if g.ndim != 2:
        raise ValueError("g must be a 2D array (n_lags, n_dirs).")

    # Grid
    x = np.linspace(-1.0, 1.0, Nr)
    xx, yy = np.meshgrid(x, x)
    yy = -yy

    rr = np.sqrt(xx**2 + yy**2)

    # Angles: robust quadrant handling
    theta = (np.arctan2(yy, xx) + 2*np.pi) % (2*np.pi)

    if detector == "airy":
        theta = (theta - np.pi/4) % (2*np.pi)

    # Map radius -> lag index
    n_lags = g.shape[0]
    r_clipped = np.clip(rr, 0.0, 1.0)
    ri = np.floor(r_clipped * (n_lags - 1)).astype(np.int64)  # (Nr, Nr)

    # Output
    z = np.zeros((Nr, Nr), dtype=float)

    if detector == "airy6":
        # Build 6 directional differences (same as your code)
        G_diff = np.zeros((n_lags, 6), dtype=float)
        gmean = np.mean(g)

        G_diff[:, 0] = (g[:, 1] - g[:, 4]) / gmean  # top right
        G_diff[:, 1] = (g[:, 0] - g[:, 3]) / gmean  # top
        G_diff[:, 2] = (g[:, 5] - g[:, 2]) / gmean  # top left
        G_diff[:, 3] = -G_diff[:, 0]               # bottom left
        G_diff[:, 4] = -G_diff[:, 1]               # bottom
        G_diff[:, 5] = -G_diff[:, 2]               # bottom right

        # Bin angle into 6 sectors of width pi/3
        th_idx = np.floor(theta / (np.pi/3)).astype(np.int64)
        th_idx = np.clip(th_idx, 0, 5)

        # Lookup
        z = G_diff[ri, th_idx]

        # Flow vector (your formula)
        Gsum = np.sum(G_diff, axis=0)
        flow = [Gsum[1] + 0.5*Gsum[0] - 0.5*Gsum[5],
                0.87*Gsum[0] + 0.87*Gsum[5]]
        flow = [0.3 * i for i in flow]

    else:
        # 2-component “flow field” projected onto angle
        G_diff = np.zeros((n_lags, 2), dtype=float)
        gmean = np.mean(g)

        # keep your original mapping
        G_diff[:, 0] = (g[:, 1] - g[:, 3]) / gmean  # vertical (U-D)
        G_diff[:, 1] = (g[:, 0] - g[:, 2]) / gmean  # horizontal (R-L)

        # Lookup the radial profiles once
        gd0 = G_diff[ri, 0]  # (Nr, Nr)
        gd1 = G_diff[ri, 1]  # (Nr, Nr)

        z = gd1 * np.cos(theta) + gd0 * np.sin(theta)

        flow = np.sum(G_diff, axis=0)

    z[rr > 1.0] = np.nan
    return z, flow


def g2polar_old(g, smoothing=3, norm=None):
    """
    Convert fcs correlation curves to polar flow heatmap
    
    Parameters
    ----------
    g : 1D np.array()
        Array with G values.
    smoothing : int
        Number indicating moving average window for smoothing

    Returns
    -------
    z : 1D np.array()
        Array with z values.
    
    """
    # smooth function
    Gsmooth = np.convolve(g, np.ones(smoothing)/smoothing, mode='valid')
    
    # calculate difference
    Gsmooth /= Gsmooth[0]
    Gdiff = Gsmooth - Gsmooth[0]
    Gout = Gdiff[1:] - Gdiff[0:-1]
    
    # smooth
    Gout = np.convolve(Gout, np.ones(10)/10, mode='valid')
    
    Gout -= np.min(Gout)
    Gout2 = []
    Gout2.append(Gout[0])
    offset = 0
    for i in range(len(Gout)-1):
        
        if Gout[i+1] > Gout[i]:
            offset += Gout[i+1] - Gout[i]
            Gout2.append(Gout[i+1]-offset)
        else:
            Gout2.append(Gout[i+1]-offset)
        
    return Gout2
    
    
    # set
    mask = np.clip(Gout[1:] - Gout[0:-1], None, 0)
    mask[mask < 0] = 1
    
    diff = np.diff(Gout)  # Calculate differences between consecutive elements
    mask2 = diff >= 0  
    Gout[1:][mask2] = Gout[:-1][mask2]
    
    
    
    
    Gout[1:] = Gout[1:]*mask
    
    Gout = np.convolve(Gout, np.ones(smoothing)/smoothing, mode='valid')
    
    return Gout
    
    # now we have the radius as a function of the color --> invert this
    rad = np.linspace(0, 1, 255)
    z = np.zeros(len(rad))
    
    for ri, r in enumerate(rad):
        z[ri] = Gdiff[np.clip(int(r*len(Gdiff)), 0, len(Gdiff)-1)]
    
    z += 1
    #z *= 1 - sig(rad,0.5,1)
    return np.cumsum(z - (1 - sig(rad,0.5,10))) * (1-rad)
        
    
    # calculate derivative
    Gdiff = np.clip(Gsmooth[1:] - Gsmooth[0:-1], a_min=None, a_max=0)
    Gdiff *= -1
    
    # normalized cumulative sum
    #Gdcum = np.cumsum(Gdiff)
    #Gdcum /= np.max(Gdcum)
    Gdcum = np.cumsum(Gsmooth)
    
    if norm is None:
        Gdcum /= np.max(Gdcum)
    else:
        Gdcum /= norm
    
    # now we have the radius as a function of the color --> invert this
    rad = np.linspace(0, 1, 255)
    z = np.zeros(len(rad))
    
    for ri, r in enumerate(rad):
        [dummy, idx] = find_nearest(Gdcum, r)
        z[ri] = 1 - idx / len(Gdcum)
    
    # apply sigmoid function to be more sensitive for changes near the center
    #z = sig(z, len(Gdcum)/1.5, 25/len(Gdcum))
    
    z = np.convolve(z, np.ones(7)/7, mode='valid')
    
    return z