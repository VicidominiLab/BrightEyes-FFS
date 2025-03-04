import numpy as np
from ..tools.find_nearest import find_nearest


def sig(x, x0, a):
 return 1/(1 + np.exp(-a*(x-x0)))


def g2polar(g, Nr=180):
    """
    Function that converts x-correlations into a polar plot to indicate anisotropy
    
    Parameters
    ----------
    g : np.array()
        Array with 4 or 6 columns, each one the G values for right, up, left, down
        cross-correlations (only y values).
    Nr : int, optional
        Number of pixels in the polar plot. The default is 180.

    Returns
    -------
    z : np.array()
        2D array with the polar plot.

    """
    
    x = np.linspace(-1, 1, Nr)
    
    xx, yy = np.meshgrid(x, x)
    yy *= -1
    rr = np.sqrt(yy**2 + xx**2)
    theta = np.arctan(yy/xx)
    theta += 2*np.pi
    theta = theta % (np.pi)
    theta[yy < 0] += np.pi
    
    z = rr*0
    
    g_shape = np.shape(g)
    num_curves = g_shape[1]
    cmax = g_shape[0] - 1
    
    if num_curves == 4:
        angles = np.asarray([0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi])
    elif num_curves == 6:
        ['Right', 'UpRight', 'UpLeft', 'Left', 'DownLeft', 'DownRight']
        angles = np.asarray([0, np.pi/3, 2/3*np.pi, np.pi, 4/3*np.pi, 5/3*np.pi, 2*np.pi])
    else:
        return z
    
    for x in range(Nr):
        for y in range(Nr):
            r = np.clip(rr[y, x], 0, 1)
            th = theta[y, x]
            _, idx = find_nearest(angles, th)
            idx = np.mod(idx, len(angles)-1)
            color = g[np.clip(int(r*len(g[:,1])), 0, cmax), idx]
            z[y, x] = color
    
    z[rr > 1] = None
    return z

def g2flow(g, Nr=180, detector='square'):
    """
    Function that converts x-correlations into a polar plot to indicate flow
    
    Parameters
    ----------
    g : np.array()
        Array with 4 rows, each one the G values for right, up, left, down
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
        List with 2 values indicating the flow in tge [up, right] direction.

    """
    
    x = np.linspace(-1, 1, Nr)
    
    xx, yy = np.meshgrid(x, x)
    yy *= -1
    rr = np.sqrt(yy**2 + xx**2)
    theta = np.arctan(yy/xx)
    if detector == 'airy':
        # all angles shift with pi/4
        theta -= np.pi / 4
    theta += 2*np.pi
    theta = theta % (np.pi)
    theta[yy < 0] += np.pi
    
    z = rr*0
    
    G_diff = np.zeros((len(g[:, 0]), 2))
    G_diff[:, 0] = (g[:,1] - g[:,3]) / np.mean(g)
    G_diff[:, 1] = (g[:,0] - g[:,2]) / np.mean(g)
    cmax = len(G_diff[:,0]) - 1
    
    for x in range(Nr):
        for y in range(Nr):
            r = np.clip(rr[y, x], 0, 1)
            th = theta[y, x]
            color = G_diff[np.clip(int(r*len(G_diff[:,1])), 0, cmax), 1]*np.cos(th)
            color += G_diff[np.clip(int(r*len(G_diff[:,0])), 0, cmax), 0]*np.sin(th)
            z[y, x] = color
    
    #z = gaussian_filter(z, sigma=sigma)
    z[rr > 1] = None
    flow = np.sum(G_diff, 0)
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