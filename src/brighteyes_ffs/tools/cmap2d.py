import matplotlib.pyplot as plt
import numpy as np

def cmap2d(lifetimeIm, intensityIm, ltimeBound=['auto', 'auto'], intBound=['auto', 'auto'], cmap='rainbow', plotFig=False, equalize_lum=True):
    """
    Make FLIM image by combining lifetimeImage and intensityImage

    Parameters
    ----------
    lifetimeIm : np.array()
        2D array with lifetime values.
    intensityIm : np.array()
        2D array with intensity values.
    ltimeBound : list, optional
        Lower and upper bound of lifetime values. The default is ['auto', 'auto'].
    intBound : list, optional
        Lower and upper bound of intensity values. The default is ['auto', 'auto'].
    cmap : string, optional
        color map. The default is 'rainbow'.

    Returns
    -------
    outputIm : np.array()
        FLIM image.

    """
    
    
    Ny = np.shape(lifetimeIm)[0]
    Nx = np.shape(lifetimeIm)[1]
    
    if intBound[0] == 'auto':
        intBound[0] = np.min(intensityIm)
    if intBound[1] == 'auto':
        intBound[1] = np.max(intensityIm)
    if ltimeBound[0] == 'auto':
        ltimeBound[0] = np.min(lifetimeIm)
    if ltimeBound[1] == 'auto':
        ltimeBound[1] = np.max(lifetimeIm)
    
    intensityIm = np.clip(intensityIm, intBound[0], intBound[1])
    intensityIm = (intensityIm - intBound[0]) / (intBound[1] - intBound[0])
    lifetimeIm = np.clip(lifetimeIm, ltimeBound[0], ltimeBound[1]) 
        
    outputIm = np.zeros((Ny, Nx, 3))
    
    
    if cmap != 'seaborn-colorblind':
        cmap = plt.get_cmap(cmap)
        #N = np.size(cmap.colors, 0)
        N = cmap.N
        cmapArray = np.zeros((N, 3))
        for i in range(N):
            cmapArray[i, :] = cmap(i)[0:3]
        # equalize luminescence
        if equalize_lum:
            lmax = np.max((299*cmapArray[:,0] + 587*cmapArray[:,1] + 114*cmapArray[:,2]) / 1000)
            for i in range(N):
                lum = (299*cmapArray[i,0] + 587*cmapArray[i,1] + 114*cmapArray[i,2]) / 1000
                lumfactor = np.min((lmax / lum, 1/np.max(cmapArray[i, :])))
                cmapArray[i, :] *= lumfactor
        
        # get index of color map for lifetime
        idx = (np.floor((lifetimeIm - ltimeBound[0]) / (ltimeBound[1] - ltimeBound[0]) * N)).astype(int)
        idx = np.clip(idx, 0, N-1)
        for k in range(3):
            outputIm[:,:,k] = np.take(cmapArray[:,k], idx) * intensityIm
        
        # get rgb lifetime colors and scale with intensity
        # for k in range(3):
        #     outputIm[:, :, k] = cmap(idx / N)[k] * intensityIm[i, j] for k in range(3)]
        # for i in range(Ny):
        #     for j in range(Nx):
        #         #outputIm[i, j, :] = [cmap.colors[idx[i, j]][k] * intensityIm[i, j] for k in range(3)]
        #         outputIm[i, j, :] = [cmap(idx[i, j] / N)[k] * intensityIm[i, j] for k in range(3)]
    
    if plotFig:
        plt.figure()
        plt.imshow(outputIm)
    
    return outputIm
    
