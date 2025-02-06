import numpy as np
from skimage import measure
from imutils import contours
import cv2
import imutils
import matplotlib.pyplot as plt

def exppsf(image, region=20, plots=False, blur=11, threshold=[20, 255]):
    """
    Measure experimental PSF based on an image of beads.
    Multiple beads are automatically found, overlapped with each other and
    summed for each channel separately. The resulting array is returned.

    Parameters
    ----------
    image : np.array
        2D array with (x, y) image data or
        3D array with (x, y, c) image data.
    region : int, optional
        Length of the square (in px) around each bead. The default is 20 pixels.
    plots : boolean, optional
        Plot results. The default is False.
    blur : int, optional
        Amount of blurring that should be added first. The default is 11.
    threshold : list, optional
        List of two values for the lower and upper threshold. The default is [20, 255].

    Returns
    -------
    PSFsum : np.array
        3D array with for each channel the summed PSF.
    PSFs : np.array
        4D array (region, region, Nc, Npart) with the image of each
        particle separately.

    """
    
    # check dimensions
    imageShape = np.shape(image)
    Ny = imageShape[0]
    Nx = imageShape[1]
    if len(np.shape(image)) > 2:
        Nchannels = np.shape(image)[2]
        # sum over all channels
        imageSum = np.sum(image, 2)
    else:
        Nchannels = 1
        imageSum = image
    
    if plots:
        plt.figure()
        plt.imshow(imageSum)
    
    particles = find_particles(imageSum, blur=blur, threshold=threshold)
    
    if plots:
        plt.figure()
        plt.scatter(particles[:,0], particles[:,1])
    
    # sum over all PSFs for each channel
    Nparticles = len(particles)
    PSFs = np.zeros((region, region, Nchannels, Nparticles))
    r = int(np.ceil(region / 2))
    # check for boundary effects
    NpartReal = 0
    for i in range(Nparticles):
        y = int(particles[i, 0])
        x = int(particles[i, 1])
        if y > r and y < Ny-r and x > r and x < Nx-r:
            if Nchannels > 1:
                PSFs[:,:,:,NpartReal] = image[y-r:y+r,x-r:x+r,:]
            else:
                PSFs[:,:,:,NpartReal] = np.expand_dims(image[y-r:y+r,x-r:x+r], axis=2)
            NpartReal += 1
    PSFs = PSFs[:,:,:,0:NpartReal]
    PSFsum = np.sum(PSFs, 3)
    
    return PSFsum, PSFs
        

def find_particles(im, blur=3, threshold=[0, 255], nPx=10, returnAll=False):
    """
    Find particles in image

    Parameters
    ----------
    im : np.array()
        2D array with (x, y) image data.
    blur : int, optional
        Int, width (in px) of the Gaussian blur applied first. The default is 3.
    threshold : list, optional
        [min, max] values for thresholding. The default is [0, 255].
    nPx : int, optional
        Number of pixels. The default is 10.
    returnAll : boolean, optional
        Return 4 variables with all results. The default is False.

    Returns
    -------
    particles
        2D array with particle coordinates.

    """
    
    # blur image
    blurred = cv2.GaussianBlur(im, (blur, blur), 0)
    
    # change data type
    blurred = blurred / np.max(blurred) * 255
    blurred = blurred.astype(np.uint8)
    
    # apply threshold
    thresh = cv2.threshold(blurred, threshold[0], threshold[1], cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=1)
    
    # find peaks
    labels = measure.label(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
    	# if this is the background label, ignore it
    	if label == 0:
    		continue
    	# otherwise, construct the label mask and count the
    	# number of pixels 
    	labelMask = np.zeros(thresh.shape, dtype="uint8")
    	labelMask[labels == label] = 255
    	numPixels = cv2.countNonZero(labelMask)
    	# if the number of pixels in the component is sufficiently
    	# large, then add it to our mask of "large blobs"
    	if numPixels > nPx:
    		mask = cv2.add(mask, labelMask)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    Ncnts = len(cnts[0])
    print(str(Ncnts) + ' particles found')
    if Ncnts > 0:
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]
        particles = np.zeros((len(cnts), 2))
        # loop over the contours
        for (i, c) in enumerate(cnts):
        	# draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)
            [(cX, cY), radius] = cv2.minEnclosingCircle(c)
            particles[i, 0] = int(cY)
            particles[i, 1] = int(cX)
    else:
        particles = None
        
    if returnAll:
        return particles, Ncnts, blurred, thresh
    
    return particles