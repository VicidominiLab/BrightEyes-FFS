import numpy as np

def file_to_analogcount(fname, dtype=">u8", print_info=False):
    """
    Read a bin file and returns an array with the decoded count for each measurement
    Args:
        fname: name of the file containing the data

    Returns:
        A numpy array of unsigned int16 os size N x 25 where N is the number of measurements 
    """
    try:
        raw = np.fromfile(fname, dtype=dtype)

    except:
        if print_info:
            print("Error reading binary file")
        return None

    elements = raw.shape[0]
    
    if print_info:
        print(f"Elements: {elements}")

    print(raw)

    out = np.zeros((elements , 2), dtype = 'int32')
    out[:,0] = np.bitwise_and(raw, int(2**32-1)) # 32 bits
    out[:,1] = np.bitwise_and(np.right_shift(raw, 32), int(2**32-1)) # 4 bits
    
    outplot = np.reshape(out, (1, 200, 200, 10, 2))
    
    outplot = np.squeeze(np.sum(outplot, 3))
    
    # plt.figure()
    # plt.imshow(outplot[:,:,1])
    # plt.colorbar()

    # plt.figure()
    # plt.plot(np.concatenate((outplot[100,:,0], outplot[101,:,0], outplot[102,:,0])))
    # plt.scatter(list(range(600)),np.concatenate((outplot[100,:,0], outplot[101,:,0], outplot[102,:,0])))
    # plt.plot(np.concatenate((outplot[100,:,1], outplot[101,:,1], outplot[102,:,1])))
    # plt.scatter(list(range(600)),np.concatenate((outplot[100,:,1], outplot[101,:,1], outplot[102,:,1])))


    return out, outplot