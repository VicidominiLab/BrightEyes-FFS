def checkfname(fname, extension):
    """
    Check if string ends with given extension

    Parameters
    ----------
    fname : string
        String with file name.
    extension : string
        Extension that fname should have.

    Returns
    -------
    fname : string
        String with the original fname and added extension if needed.

    """
    
    if fname is None:
        return None
    
    extL = len(extension)
    if len(fname) <= extL or fname[-extL:] != extension:
        fname = fname + "." + extension
    return fname
