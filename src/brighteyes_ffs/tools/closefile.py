import pandas


def closefile(fname):
    """
    Close .h5 file

    Parameters
    ----------
    fname : string
        Path to and name of the file.

    Returns
    -------
    random string.

    """
    
    if fname[-3:] == ".h5":
        # h5 file
        fileToClose = pandas.HDFStore(fname)
        fileToClose.close()
        fileToClose = "random string"
    
    return(fileToClose)

