from os.path import split

def path2fname(path, properWay=False):
    """
    Split path into file name and folder

    Parameters
    ----------
    path : path
        Path to a file [string], e.g. 'C:\\Users\\SPAD-fcs\\file.ext'.
    properWay : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fname : str
        File name, e.g. 'file.ext'.
    folderName : path
        Folder name, e.g. 'C:\\Users\\SPAD-fcs\\'.

    """
    
    if properWay:
        output = split(path)
        folderName = output[0]
        fname = output[1]
    else:
        path = path.replace("/", "\\")
        fname = path.split('\\')[-1]
        folderName = '\\'.join(path.split('\\')[0:-1]) + '\\'
    
    return fname, folderName
