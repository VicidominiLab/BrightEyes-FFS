import os

def list_files(directory='C:/Users/SPAD-fcs/OneDrive - Fondazione Istituto Italiano Tecnologia', filetype='bin', substr=False):
    """
    Find all files with file extension 'filetype' in the given directory.
    Subfolders included.

    Parameters
    ----------
    directory : path, optional
        Directory path string, e.g. 'C:\\Users\\SPAD-fcs'
    filetype : str, optional
        File extension. The default is 'bin'.
    substr : str or Boolean False, optional
        Only files that contain "substr" are returned
        Use False to not filter on substr. The default is False.

    Returns
    -------
    files : list
        List of file names.

    """
    
    # use current directory is empty string is given
    if directory == "":
        directory = os.getcwd()
    
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(directory):
        for file in f:
            if '.' + filetype in file and (substr == False or substr in file):
                files.append(os.path.join(r, file))
        
    return files
