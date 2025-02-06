import os
from .fcs2corr import Correlations
from ..tools.list_files import list_files
from ..tools.csv2array import csv2array

def load_g(fname_root, folder_name="", print_fnames=True):
    """
    Load correlations from .csv files into data object

    Parameters
    ----------
    fname_root : string
        'Root' of the file names. E.g. if the .csv files are named
        'FCS_beads_central', 'FCS_beads_sum3', and 'FCS_beads_sum5',
        then 'FCS_beads_' is the common root.
    folderName : string, optional
        Name of the folder to look into. The default is "".
    print_fnames : boolean, optional
        Print the names of the files that were found. The default is True.

    Returns
    -------
    G : correlations object
        Data object with G.central, G.sum3, G.sum5 etc. the correlation
        functions.

    """
    
    G = Correlations()
    files = list_files(folder_name, "csv", fname_root)
    for file in files:
        setattr(G, strip_g_fname(file, fname_root, print_fnames), csv2array(file, ','))
    G.dwellTime = 1e6 * csv2array(file, ',')[1, 0] # in µs
    print('--------------------------')
    print(str(len(files)) + ' files found.')
    print('--------------------------')
    return G


def strip_g_fname(fname, fname_root, print_fnames=True):
    fname = os.path.basename(fname)
    dummy, file_extension = os.path.splitext(fname)
    index = fname.find(fname_root)
    fname = fname[index + len(fname_root):-len(file_extension)]
    if print_fnames:
        print(fname)
    return fname
    