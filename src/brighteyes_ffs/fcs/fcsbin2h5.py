import h5py
from .meas_to_count import file_to_fcs_count

def fcsbin2h5(binfile, h5file):
    data = file_to_fcs_count(binfile)

    h5f = h5py.File(h5file, 'w')
    h5f.create_dataset('dataset_1', data=data, compression="gzip")
    
    h5f.close()