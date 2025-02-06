import h5py
import numpy as np
from datetime import datetime

class DataFileParam:
    def __init__(self):
        self.Creator = 'Name of the person/setup'
        self.DataSetDirectoryName = 'DataSet'
        self.Description = 'Measurement description, e.g. imaging test fluor beads'
        self.DocumentType = 'Image' # e.g. Image, fcs, SPT
        self.DichroicList = ['Chroma ZT405/488/561/640rpc']
        self.EmList = ['BP500-550']
        self.ExList = ['488']
        self.ExPowerList = ['10']
        self.ExPowerUnit = ['uW']
        self.ExFrequency = ['80']
        self.ExFrequencyUnit = ['MHz']
        self.Filename = 'File name'
        self.NameList = ['Channel name, e.g. cytoplasm']
        self.NumberOfDataSets = [1]
        self.NumericalAperture = 1.4
        self.Objective = '100x/1.4 Leica oil immersion'
        self.pxDwellTime = 10 # total time for one pixel (so over all time bins)
        self.pxDwellTimeUnit = 'us'
        self.pxSizeX = 1
        self.pxSizeY = 1
        self.pxSizeZ = 1
        self.SPADholdOff = 100 # hold-off time
        self.SPADholdOffUnit = 'ns'
        self.SPADmodel = 'BCD cooled'
        self.SPADtemperature = -15
        self.SPADtemperatureUnit = 'degr C'
        self.RecordingDate = [str(datetime.now())[0:23]] # list of recording times for each frame
        self.Unit = 'um'

def np2hdf5(filename, data, dataformat='Tzyxtc', param=DataFileParam()):
    """
    Convert data array to .hdf5 file
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    filename    Name of the .hdf5 file
    data        N dimensional np.array with the raw data
                Default 6 dim (T, z, y, x, t, c)
                If fewer dimensions, or different order, use dataformat
                parameter to describe what's inside
    dataformat  Order of the dimensions, e.g. if data is 2D xy image,
                dataformat will be 'xy'
    param       Object with the metadata parameters
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    .hdf5 file with always a 6 dimensional array for the data
    ===========================================================================
    """
    
    # check file name extension
    if len(filename) <= 4 or filename[-4:] != 'hdf5':
        filename = filename + ".hdf5"
    
    # not used at the moment
    if np.max(data) > 255:
        datatype = 'uint16'
    else:
        datatype = 'uint8'
    
    # make sure data has 6 dimensions
    dataShape = np.shape(data)
    Ndim = len(dataShape)
    for i in range(6-Ndim):
        data = np.expand_dims(data, Ndim+i)
    
    # check order of dimensions
    defOrder = 'Tzyxtc'
    order = []
    newdim = 0
    for i in range(6):
        dim = defOrder[i]
        if dim in dataformat:
            order.append(dataformat.find(dim))
        else:
            order.append(Ndim+newdim)
            newdim += 1
    data = np.transpose(data, order)
    
    # calculate finger print
    dataShape = np.shape(data)
    Nc = int(np.sqrt(dataShape[5]))
    fp = np.reshape(np.sum(np.sum(np.sum(np.sum(np.sum(data, 4), 3), 2), 1), 0), (Nc, Nc)) #fp = np.reshape(np.sum(data, axis=(4, 3, 2, 1, 0)), (Nc, Nc))
    
    # ---------------------------- Add data to file ---------------------------
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('RawData', data=data, compression="gzip")
    h5f.create_dataset('FingerPrint', data=fp, compression="gzip")
    
    # ------------------------------- Metadata --------------------------------
	# use same structure of mattia's metadata
    md = h5f.create_group("Metadata")
    fields = list(param.__dict__)
    for field in fields:
        md.attrs[field] = getattr(param, field)

    h5f.close()


def hdf5_read(filename, read='RawData', squeeze=False):
    """
    Read .hdf5 file with image/fcs/SPT data
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    filename    Name of the .hdf5 file
    read        Object to read
                    'RawData': read raw data
                    'FingerPrint': read finger print
    squeeze     Remove singleton dimenions
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    (6) dimensional numpy array or metadata
        hdf5_read('file.hdf5', read='RawData')
        hdf5_read('file.hdf5', read='pxDwellTime')
    ===========================================================================
    """
    with h5py.File(filename, "r") as f:
        if read in list(DataFileParam().__dict__):
            # return metadata
            return f['Metadata'].attrs[read]
        
        # Get the data
        data = f[read]
        
        if read == 'RawData':
            if squeeze:
                return np.squeeze(data[:,:,:,:,:,:])
            return data[:,:,:,:,:,:]
        
        elif read == 'FingerPrint':
            return data[:,:]
        
        return data[:]
    
    return None
