import numpy as np
import os
import argparse
import h5py
import sys
import tifffile
from ome_types import from_xml
from ..tools.list_files import list_files
from ..tools.array2tiff import array2tiff, array2rgbtiff
from .get_fcs_info import get_file_info, get_metafile_from_file, get_file_info_czi
import czifile
#import libttp

"""
This set of functions allows reading a binary file containing SPAD measurements
using only the file name. The parameters are extracted from the matrix using
the tags. The assumption is that the parameters are constant and that all the
frames are complete.
Author: Sebastian Acuna and Eli Slenders
"""

def file_to_count(fname, datatype=np.uint16, print_info=False, n_points=-1, n_offset=0):
    """
    Read a bin file and returns an array with the decoded count for each measurement
    Args:
        fname: name of the file containing the data

    Returns:
        A numpy array of unsigned int16 os size N x 25 where N is the number of measurements 
    """
    try:
        if n_points != -1:
            n_points = n_points * 2
            NbytesOffset = 16 * n_offset
            raw = np.fromfile(fname, dtype=">u8", count=n_points, offset=NbytesOffset)
        else:
            raw = np.fromfile(fname, dtype=">u8")

    except:
        if print_info:
            print("Error reading binary file")
        return None

    elements = raw.shape[0]
    positions = int(elements/2)
    raw_pos = np.reshape(raw, (positions, 2))
    if print_info:
        print(f"Elements: {elements}")
        print(f"Positions: {positions}")
        print(f"data table: {raw_pos.shape}")

    time_per_pixel_tag = np.bitwise_and(raw_pos[:,1], 0b1)
    idx = np.argmax(time_per_pixel_tag != time_per_pixel_tag[0]) # positions per time
    time_per_pixel = int(idx)
    if print_info:
        print(f"time per pixel: {time_per_pixel}")

    frame_tag = np.bitwise_and(np.right_shift(raw_pos[:,1], 2), 0b1)
    idx = np.argmax(frame_tag != frame_tag[0]) # positions per frame
    if idx == 0:
        if print_info:
            print("Unique frame")
        frames = 1
    else:
        frames = int(positions/idx) # TODO: check condition with larger dataset
        
    line_tag = np.bitwise_and(np.right_shift(raw_pos[:,1], 1), 0b1)
    idx = int(np.argmax(line_tag != line_tag[0])/time_per_pixel) # positions  per line
    if print_info:
        print(f"Positions per lines: {idx}")
    x = int(idx)
    y = int(positions/x/time_per_pixel/frames)

    if print_info:
        print(f"Dimensions: Y:{y}, X:{x}")

    out = np.zeros((positions , 25), dtype = datatype)

    matrix_to_count(raw_pos, out)

    return out, frames, y, x, time_per_pixel


def file_to_fcs_count(fname, datatype=np.uint16, n_points=-1, n_offset=0, h5dataset=None):
    """
    Read a *.bin, *.tiff, *.h5 file and returns an array with the decoded count
    Args:
        fname: name of the file containing the data
    Returns:
        A numpy array of unsigned int16 os size N x 25 where N is the number of measurements 
    """
    if fname[-3:] == '.h5':
        with h5py.File(fname, "r") as f:
            # Get the data
            if h5dataset is not None:
                data = f[h5dataset]
            else:
                key = None
                keys = list(f.keys())
                Nkeys = len(keys)
                if Nkeys > 1:
                    for i in range(Nkeys):
                        if 'data' in keys[i] and 'meta' not in keys[i]:
                            key = keys[i]
                if key is None:
                    key = keys[0]
                if 'data' in keys:
                    key = 'data'
                data = f[key]
                if len(np.shape(data)) > 2:
                    # flatten data to (t, c)
                    data = data[:]
                    data = data.reshape(-1, data.shape[-1])
                print(np.shape(data))
            if n_points == -1:
                out = data[n_offset:]
            else:
                out = data[n_offset:n_offset+n_points]
    elif fname.endswith('.tif') or fname.endswith('.tiff'):
        if n_points == -1:
            data = np.squeeze(tifffile.imread(fname))
            if len(data.shape) < 2:
                print("Image does not have 2 dimensions.")
                return None
            slices = [slice(n_offset, None)] + [slice(0, None)] + [0] * (len(data.shape) - 2)
            out = data[tuple(slices)]
        else:
            with tifffile.TiffFile(fname) as tif:
                ome_metadata_xml = tif.ome_metadata
                if ome_metadata_xml is None:
                    im = tifffile.imread(fname)
                    data_shape = np.shape(im)
                else:
                    ome_metadata = from_xml(ome_metadata_xml)
                    pixels = ome_metadata.images[0].pixels
                    px = pixels.size_x if pixels.size_x and pixels.size_x > 1 else None
                    py = pixels.size_y if pixels.size_y and pixels.size_y > 1 else None
                    pz = pixels.size_z if pixels.size_z and pixels.size_z > 1 else None
                    pc = pixels.size_c if pixels.size_c and pixels.size_c > 1 else None
                    pt = pixels.size_t if pixels.size_t and pixels.size_t > 1 else None
                    data_shape = (pt, pc, pz, py, px)
                    data_shape = tuple(x for x in data_shape if x is not None)
                if len(data_shape) < 2:
                    print("Image does not have 2 dimensions.")
                    return None
                slices = [slice(n_offset, n_offset + n_points)] + [slice(0, data_shape[1])] + [0] * (len(data_shape) - 2)
                out = tif.asarray()[tuple(slices)]
    
    elif fname.endswith('.czi'):
        # check if corresponding h5 exists, if not create it
        fname_h5 = fname[:-4] + '.h5'
        if not os.path.exists(fname_h5):
            print('converting czi file')
            _ = czi2h5(fname)
        with h5py.File(fname_h5, "r") as f:
            # Get the data
            data = f['data']
            print(np.shape(data))
            if n_points == -1:
                out = data[n_offset:]
            else:
                out = data[n_offset:n_offset+n_points]
            out = out[:]
    else:
        try:
            n_points = n_points * 2
            NbytesOffset = 16 * n_offset
            raw = np.fromfile(fname, dtype=">u8", count=n_points, offset=NbytesOffset)
    
        except:
            print("Error reading binary file")
            return None
    
        elements = raw.shape[0]
        print(f"Elements: {elements}")
        positions = int(elements/2)
        print(f"Positions: {positions}")
    
        print("Freeing memory")
        out = np.zeros((positions , 25), dtype = datatype)
        print("Done.")
        
        raw_pos = np.reshape(raw, (positions, 2))
        print(f"data table: {raw_pos.shape}")
    
        print("Converting data to counts")
        matrix_to_count(raw_pos, out)
        print("Done.")

    return out


def czi2h5(fname):
    """
    Convert a Zeiss czi file to h5. The h5 file is stored in the same location 

    Parameters
    ----------
    fname : str/path
        Path to the czi file

    Returns
    -------
    fname_h5 : string
        Path of the .h5 file.

    """
    out = np.squeeze(czifile.imread(fname))
    mdata = get_file_info_czi(fname)
    while len(np.shape(out)) > 3:
        out = out[:,0]
    out = np.transpose(out.reshape((32,out.shape[1]*out.shape[2])))
    fname_h5 = fname[:-4] + ".h5"
    fname_h5 = numpy2h5(fname_h5, out, mdata.timeResolution)
    return fname_h5


def numpy2h5(fname, data, time_resolution):
    """
    Convert 2D numpy array(Nt, Nc) to h5 file
    with Nt the number of time points and Nc the number of channels

    Parameters
    ----------
    fname : path
        Path to .h5 file.
    data : np.array()
        2D array with data.
    time_resolution : float
        Dwell time in microseconds

    Returns
    -------
    fname : path
        Path to .h5 file.

    """
    with h5py.File(fname, "w") as f:
        inf = f.create_group('configurationGUI_beforeStart')
        inf.attrs['nrep'] = 1
        inf.attrs['nframe'] = 1
        inf.attrs['ny'] = 1
        inf.attrs['nx'] = 1
        inf.attrs['range_z'] = 0
        inf.attrs['range_y'] = 0
        inf.attrs['range_x'] = 0
        inf.attrs['timebin_per_pixel'] = len(data)
        inf.attrs['time_resolution'] = time_resolution
        f.create_dataset('data',data=data, compression="gzip")
    return fname


def bin2np(fname, datatype=np.uint16, Nchunks=-1, metadata=None, sum_tbins=True, read_chunks='all'):
    # r, y, x, t, c
    if Nchunks == -1:
        out, frames, y, x, time_per_pixel = file_to_count(fname, datatype)
        im = reshape_to_5d(out, frames, y, x, time_per_pixel)
    else:
        if metadata is None:
            metafile = get_metafile_from_file(fname)
            metadata = get_file_info(metafile)
        dwellTime = 1e-6 * metadata.timeResolution # s
        duration = metadata.duration
        split = duration / Nchunks
        chunkSize = int(np.round(split / dwellTime))
        if read_chunks == 'all':
            read_chunks = Nchunks
        for chunk in range(read_chunks):
            print('loading chunk ' + str(chunk) + '/' + str(Nchunks-1))
            [out, frames, y, x, time_per_pixel] = file_to_count(fname, datatype=datatype, n_points=chunkSize, n_offset=chunk*chunkSize)
            data = reshape_to_5d(out, frames, y, x, time_per_pixel)
            if sum_tbins:
                data = np.sum(data, 3) # sum over time bins --> f, y, x, c
            else:
                data = data[:,:,:,-1,:] # take only last time bin --> f, y, x, c (keeping the original structure requires too much memory)
            if chunk == 0:
                im = np.zeros((int(Nchunks*frames), y, x, 25))
            im[chunk*frames:(chunk+1)*frames, :, :, :] = data
            
    return im


def matrix_to_count(values, out):
    """
    Read an array of N measurements and write the count values in the out
    array

    Args:
        values: N x 2 unsigned int array with measurements
        out:    N x 25 unsigned int array for storing results

    Returns:
        The matrix out filled with the count
    """

    out[:,0] = np.bitwise_and(np.right_shift(values[:,0], 64 - 59), 0b1111) # 4 bits
    out[:,1] = np.bitwise_and(np.right_shift(values[:,0], 64 - 55), 0b1111) # 4 bits
    out[:,2] = np.bitwise_and(np.right_shift(values[:,0], 64 - 51), 0b1111) # 4 bits
    out[:,3] = np.bitwise_and(np.right_shift(values[:,0], 64 - 47), 0b1111) # 4 bits
    out[:,4] = np.bitwise_and(np.right_shift(values[:,0], 64 - 43), 0b1111) # 4 bits
    out[:,5] = np.bitwise_and(np.right_shift(values[:,0], 64 - 39), 0b1111) # 4 bits
    
    out[:,6] = np.bitwise_and(np.right_shift(values[:,1], 64 - 59), 0b11111) # 5 bits
    out[:,7] = np.bitwise_and(np.right_shift(values[:,1], 64 - 54), 0b111111) # 6 bits
    out[:,8] = np.bitwise_and(np.right_shift(values[:,1], 64 - 48), 0b11111) # 5 bits
    out[:,9] = np.bitwise_and(np.right_shift(values[:,1], 64 - 43), 0b1111) # 4 bits
    out[:,10] = np.bitwise_and(np.right_shift(values[:,1], 64 - 39), 0b1111) # 4 bits
    out[:,11] = np.bitwise_and(np.right_shift(values[:,1], 64 - 35), 0b111111) # 6 bits
    out[:,12] = np.bitwise_and(np.right_shift(values[:,1], 64 - 29), 0b1111111111) # 10 bits
    out[:,13] = np.bitwise_and(np.right_shift(values[:,1], 64 - 19), 0b111111) # 6 bits
    out[:,14] = np.bitwise_and(np.right_shift(values[:,1], 64 - 13), 0b1111) # 4 bits
    out[:,15] = np.bitwise_and(np.right_shift(values[:,1], 64 - 9), 0b1111) # 4 bits
    out[:,16] = np.right_shift(values[:,1], 64 - 5) # 5 bits
    
    out[:,17] = np.bitwise_and(np.right_shift(values[:,0], 64 - 35), 0b111111) # 6 bits 
    out[:,18] = np.bitwise_and(np.right_shift(values[:,0], 64 - 29), 0b11111) # 5 bits
    out[:,19] = np.bitwise_and(np.right_shift(values[:,0], 64 - 24), 0b1111) # 4 bits
    out[:,20] = np.bitwise_and(np.right_shift(values[:,0], 64 - 20), 0b1111) # 4 bits
    out[:,21] = np.bitwise_and(np.right_shift(values[:,0], 64 - 16), 0b1111) # 4 bits
    out[:,22] = np.bitwise_and(np.right_shift(values[:,0], 64 - 12), 0b1111) # 4 bits
    out[:,23] = np.bitwise_and(np.right_shift(values[:,0], 64 - 8), 0b1111) # 4 bits
    out[:,24] = np.bitwise_and(np.right_shift(values[:,0], 64 - 4), 0b1111) # 4 bits
    

def reshape_to_5d(count, frames, y, x, time_per_pixel):
    """
    Reshapes the 2D count matrix to a 5D array (frames, y, x, time, sensor)

    Args:
        count: N x 25 count matrix
        frames: number of frames contained in matrix
        y:
        x:
        time:

    Returns:
        A 5-D matrix with dimensions (frames, y, x, time, sensor)
    """

    return np.reshape(count, (frames, y, x, time_per_pixel, 25))

def reshape_to_6d(count, r, z, y, x, t=1, c=25):
    """
    Reshapes the data to a 6D array 

    Args:
        count: N x 25 count matrix
        frames: number of frames contained in matrix
        y:
        x:
        time:

    Returns:
        A 6-D matrix with dimensions (r, z, y, x, time, sensor)
    """

    return np.reshape(count, (r, z, y, x, t, 25))

def image2h5(fname, sum_time=True, save_time_ind=False):
    """
    Convert bin file to h5 file
        fname       file name
        sum_time    True to sum over all time bins, false otherwise
        save_time_ind Save all time frames in separate files
        TO DO:
            add metadata to file:
                data.pixelsize = 0.05
                data.pixelsizeU = 'um', etc.
    """
    print(fname)
    [out, frames, y, x, time_per_pixel] = file_to_count(fname)
    data = reshape_to_5d(out, frames, y, x, time_per_pixel)
    if np.ndim(data) == 4 and frames == 1 and sum_time:
        # 4D data set [y, x, time, ch] --> sum over time bins
        dataOut = np.sum(data, 2)
        dataOut = np.float64(dataOut)
        # channel must be first channel
        dataOut = np.transpose(dataOut, (2, 0, 1))
    elif np.ndim(data) == 5 and sum_time:
        # 5D data set [z, y, x, time, ch] --> sum over time bins
        dataOut = np.sum(data, 3)
        dataOut = np.float64(dataOut)
        dataOut = np.transpose(dataOut, (3, 0, 1, 2))
    else:
        print('not summed over time bins')
        # channel must be first channel
        dataOut = np.squeeze(data)
        if save_time_ind:
            dataOut = np.transpose(dataOut, (3, 0, 1, 2))
    
    dataOut = np.squeeze(dataOut)
    
    if type(save_time_ind) == bool and save_time_ind:
        for i in range(np.shape(dataOut)[-1]):
            print("Saving frame " + str(i))
            h5f = h5py.File(fname[:-4] + "_frame_" + str(i) + ".h5", 'w')
            h5f.create_dataset('dataset_1', data=dataOut[:,:,:,i])
            h5f.close()
    elif save_time_ind == "alternate":
        # channel, y, x, time
        Nt = np.shape(dataOut)[-1]
        for sumT in range(int(Nt/2)):
            print("Summing over " + str(sumT+1) + " frames")
            dataOutSum0 = np.squeeze(np.sum(dataOut[:,:,:,0:2*sumT+1:2], 3))
            dataOutSum1 = np.squeeze(np.sum(dataOut[:,:,:,1:2*sumT+2:2], 3))
            # store frame 0
            h5f = h5py.File(fname[:-4] + "_sum_" + str(sumT+1) + "_frame_0.h5", 'w')
            h5f.create_dataset('dataset_1', data=dataOutSum0)
            h5f.close()
            # store frame 1
            h5f = h5py.File(fname[:-4] + "_sum_" + str(sumT+1) + "_frame_1.h5", 'w')
            h5f.create_dataset('dataset_1', data=dataOutSum1)
            h5f.close()
    else:
        h5f = h5py.File(fname[:-3] + "h5", 'w')
        h5f.create_dataset('dataset_1', data=dataOut)
        h5f.close()
    
    return dataOut

def locseries2h5(fname, sumch = True, hot_pixels = []):
    """
    Convert bin file with time series of images to h5 file with channels summed
        fname       file name
    """
    print(fname)
    [out, frames, y, x, time_per_pixel] = file_to_count(fname)
    data = reshape_to_5d(out, frames, y, x, time_per_pixel)
    
    # t, x, y, c
    dataOut = np.squeeze(data)
    
    # sum over channels
    if sumch:
        Nch = np.shape(dataOut)[3]
        usedPixels = [i for i in range(Nch) if i not in hot_pixels]
        dataOut  = np.sum(dataOut[:,:,:,usedPixels], 3)
    
    dataOut = dataOut.astype(np.uint16)
    
    h5f = h5py.File(fname[:-3] + "h5", 'w')
    h5f.create_dataset('dataset_1', data=dataOut)
    h5f.close()
    
    return dataOut

    
def all_timetr_2_h5(folder=''):
    files = list_files(folder, filetype='ttr')
    filesh5 = list_files(folder, filetype='h5')
    for i, file in enumerate(filesh5):
        filesh5[i] = file[:-3] + '.ttr'
    for file in files:
        if file not in filesh5:
            print("converting " + file)
            #ttr2h5(file)
    print('Done')

def bin2h5(fname):
    """
    Convert bin file to h5 file with always 6D (r, z, x, y, t, c)
        fname       file name
            add metadata to file:
                data.pixelsize = 0.05
                data.pixelsizeU = 'um', etc.
    """

    [out, frames, y, x, time_per_pixel] = file_to_count(fname)
    data = reshape_to_6d(out, 1, frames, y, x, time_per_pixel, 25)
           
    # store data
    h5f = h5py.File(fname[:-4] + ".h5", 'w')
    h5f.create_dataset('dataset_1', data=data)
    h5f.close()
    print('Done')

def all_bin_images_2_tiff(folder):
    """
    Convert all bin files in a folder to tiff images
        folder      path to folder (use either \\ or / to go into a folder)
    """
    files = list_files(folder)
    for file in files:
        print("saving " + file)
        dummy = image2tiff(file)


def image2tiff(fname):
    """
    Convert bin file to tiff image file
        fname       file name
    """
    
    [out, frames, y, x, time_per_pixel] = file_to_count(fname)
    data = reshape_to_5d(out, frames, y, x, time_per_pixel)
    print(np.shape(data))
    info = get_file_info(fname[:-4] + '_info.txt')
    if np.ndim(data) == 4 and frames == 1:
        # 4D data set [y, x, time, ch] --> sum over time bins
        dataOut = np.sum(data, 2)
    elif np.ndim(data) == 5:
        # 5D data set [z, y, x, time, ch] --> sum over time bins and z
        dataOut = np.sum(data, 3)
        dataOut = np.sum(dataOut, 0)
    
    dataOut = np.float64(dataOut)    
    dataOut = np.squeeze(dataOut)
    
    print(np.shape(dataOut))
    
    array2tiff(dataOut, fname[:-4], pxsize=info.pxsize, dim="yxz", transpose3=True)
    array2rgbtiff(np.sum(dataOut, 2), fname[:-4] + '_RGB')


def all_bin_images_2_h5(folder, sum_time=True, save_time_ind=False):
    """
    Convert all bin files in a folder to h5 files
        folder      path to folder (use either \\ or / to go into a folder)
        sum_time    True to sum over all time bins, false otherwise
    """
    files = list_files(folder)
    for file in files:
        print("converting " + file)
        dummy = image2h5(file, sum_time, save_time_ind)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Converter from binary file to measurement matrix"
    )

    parser.add_argument(
        "binary",
        help="binary file name")

    args = parser.parse_args()
    fname = args.binary
    count, frames, y, x, time_per_pixel = file_to_count(fname)

    if count is None:
        print("Failed to process data. Closing.")
        sys.exit(0)

    file_name, extension = os.path.splitext(fname) # Get filename without extension

    print("Saving 5D matrix...", sep="")
    count5d = reshape_to_5d(count, frames, y, x, time_per_pixel)
    np.save(file_name + ".npy", count5d)
    print("Done.")

    




