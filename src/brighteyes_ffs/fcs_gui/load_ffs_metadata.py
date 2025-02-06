from ..fcs.get_fcs_info import get_file_info, get_metafile_from_file
from .analysis_settings import FFSmetadata

def load_ffs_metadata(fname):
    """
    Load FFS metadata. Function assumes that metadata is stored in 
    'ffs_info.txt' with ffs the raw file name  (i.e. "fname" without .bin), e.g.
    'myFCSmeasurement.bin' is the raw data file, and
    'myFCSmeasurement_info.txt' is the metadata file
    OR
    metadata is stored in same .h5 file as the actual data
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    ffs         File name of raw data file (not metadata file).
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    metadata    analysis_settings metadata object with all fields filled in
                returns None if the file is not found
    ===========================================================================
    """
    
    # metadata
    fname = get_metafile_from_file(fname)
    md = get_file_info(fname)
    if md is not None:
        metadata = FFSmetadata()
        metadata_keys = [
            'num_pixels',
            'num_lines',
            'num_frames',
            'range_x',
            'range_y',
            'range_z',
            'num_datapoints',
            'hold_off_x5',
            'hold_off',
            'time_resolution',
            'dwelltime',
            'duration',
            'pxsize',]
        md_keys = [
            'numberOfPixels',
            'numberOfLines',
            'numberOfFrames',
            'rangeX',
            'rangeY',
            'rangeZ',
            'numberOfDataPoints',
            'holdOffx5',
            'holdOff',
            'timeResolution',
            'dwellTime',
            'duration',
            'pxsize',]
        for i, metadata_key in enumerate(metadata_keys):
            if md_keys[i] in list(md.__dict__.keys()) and getattr(md, md_keys[i]) is not None:
                setattr(metadata, metadata_key, getattr(md, md_keys[i]))
            else:
                setattr(metadata, metadata_key, 0)
        # get coordinates from file name
        # Cells_DEKegfp_4_LP70_75x75um_1500x1500px_y_52_x_1480.bin
        # returns [52, 1480]
        idx = fname.rfind("y_") + 2
        idx2 = fname.rfind("_x_")
        if fname[idx2+3:].rfind("_"):
            idx3 = fname.rfind("_")
        else:
            idx3 = fname.rfind(".")
        try:
            y = int(fname[idx:idx2])
        except:
            y = 0
        try:
            x = int(fname[idx2+3:idx3])
        except:
            x = 0
        metadata.coords = [y, x]
    else:
        metadata = None
    return metadata
