import h5py

# how to use
# file = h5data('myfilename.h5')
# file.pxsize()
#   0.040
#
# to do: detector hold-off

class h5data():
    def __init__(self, fname):
        self.fname = fname
    
    def rangex(self):
        # range in um
        with h5py.File(self.fname, "r") as f:
            return(f['configurationGUI_beforeStart'].attrs['range_x'])
    
    def rangey(self):
        # range in um
        with h5py.File(self.fname, "r") as f:
            return(f['configurationGUI_beforeStart'].attrs['range_y'])
    
    def rangez(self):
        # range in um
        with h5py.File(self.fname, "r") as f:
            return(f['configurationGUI_beforeStart'].attrs['range_z'])
                
    def tbinpx(self):
        # number of time bins per pixel
        with h5py.File(self.fname, "r") as f:
            return(f['configurationGUI_beforeStart'].attrs['timebin_per_pixel'])
    
    def tres(self):
        # time resolution in us
        with h5py.File(self.fname, "r") as f:
            return(f['configurationGUI_beforeStart'].attrs['time_resolution'])
    
    def pxdwelltime(self):
        # pixel dwell time in us
        return(self.tres() * self.tbinpx())
    
    def frametime(self):
        # frame time in s
        return(self.pxdwelltime() * self.nx() * self.ny() / 1e6)
    
    def framerate(self):
        # frame rate in Hz
        return(1 / self.frametime())
    
    def nx(self):
        # number of pixels in x direction
        with h5py.File(self.fname, "r") as f:
            return(f['configurationGUI_beforeStart'].attrs['nx'])
            
    def ny(self):
        # number of pixels in y direction
        with h5py.File(self.fname, "r") as f:
            return(f['configurationGUI_beforeStart'].attrs['ny'])
    
    def nz(self):
        # number of frames
        with h5py.File(self.fname, "r") as f:
            return(f['configurationGUI_beforeStart'].attrs['nframe'])
    
    def nrep(self):
        # number of frames
        with h5py.File(self.fname, "r") as f:
            return(f['configurationGUI_beforeStart'].attrs['nrep'])
    
    def pxsize(self): # also y and z
        # pixel size in x direction
        return(self.rangex() / self.nx())
    
    def nmicroim(self):
        # total number of microimages read during the measurement
        return(self.nx() * self.ny() * self.nz() * self.nrep() * self.tbinpx())
    
    def ndatapoints(self):
        # total number of words transferred from low level to high level
        # 2 words per microimage
        return(2 * self.nmicroim())
    
    def duration(self):
        # total measurement duration in s
        return(self.nmicroim() * self.tres() * 1e-6)
    
    
def printmetadata(file):
    if not isinstance(file, h5data):
        file = h5data(file)
    for prop in ['tbinpx', 'nx', 'ny', 'nz', 'nrep',
                 'tres', 'rangex', 'rangey', 'rangez',
                 'pxdwelltime', 'frametime', 'framerate', 'pxsize',
                 'ndatapoints', 'duration']:
        print(prop, end = '')
        print(' ' * int(14 - len(prop)), end = '')
        print(str(getattr(file, prop)()))