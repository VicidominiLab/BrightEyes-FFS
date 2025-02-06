import numpy as np

def time2string(seconds):
    if seconds > 3600:
        progress = int(np.floor(seconds / 3600))
        if progress > 1:
            return str(progress) + " hours"
        else:
            return str(progress) + " hour"
        
    elif seconds > 60:
        progress = int(np.floor(seconds / 60))
        if progress > 1:
            return str(progress) + " minutes"
        else:
            return str(progress) + " minute"
        
    else:
        progress = int(np.floor(seconds))
        if progress > 1 or progress == 0:
            return str(progress) + " seconds"
        else:
            return str(progress) + " second"