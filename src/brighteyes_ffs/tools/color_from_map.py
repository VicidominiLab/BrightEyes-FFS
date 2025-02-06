import matplotlib.pyplot as plt
import numpy as np
import seaborn

def color_from_map(value, startv=0, stopv=1, cmap='viridis'):
    if cmap != 'seaborn-colorblind':
        cmap = plt.get_cmap(cmap)
        
        try:
            N = np.size(cmap.colors, 0)
        except:
            N = 255
        idx = int(np.floor((value - startv) / (stopv - startv) * N))
        idx = np.clip(idx, 0, N*1)
        
        try:
            c = cmap.colors[idx]
        except:
            c = cmap(idx/N)[0:3]
        
        return c
    else:
        cmapArray = seaborn.color_palette('colorblind')
        N = len(cmapArray)
        idx = int(np.floor((value - startv) / (stopv - startv) * N))
        idx = np.clip(idx, 0, N*1)
        return cmapArray[idx]
