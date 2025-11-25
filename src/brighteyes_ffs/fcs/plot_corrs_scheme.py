from .detectors import detector_element_coordinates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import CirclePolygon
from ..tools.color_from_map import color_from_map
from .extract_spad_data_kw import keyword_2_ch
import re


def plot_correlations_microimage(detector, list_of_g, list_of_g_out=None, averaging=None, figsize=(5,5)):
    """
    Plot scheme of the detector with correlations indicated as colored pixels
    or arrows.
    Uses output from get_corr_input, e.g.
    list_of_g, list_of_g_out, averaging = get_corr_input('pair-correlation fcs', 'airyscan')
    plot_correlations_microimage('airyscan', list_of_g, list_of_g_out, averaging)

    Parameters
    ----------
    detector : str
        DESCRIPTION.
    list_of_g : list
        List of the correlations that are calculated.
    list_of_g_out : list of str, optional
        Names of the correlations. The default is None.
    averaging : list, optional
        List of which correlations are averaged. The default is None.
    figsize : tuple, optional
        Figure size. The default is (5,5).

    Returns
    -------
    Plot.

    """
    
    xx = detector_element_coordinates(detector, element=None)
    
    if type(list_of_g[0]) is not str or str.lower(list_of_g[0]) != 'crossall':
        
        if len(list_of_g) > 6:
            n_rows = int(np.ceil(len(list_of_g)/6))
            n_columns = 6
        else:
            n_rows = 1
            n_columns = len(list_of_g)
        
        fig, axs = plt.subplots(n_rows, n_columns, figsize=figsize, squeeze=False, constrained_layout=True)
        axs = np.atleast_1d(axs)
        used_ch = []
        
        if list_of_g_out is None:
            list_of_g_out = list_of_g
        
        for idx_scalar, corr in enumerate(list_of_g):
    
            row = idx_scalar // 6
            col = idx_scalar % 6
            ax = axs[row, col]
            clr = color_from_map(32, startv=0, stopv=41, cmap='inferno')
    
            # draw detector elements
            for c in range(len(xx)):
                ax.add_patch(CirclePolygon((xx[c, 0], xx[c, 1]), radius=0.45, resolution=51, alpha=1, color='grey'))
            ax.set_xlim([np.min(xx)-0.5, np.max(xx)+0.5])
            ax.set_ylim([np.min(xx)-0.5, np.max(xx)+0.5])        
    
            if type(corr) == int:
                # autocorrelation single channel
                ax.add_patch(CirclePolygon((xx[corr, 0], xx[corr, 1]), radius=0.45, resolution=51, alpha=1, color=clr))
                used_ch.append(xx[corr, :])
                
            elif corr[0] == 'x':
                # cross-correlation two channels, e.g., x0412 between 4 and 12
                c0 = int(corr[1:3]) # first channel
                c1 = int(corr[3:5]) # second channel
                used_ch.append(xx[c0, :])
                used_ch.append(xx[c1, :])
                if c0 == c1:
                    ax.add_patch(CirclePolygon((xx[c0, 0], xx[c0, 1]), radius=0.45, resolution=51, alpha=1, color=clr))
                else:
                    ax.add_patch(CirclePolygon((xx[c0, 0], xx[c0, 1]), radius=0.45, resolution=51, alpha=1, color=clr))
                    ax.add_patch(CirclePolygon((xx[c1, 0], xx[c1, 1]), radius=0.45, resolution=51, alpha=1, color=clr))
                    ax.arrow(xx[c0, 0], xx[c0, 1], xx[c1, 0]-xx[c0, 0], xx[c1, 1]-xx[c0, 1], width=0.1, length_includes_head=True, alpha=0.5, color='k', edgecolor=None)
                
            elif corr[0] == 'C':
                # crosscorrelation custom sum of channels
                xpos = np.max([corr.find('X'), corr.find('x')])
                if xpos > -1:
                    sum_ch = [int(i) for i in re.findall(r'\d+', corr[0:xpos])]
                    ch0_mean = np.mean(xx[sum_ch,:], 0)
                    for k in sum_ch:
                        used_ch.append(xx[k, :])
                        ax.add_patch(CirclePolygon((xx[k, 0], xx[k, 1]), radius=0.45, resolution=51, alpha=0.5, color=clr))
                    sum_ch = [int(i) for i in re.findall(r'\d+', corr[xpos:])]
                    ch1_mean = np.mean(xx[sum_ch,:], 0)
                    for k in sum_ch:
                        clr2 = color_from_map(32, startv=0, stopv=41, cmap='inferno')
                        ax.add_patch(CirclePolygon((xx[k, 0], xx[k, 1]), radius=0.45, resolution=51, alpha=1, color=clr2))
                        used_ch.append(xx[k, :])
                    ax.arrow(ch0_mean[0], ch0_mean[1], ch1_mean[0]-ch0_mean[0], ch1_mean[1]-ch0_mean[1], width=0.1, length_includes_head=True, alpha=0.5, color='k', edgecolor=None)
                    
                else:
                    sum_ch = [int(i) for i in re.findall(r'\d+', corr)]
                    for k in sum_ch:
                        ax.add_patch(CirclePolygon((xx[k, 0], xx[k, 1]), radius=0.45, resolution=51, alpha=1, color=clr))
                        used_ch.append(xx[k, :])
            
            else:
                # keyword
                sum_ch = keyword_2_ch[corr]
                for k in sum_ch:
                    ax.add_patch(CirclePolygon((xx[k, 0], xx[k, 1]), radius=0.45, resolution=51, alpha=1, color=clr))
                    used_ch.append(xx[k, :])
            
            plt.xticks([])
            plt.yticks([])
            ax.set_axis_off()
            ax.set_title(str(list_of_g_out[idx_scalar]), fontsize=9)
            ax.set_box_aspect(1)
        
        for idx_scalar, corr in enumerate(list_of_g):
            row = idx_scalar // 6
            col = idx_scalar % 6
            ax = axs[row, col]
            for i in xx:
                if not any(np.array_equal(i, pt) for pt in used_ch):
                    ax.add_patch(CirclePolygon((i[0], i[1]), radius=0.45, resolution=51, alpha=0.7, color='white'))
                
    else:
        #cross all
        if len(averaging) > 6:
            n_rows = int(np.ceil(len(averaging)/6))
            n_columns = 6
        else:
            n_rows = 1
            n_columns = len(averaging)
        
        fig, axs = plt.subplots(n_rows, n_columns, figsize=figsize, squeeze=False, constrained_layout=True)
        axs = np.atleast_1d(axs)
        used_ch = []
        
        if list_of_g_out is None:
            list_of_g_out = averaging
        
        for idx_scalar, av in enumerate(averaging):
            
            row = idx_scalar // 6
            col = idx_scalar % 6
            ax = axs[row, col]
            
            # draw detector elements
            for c in range(len(xx)):
                ax.add_patch(CirclePolygon((xx[c, 0], xx[c, 1]), radius=0.45, resolution=51, alpha=1, color='grey'))
            ax.set_xlim([np.min(xx)-0.5, np.max(xx)+0.5])
            ax.set_ylim([np.min(xx)-0.5, np.max(xx)+0.5])       
        
            all_ch = [int(ch_nr) for ch_nr in re.findall(r'\d+', av)]
            for j in range(int(len(all_ch)/2)):
                c0 = all_ch[2*j]
                c1 = all_ch[2*j+1]
                used_ch.append(xx[c0, :])
                used_ch.append(xx[c1, :])
                clr0 = color_from_map(32, startv=0, stopv=41, cmap='inferno')
                ax.add_patch(CirclePolygon((xx[c0, 0], xx[c0, 1]), radius=0.45, resolution=51, alpha=1, color=clr0))
                ax.add_patch(CirclePolygon((xx[c1, 0], xx[c1, 1]), radius=0.45, resolution=51, alpha=1, color=clr0))
            for j in range(int(len(all_ch)/2)):
                c0 = all_ch[2*j]
                c1 = all_ch[2*j+1]
                ax.arrow(xx[c0, 0], xx[c0, 1], xx[c1, 0]-xx[c0, 0], xx[c1, 1]-xx[c0, 1], width=0.1, length_includes_head=True, alpha=0.5, color='k', edgecolor=None)
        
            plt.xticks([])
            plt.yticks([])
            ax.set_axis_off()
            ax.set_title(str(list_of_g_out[idx_scalar]), fontsize=9)
            ax.set_box_aspect(1)
        
        for idx_scalar, corr in enumerate(averaging):
            row = idx_scalar // 6
            col = idx_scalar % 6
            ax = axs[row, col]
            for i in xx:
                if not any(np.array_equal(i, pt) for pt in used_ch):
                    ax.add_patch(CirclePolygon((i[0], i[1]), radius=0.45, resolution=51, alpha=0.7, color='white'))
    
    while idx_scalar < n_columns * n_rows:
        row = idx_scalar // 6
        col = idx_scalar % 6
        ax = axs[row, col]
        plt.xticks([])
        plt.yticks([])
        ax.set_axis_off()
        idx_scalar += 1