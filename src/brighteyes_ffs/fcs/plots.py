import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ..fcs.fcs_polar import g2flow, g2polar
from ..fcs_gui.timetrace_end import timetrace_end
from ..tools.color_from_map import color_from_map
from ..tools.fit_curve import fit_curve

def plot_timetrace(time_trace, duration, chunksize=None, good_chunks=None, fig=None, ax=None, figsize=(4,2), cmap='viridis', return_fig=False):
    """
    Plot intensity time traces

    Parameters
    ----------
    time_trace : np.array()
        2D array [Nt x Nc] with Nt time points and Nc number of channels.
    duration : float
        Duration of the measurement (s).
    chunksize : float
        Duration of a single segment (s).
    good_chunks : list
        List of the good segments
    fig : plt.figure(), optional
        Figure. If none, open new figure. The default is None.
    ax : axis, optional
        Axis. If None, open new figure. The default is None.
    figsize : tuple, optional
        Size of the figure. The default is (4,2).
    cmap : str, optional
        Color map. The default is 'viridis'.
    return_fig : boolean, optional
        Return the figure and axis handle or not?

    Returns
    -------
    fig, ax. : figure and axis
        If False, nothing is returned.

    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    n_t = timetrace_end(time_trace) # time_trace is a compressed version of the actual time_trace with between 900-1000 data points
    time_trace = np.asarray(time_trace)
    
    # number of channesl
    if time_trace.ndim == 1:
        n_ch = 1
    if time_trace.ndim == 2:
        n_ch = int(time_trace.shape[1])
        
    time = np.linspace(0, duration, n_t)
    
    for i in range(n_ch):
        plt.plot(time, time_trace[0:n_t,i], c=color_from_map(i, 0, n_ch, cmap))
    ymin = np.min(time_trace)
    ymax = np.max(time_trace)
    
    if chunksize is not None and good_chunks is not None:
        num_chunks = int(np.floor(duration / chunksize))
        splits = np.arange(0, (num_chunks+1)*chunksize, chunksize)
        if len(splits) <= 101:
            # color chunks red if not used for calculating average correlation
            if good_chunks is not None:
                for i in range(len(splits) - 1):
                    if i not in good_chunks:
                        rect = patches.Rectangle((splits[i], ymin), (splits[i+1]-splits[i]), ymax-ymin, fc='r', alpha=0.3)
                        ax.add_patch(rect)
    
    plt.xlim([0, duration])
    plt.ylim([ymin, ymax])
    plt.xlabel('Time (s)')
    plt.ylabel('Photon counts per bin')
    plt.tight_layout()
    
    if return_fig:
        return fig, ax
    

def plot_atimetrace(raw_data, chunksize=None, good_chunks=None, fig=None, ax=None, figsize=(4,2), cmap='viridis', maxseg=1000, return_fig=False):
    """
    Plot time trace for time-tagged data

    Parameters
    ----------
    raw_data : ATimesData object
        Object with fields 'det0', 'det1', etc.
    chunksize : float
        Duration of a single segment (s).
    good_chunks : list
        List of the good segments
    fig : plt.figure(), optional
        Figure. If none, open new figure. The default is None.
    ax : axis, optional
        Axis. If None, open new figure. The default is None.
    figsize : tuple, optional
        Size of the figure. The default is (4,2).
    cmap : str, optional
        Color map. The default is 'viridis'.
    maxseg : int, optional
        Number of time points considered. The default is 1000.
    return_fig : boolean, optional
        Return the figure and axis handle or not?

    Returns
    -------
    fig, ax. : figure and axis
        If False, nothing is returned.

    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    time_trace = np.zeros((maxseg, raw_data.num_channels))
    
    time = getattr(raw_data, "det12")[:,0]
    timeAbs = time * raw_data.macrotime
    time_bins = np.linspace(0, np.max(timeAbs), maxseg + 1)
    n_ch = raw_data.num_channels
    
    for i, det in enumerate(raw_data.all_channels):
        time = getattr(raw_data, det)[:,0]
        timeAbs = time * raw_data.macrotime
        [Itrace, timeBins] = np.histogram(timeAbs, time_bins)
        time_trace[:, i] = Itrace[0:] #/ (timeBins[2] - timeBins[1]) / 1e3
    
    for i in range(n_ch):
        plt.plot(timeBins[:-1], time_trace[:,i] / (timeBins[2] - timeBins[1]) / 1e3, color=color_from_map(i, 0, n_ch, cmap))
    plt.xlabel('Time (s)')
    plt.ylabel('Photon count rate (kHz)')
    plt.tight_layout()
    
    if return_fig:
        return fig, ax


def plot_atimeshist(raw_data, plot_fields='hist', fig=None, ax=None, figsize=(4,3), cmap='viridis', return_fig=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    plot_channels = raw_data.get_channels(plot_fields)

    for i, ch in enumerate(plot_channels):
        hist = getattr(raw_data, ch)
        ax.plot(1e-3*hist[:,0], hist[:,1], color=color_from_map(i, 0, len(plot_channels), cmap))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Nr. of photons")
    plt.tight_layout()
    
    if return_fig:
        return fig, ax
    

def plot_corrs(G, idx=None, fig=None, ax=None, figsize=None, cmap='viridis', return_fig=False):
    """
    Plot correlations

    Parameters
    ----------
    G : Correlations object
        Correlations object from fcs2corr .
    idx : list
        List of good segments
    fig : plt.figure(), optional
        Figure. If none, open new figure. The default is None.
    ax : axis, optional
        Axis. If None, open new figure. The default is None.
    figsize : tuple, optional
        Size of the figure. The default is (4,2).
    cmap : str, optional
        Color map. The default is 'viridis'.
    return_fig : boolean, optional
        Return the figure and axis handle or not? Default is false.

    Returns
    -------
    fig, ax. : figure and axis
        If False, nothing is returned.

    """
    list_of_g_out = G.list_of_g_out
    
    if len(list_of_g_out) > 6:
        list_of_g_out = list_of_g_out[0:6]
    
    n_chunks = G.num_chunks
    n_corr = len(list_of_g_out)
    
    if figsize is None:
        figsize = (3*n_corr,3)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, n_corr, figsize=figsize)

    for i, corr in enumerate(list_of_g_out):
        for j in range(n_chunks):
            Gsingle = getattr(G, corr + '_chunk' + str(j))
            if idx is not None:
                # color according to removed or not
                color = 'r'
                if j in idx:
                    color = 'grey'
            else:
                # random color
                color = color_from_map(j, 0, n_chunks, cmap)
            ax[i].scatter(Gsingle[1:,0], Gsingle[1:,1], s=4, color=color, label='chunk ' + str(j))
        
        if idx is not None:
            ax[i].plot(Gsingle[1:,0], getattr(G, corr + '_averageX')[1:,1], 'k')
        ax[i].set_xscale('log')
        ax[i].set_title(corr)
        ax[i].set_xlabel('Lag time (s)')
        ax[i].set_ylabel('G')
    plt.tight_layout()
    
    if return_fig:
        return fig, ax


def plot_corrs_av(G, av='_averageX', fig=None, ax=None, figsize=None, cmap='tab20c', return_fig=False):
    """
    Plot average correlations

    Parameters
    ----------
    G : Correlations object
        Correlations object from fcs2corr .
    av : str
        Use '_average' for average over all segments or
        '_averageX' for average over good chunks only
    fig : plt.figure(), optional
        Figure. If none, open new figure. The default is None.
    ax : axis, optional
        Axis. If None, open new figure. The default is None.
    figsize : tuple, optional
        Size of the figure. The default is (4,2).
    cmap : str, optional
        Color map. The default is 'viridis'.
    return_fig : boolean, optional
        Return the figure and axis handle or not? Default is false.

    Returns
    -------
    fig, ax. : figure and axis
        If False, nothing is returned.

    """
    base_names, max_filt = G.list_of_g_out_filters
    list_of_g_out = G.list_of_g_out
    
    if max_filt is None:
        if figsize is None:
            figsize = (4,3)
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    
        for i, corr in enumerate(list_of_g_out):
            Gsingle = getattr(G, corr + av)
                
            color = color_from_map(i, 0, len(list_of_g_out), cmap)
            ax.scatter(Gsingle[1:,0], Gsingle[1:,1], s=4, color=color, label=corr)
            ax.set_xscale('log')
            ax.set_xlabel('Lag time (s)')
            ax.set_ylabel('G')
        if len(list_of_g_out) < 8:
            plt.legend()
    else:
        # filters used
        n_filt = max_filt + 1
        filters = ["F" + str(i) for i in range(n_filt)]
        filters[0] = ''
        
        if figsize is None:
            figsize = (2*n_filt,3)
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, n_filt, figsize=figsize)
        
        for f, filtername in enumerate(filters):
            for i, corr in enumerate(base_names):
                Gsingle = getattr(G, corr + filtername + av)
                    
                color = color_from_map(i, 0, len(list_of_g_out), cmap)
                ax[f].scatter(Gsingle[1:,0], Gsingle[1:,1], s=4, color=color, label=corr)
                ax[f].set_xscale('log')
                ax[f].set_xlabel('Lag time (s)')
                ax[f].set_ylabel('G')
            if len(list_of_g_out) < 8:
                ax[f].legend()
    
    plt.tight_layout()
    
    if return_fig:
        return fig, ax
    

def plot_corrs_fit(G_all, tau, list_of_g_out, fitresults, fig=None, ax=None, figsize=None, size=12, color=color_from_map(0.3, 0, 1, 'viridis'), cmap='viridis', plot_3_col=False, return_fig=False):
    """
    Plot correlations and fit

    Parameters
    ----------
    G_all : np.array()
        2D array [Ntau x Ng] with Ng correlations of length Ntau.
    tau : np.array()
        1D array [Ntau] with the tau values.
    list_of_g_out : list
        List of str with correlation names.
    fitresults : list
        List of fit results, each element output of fcs_fit.
    fig : plt.figure(), optional
        Figure. If none, open new figure. The default is None.
    ax : axis, optional
        Axis. If None, open new figure. The default is None.
    figsize : tuple, optional
        Size of the figure. The default is None.
    size : int, optional
        Size of the scatter points in the plot. The default is 12.
    color : color value, optional
        Color of the scatter plot. The default is color_from_map(0.3, 0, 1, 'viridis').
    return_fig : boolean, optional
        Return the figure and axis handle or not? Default is false.

    Returns
    -------
    fig, ax. : figure and axis
        If False, nothing is returned.

    """
    
    n_corr = len(list_of_g_out)
    
    try:
        temp = fitresults.fun[:,0]
        fit = 'global'
    except:
        fit = 'individual'
    
    if n_corr < 7 and not plot_3_col:
        # plot each corr and fit in separate panel
        if figsize is None:
            figsize = (3*n_corr,3)
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(2, n_corr, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    
        for i, corr in enumerate(list_of_g_out):
            ax[0,i].scatter(tau, G_all[:,i], s=size, color=color, label=corr)
            
            if fitresults is not None:
                # plot fit
                if fit == 'global':
                    residuals = fitresults.fun[:,i]
                    plot_fit = G_all[:,i]-residuals
                else:
                    fitresult = fitresults[i]
                    residuals = fitresult.fun
                    plot_fit = G_all[:,i]-residuals
            
                ax[0,i].plot(tau, plot_fit, c='k')
                ax[1,i].plot(tau, residuals, c=color)
            
            ax[0,i].set_xscale('log')
            ax[0,i].set_title(corr)
            ax[0,i].set_ylabel('G')
            ax[1,i].set_xscale('log')
            ax[1,i].set_xlabel('Lag time (s)')
            ax[1,i].set_ylabel('Residuals')
    else:
        # plot 3 panels: corr - fit - both
        if figsize is None:
            figsize = (8,3)
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 3, figsize=figsize)
    
        for i, corr in enumerate(list_of_g_out):
            color = color_from_map(i, 0, len(list_of_g_out), cmap)
            ax[0].scatter(tau, G_all[:,i], s=size, color=color, label=corr)
            ax[2].scatter(tau, G_all[:,i], s=size, color=color, label=corr)
            
            if fitresults is not None:
                # plot fit
                if fit == 'global':
                    residuals = fitresults.fun[:,i]
                    plot_fit = G_all[:,i]-residuals
                else:
                    fitresult = fitresults[i]
                    residuals = fitresult.fun
                    plot_fit = G_all[:,i]-residuals
            
                ax[1].plot(tau, plot_fit, c=color)
                ax[2].plot(tau, plot_fit, c='k')
        
        for i in range(3):
            ax[i].set_xscale('log')
            #ax[i].set_title(corr)
            ax[i].set_ylabel('G')
            ax[i].set_xscale('log')
            ax[i].set_xlabel('Lag time (s)')
            ax[i].set_ylabel('Residuals')
    
    plt.tight_layout()
    
    if return_fig:
        return fig, ax
    

def plot_difflaw(taufit, w0, fig=None, ax=None, figsize=None, cmap='viridis', return_fig=False):
    """
    Plot diffusion law

    Parameters
    ----------
    taufit : list of float
        Fitted tau values.
    w0 : list of float
        w0 values.
    fig : plt.figure(), optional
        Figure. If none, open new figure. The default is None.
    ax : axis, optional
        Axis. If None, open new figure. The default is None.
    figsize : tuple, optional
        Size of the figure. The default is None.
    cmap : str, optional
        Color map. The default is 'viridis'.
    return_fig : boolean, optional
        Return the figure and axis handle or not? Default is false.

    Returns
    -------
    fig, ax. : figure and axis
        If False, nothing is returned.

    """
    fitresult = fit_curve(taufit, (w0)**2, 'linear', [1, 1], [1, 1], [-1e6, -1e6], [1e6, 1e6], savefig=0)
    n_points = len(taufit)
    
    if figsize is None:
        figsize = (3,3)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    for i in range(n_points):
        plt.scatter(w0[i]**2, taufit[i], edgecolors='k', marker='o', color=color_from_map(i, 0, n_points, cmap))
        
    w02fit = np.zeros(len(w0) + 1)
    w02fit[0] = 0
    w02fit[1:] = w0**2
    taufitres = np.zeros(len(w0) + 1)
    taufitres[0] = fitresult.x[1]
    taufitres[1:] = taufit - fitresult.fun
    if fitresult.x[1] < 0:
        fitlabel = 'y = {A:.4f} x {B:.4f}'.format(A=fitresult.x[0], B=fitresult.x[1])
    else:
        fitlabel = 'y = {A:.4f} x + {B:.4f}'.format(A=fitresult.x[0], B=fitresult.x[1])
    plt.plot(w02fit, taufitres, '--', color='k', linewidth=0.7, label=fitlabel, zorder=1)
    plt.title(fitlabel)
    plt.xlabel('w0^2 (um^2)')
    plt.ylabel('Diffusion time (ms)')
    
    plt.tight_layout()
    
    if return_fig:
        return fig, ax


def plot_flow_heat_map(corrs, detector='square', fig=None, ax=None, figsize=None, cmap='PiYG', return_fig=False):
    """
    Plot a flow heat map from a set of correlation curves

    Parameters
    ----------
    corrs : np.array()
        2D array [Ntau x Ng] with Ng the number of correlation curves to use.
        E.g.
            corrs_for_flow = ['V0_H1', 'V-1_H0', 'V0_H-1', 'V1_H0']
            corrs, _, _ = G.get_av_corrs(corrs_for_flow, av='_average')
    detector : str, optional
        Detector used. The default is 'square'.
    fig : plt.figure(), optional
        Figure. If none, open new figure. The default is None.
    ax : axis, optional
        Axis. If None, open new figure. The default is None.
    figsize : tuple, optional
        Size of the figure. The default is None.
    cmap : str, optional
        Color map. The default is 'viridis'.
    return_fig : boolean, optional
        Return the figure and axis handle or not? Default is false.

    Returns
    -------
    fig, ax. : figure and axis
        If False, nothing is returned.

    """
    
    z, flow = g2flow(corrs, detector=detector)
    R = len(z) / 2 - 1
    phi = np.linspace(0, 2*np.pi, 360)
    
    if figsize is None:
        figsize = (3,3)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    plt.imshow(np.flipud(z), cmap=cmap, extent=[-R, R, -R, R], interpolation="nearest")
    #plt.arrow(90-(r/2), 90-(u/2), 2*r, 2*u, width=1, head_width=4, color='k', length_includes_head=True)
    plt.plot(R*np.cos(phi), R*np.sin(phi), '-', color='k', linewidth=1)
    plt.xlim([-1.1*R, 1.1*R])
    plt.ylim([-1.1*R, 1.1*R])
    plt.axis('off')
    
    if return_fig:
        return fig, ax


def plot_anisotropy_heat_map(corrs, fig=None, ax=None, figsize=None, cmap='PiYG', return_fig=False):
    """
    Plot a flow heat map from a set of correlation curves

    Parameters
    ----------
    corrs : np.array()
        2D array [Ntau x Ng] with Ng the number of correlation curves to use.
        E.g.
            corrs_for_flow = ['V0_H1', 'V-1_H0', 'V0_H-1', 'V1_H0']
            corrs, _, _ = G.get_av_corrs(corrs_for_flow, av='_average')
    fig : plt.figure(), optional
        Figure. If none, open new figure. The default is None.
    ax : axis, optional
        Axis. If None, open new figure. The default is None.
    figsize : tuple, optional
        Size of the figure. The default is None.
    cmap : str, optional
        Color map. The default is 'viridis'.
    return_fig : boolean, optional
        Return the figure and axis handle or not? Default is false.

    Returns
    -------
    fig, ax. : figure and axis
        If False, nothing is returned.

    """
    
    z = g2polar(corrs, Nr=1024)
    
    R = len(z) / 2
    phi = np.linspace(0, 2*np.pi, 360)
    
    if figsize is None:
        figsize = (3,3)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    plt.imshow(np.flipud(z), cmap=cmap, extent=[-R, R, -R, R], interpolation="nearest")
    plt.plot(R*np.cos(phi), R*np.sin(phi), '-', color='k', linewidth=1)
    plt.xlim([-1.1*R, 1.1*R])
    plt.ylim([-1.1*R, 1.1*R])
    plt.axis('off')
    
    if return_fig:
        return fig, ax


def plot_fida_hist(G, fitresults=None, xlim=[-0.5,28.5], ylim=[1e-6,1], yscale='log', fig=None, ax=None, figsize=None, cmap='tab20c', return_fig=False):
    """
    Plot FIDA histograms with/without fit

    Parameters
    ----------
    G : Correlations object
        Object with histograms.
    fitresults : fitresult, optional
        Either list with each element output of fit_pch or single output of fit_pch
        in case of global fit. The default is None.
    xlim : list, optional
        X limits. The default is [-0.5,28.5].
    ylim : list, optional
        Y limits. The default is [1e-6,1].
    yscale : str, optional
        'log' or 'linear' y scale. The default is 'log'.
    fig : plt.figure(), optional
        Figure. If none, open new figure. The default is None.
    ax : axis, optional
        Axis. If None, open new figure. The default is None.
    figsize : tuple, optional
        Size of the figure. The default is None.
    cmap : str, optional
        Color map. The default is 'tab20c'.
    return_fig : boolean, optional
        Return the figure and axis handle or not? Default is false.

    Returns
    -------
    fig, ax. : figure and axis
        If False, nothing is returned.

    """
    list_of_g_out = G.list_of_g_out
    
    try:
        temp = fitresults.fun[:,0]
        fit = 'global'
    except:
        fit = 'individual'
    
    if figsize is None:
        figsize = (4,3)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    for i, corr in enumerate(list_of_g_out):
        Gsingle = getattr(G, corr + '_averageX')
        plt.bar(Gsingle[0:,0], Gsingle[0:,1], label=corr, color=color_from_map(i, 0, len(list_of_g_out), cmap), alpha=0.4)
        if fitresults is not None:
            # plot fit
            if fit == 'global':
                residuals = fitresults.fun[:,i]
                plot_fit = Gsingle[:,1]-residuals
            else:
                fitresult = fitresults[i]
                plot_fit = Gsingle[:,1]-fitresult.fun
            
            plt.plot(Gsingle[:,0], plot_fit, linewidth=1.5, color=color_from_map(i, 0, len(list_of_g_out), cmap))
    plt.legend()
    plt.xlabel('Counts per bin')
    plt.ylabel('Relative frequency')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xscale('linear')
    plt.yscale(yscale)
    plt.tight_layout()
    
    if return_fig:
        return fig, ax