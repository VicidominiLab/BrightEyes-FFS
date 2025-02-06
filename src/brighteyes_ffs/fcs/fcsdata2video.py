import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from ..tools.checkfname import checkfname


def fcsdata2video(data, fname='data.mp4', ftime=100, sum_all=False):
    """
    Convert SPAD-fcs data to 5x5 pixel video

    Parameters
    ----------
    data : 2D numpy array
        Data variable, i.e. output from binfile2data
        np.array(Nt x 25) with Nt number of time frames.
    fname : string, optional
        File name. The default is 'data.mp4'.
    ftime : scalar, optional
        Frame time [ms]. The default is 100.
    sum_all : boolean, optional
        False for normal 5x5 pixel video
        True for 1x1 pixel video with the sum of all pixels
            (single-element-detector-like). The default is False.

    Returns
    -------
    Saved video.

    """
    
    Nt = np.size(data, 0)
    
    if sum_all:
        # replace data with sum
        for i in range(Nt):
            data[i,:] = data[i,:] * 0 + np.sum(data[i,:])
    
    ims = []
    Imin = np.min(data)
    Imax = np.max(data)
    
    fig = plt.figure()
    FigSize = 10.5 # must be 10.5 to make the array size and video resolution match??
    
    fig.set_size_inches(FigSize, FigSize, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for i in range(Nt):
        im = ax.imshow(np.reshape(data[i,0:25], (5, 5)), vmin=Imin, vmax=Imax)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=ftime, blit=True)
    
    fname = checkfname(fname, 'mp4')
    
    ani.save(fname)


def partpos2video(pos, fname='video.mp4', limits=[2, 4, 2, 4, 5, 7], ftime=100, part_size=5, shadow_size=1, psf=[0.15, 0.15, 0.45]):
    """
    Convert SPAD-fcs particle positions to video

    Parameters
    ----------
    pos : 3D numpy array
        [Np x 3 x Nf] data array with
        Np: number of particles
        3:  x, y, z coordinates of the particle (float)
        Nf: number of frames.
    fname : string, optional
        File name. The default is 'video.mp4'.
    limits : list of 6 numbers, optional
        Plot axes limits [xmin, xmax, ymin, ymax, zmin, zmax]. The default is [2, 4, 2, 4, 5, 7].
    ftime : scalar, optional
        Frame time [ms]. The default is 100.
    part_size : scalar, optional
        Size of the dots for the particles. The default is 5.
    shadow_size : scalar, optional
        Size of the shadows of the particles, use 0 for no shadows. The default is 1.
    psf : list of 3 numbers, optional
        [wx, wy, z0] array with psf size. The default is [0.15, 0.15, 0.45].

    Returns
    -------
    Saved video.

    """
    
    Nf = np.shape(pos)[2]
    Np = np.shape(pos)[0]
    
    # plot limits
    xmin = limits[0]
    xmax = limits[1]
    ymin = limits[2]
    ymax = limits[3]
    zmin = limits[4]
    zmax = limits[5]
    
    pos = np.asarray(pos)
    # shadow 1
    posShadow1 = np.copy(pos)
    posShadow1[:, 2, :] = zmin
    # shadow 2
    posShadow2 = np.copy(pos)
    posShadow2[:, 1, :] = ymax
    pos = np.concatenate((pos, posShadow1), axis=0)
    pos = np.concatenate((pos, posShadow2), axis=0)
    
    # set default color of particles to green
    colorList = np.zeros((3*Np, 3, Nf))
    colorList[0:Np, :, :] = np.swapaxes(np.tile(np.array([0/256, 256/256, 0/256]), (Np, Nf, 1)), 1, 2)
    # change color of particles in laser to green
    counter = 0
    for f in range(Nf):
        for p in range(Np):
            if np.abs(pos[p, 0, f] - 3) < psf[0] and np.abs(pos[p, 1, f] - 3) < psf[1] and np.abs(pos[p, 2, f] - 6) < psf[2]:
                counter += 1
                colorList[p, :, f] = np.array([0/256, 256/256, 0/256])
    
    sizeList = np.zeros((3*Np, Nf))
    sizeList[0:Np, :] = np.tile(np.array([part_size]), (Np, Nf))
    if shadow_size > 0:
        sizeList[Np:3*Np, :] = np.tile(np.array([shadow_size]), (2*Np, Nf))
    else:
        sizeList[Np:3*Np, :] = np.tile(np.array([0]), (2*Np, Nf))
    
    # do not show particles outside of plot boundaries
    sizeList[pos[:,0,:] > xmax] = 0
    sizeList[pos[:,0,:] < xmin] = 0
    sizeList[pos[:,1,:] > ymax] = 0
    sizeList[pos[:,1,:] < ymin] = 0
    sizeList[pos[:,2,:] > zmax] = 0
    sizeList[pos[:,2,:] < zmin] = 0
    # if particles outside of boundaries, also remove shadows
    sizeList[np.concatenate((sizeList[0:Np,:] == 0, sizeList[0:Np,:] == 0, sizeList[0:Np,:] == 0), axis=0)] = 0
    
    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #ax.set_zticklabels([])
    
    ims = []
    for i in range(Nf):
        #[x, y, z] = draw_ellipsoid(psf[0], psf[1], psf[2], plotFig=False)
        #ax.plot_surface(x+2, y+2, z+6, rstride=4, cstride=4, color='b', alpha=0.2)
        im = ax.scatter3D(pos[:, 0, i], pos[:, 1, i], pos[:, 2, i], color=colorList[:,:,i], s=sizeList[:,i])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=ftime, blit=True)
    
    fname = checkfname(fname, 'mp4')
    
    ani.save(fname)


def fcstraces2video(data, fname='I_vs_t_video.mp4', ftime=100, dwellTime=1, dpi_user=250, xlabel="Time [s]", ylabel="PCR [Hz]", fsize=22, ylimits="auto"):
    """
    Convert SPAD-fcs data to line plot video with intensity traces over time

    WARNING: running this function from Jupyter Notebooks returns a lower
    quality video

    Parameters
    ----------
    data : 2D numpy array
        Data variable, i.e. output from binfile2data
        np.array(Nt x 25) with Nt number of time frames.
    fname : string, optional
        File name. The default is 'I_vs_t_video.mp4'.
    ftime : scalar, optional
        Frame time [ms]. The default is 100.
    dwellTime : scalar, optional
        Measurement dwell time [µs]. The default is 1.
    dpi_user : scalar, optional
        Video quality [dots per inch]. The default is 250.
    xlabel : string, optional
        x axis label. The default is "Time [s]".
    ylabel : string, optional
        y axis label. The default is "PCR [Hz]".
    fsize : scalar, optional
        text font size. The default is 22.
    ylimits : string or list of two numbers, optional
        y limits: [ymin ymax] or "auto". The default is "auto".

    Returns
    -------
    Saved video

    """
    
    if len(np.shape(data)) == 1:
        Nch = 1
    else:
        Nch = np.shape(data)[1]
    Nt = np.shape(data)[0]
    
    # bin time
    binTime = 1e-6 * dwellTime
    
    # time vector
    time = list(range(0, Nt))
    time = [i * binTime for i in time]
    
    # rescale intensity values to frequencies
    PCRscaled = data / binTime / 1000 # kHz
    ymax = np.max(PCRscaled)
    
    fig = plt.figure()
    plt.rcParams.update({'font.size': fsize})

    # axis limits
    if ylimits=="auto":
        ymin = 0
        ymax = 1.1 * ymax
    else:
        ymin = ylimits[0]
        ymax = ylimits[1]
    
    ax = plt.axes(xlim=(0, 2*time[-1] - time[-2]), ylim=(ymin, ymax))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    lines = [plt.plot([], [])[0] for _ in range(Nch)] #lines to animate
    
    def init():
        #init lines
        for line in lines:
            line.set_data([], [])
    
        return lines #return everything that must be updated
    
    def animate(i):
        #animate lines
        for j,line in enumerate(lines):
            line.set_data(time[0:i], PCRscaled[0:i, j])
    
        return lines #return everything that must be updated
    
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=ftime, blit=True, frames=Nt)
    
    fname = checkfname(fname, 'mp4')
    
    ani.save(fname, dpi=dpi_user)
    