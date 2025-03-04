"""
    Stored in memory:
    root.image
    root.imageRaw
    root.firstFile
    root.activeFileButton
    root.FFSsettings
    (root.defaultMetaData
    root.airyDummy
    root.xtrace
    root.ytrace)
    
    To do: save corrsettings average?"""

import os
import shutil
import numpy as np
import h5py
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime

from .analysis_settings import FFSlib, FFSfile, FFSmetadata, FFScorr, CorrSettings, FitSingleObj, CorrFit
from ..fcs.fcs2corr import Correlations
from .fitmodel_class import FitModel

from ..tools.csv2array import csv2array, array2csv
from ..tools.checkfname import checkfname

#current = os.path.dirname(os.path.realpath('restore_session.py'))
#parent = os.path.dirname(current)
#sys.path.append(parent)


def restorelib(libfile, root=0):
    # restore session from .ffs or .ffz lib file
    if str(libfile).endswith('.ffs'):
        lib = restorelib_ffs(libfile, root)
        return lib
    if str(libfile).endswith('.ffz') or str(libfile).endswith('.zip'):
        lib = restorelib_ffz(libfile, root)
        return lib


def restorelib_ffz(libfile, root=0):
    # unzip file
    if root != 0:
        root.progress = 0
        root.progressMessage = 'Opening file...'
    folder = unzip(libfile)
    
    # create session library
    lib = FFSlib()
    lib.version = read_text(os.path.join(folder, 'version.txt'))
    lib.notes = read_text(os.path.join(folder, 'notes.txt'))
    lib.active_image = read_text(os.path.join(folder, 'activeImage.txt'), dtype=int)
    if lib.version != '1.0':
        lib.date_created = read_text(os.path.join(folder, 'date_created.txt'), dtype=str)
        lib.date_modified = read_text(os.path.join(folder, 'date_modified.txt'), dtype=str)
    
    # walk through folders
    images = next(os.walk(folder))[1]
    Nimages = len(images)
    for i in range(Nimages):
        if root != 0:
            root.progress = i / Nimages
            root.progressMessage = 'Reading image ' + str(i+1) + '...'
        imagefolder = os.path.join(folder, 'image' + str(i))
        imagepath = os.path.join(imagefolder, 'image.csv')
        image = csv2array(imagepath)
        imagenamepath = os.path.join(imagefolder, 'imageName.txt')
        fname = read_text(imagenamepath)
        # add image
        lib.add_image(image, fname)
        imageObj = lib.get_image(i)
        # active file
        imageObj.active_ffs = read_text(os.path.join(imagefolder, 'activeFFS.txt'), dtype=int)
        
        # check for files in image
        files = next(os.walk(imagefolder))[1]
        Nfiles = len(files)
        for j in range(Nfiles):
            if root != 0:
                root.progress = i / Nimages + j / (Nfiles * Nimages)
                root.progressMessage = 'Reading FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
            fileObj = FFSfile()
            filefolder = os.path.join(imagefolder, 'file' + str(j))
            fileObj.fname = read_text(os.path.join(filefolder, 'fname.txt'))
            fileObj.label = read_text(os.path.join(filefolder, 'label.txt'))
            fileObj.timetrace = csv2array(os.path.join(filefolder, 'timetrace.csv'))
            if -1 in fileObj.timetrace:
                fileObj.timetrace = None
            fileObj.airy = csv2array(os.path.join(filefolder, 'airy.csv'))
            if -1 in fileObj.airy:
                fileObj.airy = None
            fileObj.active_analysis = read_text(os.path.join(filefolder, 'activeAnalysis.txt'), dtype=int)
            md = FFSmetadata()
            md.num_pixels = read_text(os.path.join(filefolder, 'numberOfPixels.txt'), dtype=int)
            md.num_lines = read_text(os.path.join(filefolder, 'numberOfLines.txt'), dtype=int)
            md.num_frames = read_text(os.path.join(filefolder, 'numberOfFrames.txt'), dtype=int)
            md.range_x = read_text(os.path.join(filefolder, 'rangeX.txt'), dtype=int)
            md.range_y = read_text(os.path.join(filefolder, 'rangeY.txt'), dtype=int)
            md.range_z = read_text(os.path.join(filefolder, 'rangeZ.txt'), dtype=int)
            md.num_datapoints = read_text(os.path.join(filefolder, 'numberOfDataPoints.txt'), dtype=int)
            md.hold_off_x5 = read_text(os.path.join(filefolder, 'holdOffx5.txt'), dtype=int)
            md.hold_off = read_text(os.path.join(filefolder, 'holdOff.txt'), dtype=int)
            md.time_resolution = read_text(os.path.join(filefolder, 'timeResolution.txt'), dtype=float)
            md.dwelltime = read_text(os.path.join(filefolder, 'dwellTime.txt'), dtype=float)
            md.duration = read_text(os.path.join(filefolder, 'duration.txt'), dtype=float)
            md.pxsize = read_text(os.path.join(filefolder, 'pxsize.txt'), dtype=float)
            yc = 0
            xc = 0
            coords = check_none(csv2array(os.path.join(filefolder, 'coords.csv')))
            
            if coords is not None:
                yc = int(coords[0])
                xc = int(coords[1])
            md.coords = [yc, xc]
            fileObj.metadata = md
            imageObj.ffs_list.append(fileObj)
            # check for analysis in file
            fileObj = imageObj.get_ffs_file(j)
            analyses = next(os.walk(filefolder))[1]
            Nanalyses = len(analyses)
            for k in range(Nanalyses):
                if root != 0:
                    root.progressMessage = 'Reading correlation analysis ' + str(k+1) + ', FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                analysisfolder = os.path.join(filefolder, 'analysis' + str(k))
                
                corrObj = FFScorr()
                
                corrObj.mode = read_text(os.path.join(analysisfolder, 'mode.txt'))
                corrObj.active_fit = read_text(os.path.join(analysisfolder, 'activeFit.txt'), dtype=int)
                corrSett = CorrSettings()
                
                elString = read_text(os.path.join(analysisfolder, 'elements.txt'))
                elList = elString.split(", ")
                elList[0]=elList[0][1:]
                elList[-1]=elList[-1][0:-1]
                elListOut = []
                for l in range(len(elList)):
                    if elList[l][0] == "'":
                        elListOut.append(elList[l][1:-1])
                    else:
                        elListOut.append(int(elList[l]))
                corrSett.elements = elListOut
                
                gString = read_text(os.path.join(analysisfolder, 'listOfG.txt'))
                gList = gString.split(", ")
                gList[0]=gList[0][1:]
                gList[-1]=gList[-1][0:-1]
                gListOut = []
                for l in range(len(gList)):
                    if gList[l][0] == "'":
                        gListOut.append(gList[l][1:-1])
                    else:
                        gListOut.append(int(gList[l]))
                corrSett.list_of_g = gListOut
                
                corrSett.average = read_text(os.path.join(analysisfolder, 'average.txt'), dtype=str)
                corrSett.algorithm = read_text(os.path.join(analysisfolder, 'algorithm.txt'), dtype=str)
                corrSett.resolution = read_text(os.path.join(analysisfolder, 'resolution.txt'), dtype=int)
                corrSett.chunksize = read_text(os.path.join(analysisfolder, 'chunksize.txt'), dtype=float)
                corrSett.chunks_off = [int(l) for l in csv2array(os.path.join(analysisfolder, 'chunksOff.csv'))]
                corrObj.settings = corrSett
                # get all .csv files with correlations
                list_filesAll = os.listdir(analysisfolder)
                corrFiles = []
                for l in list_filesAll:
                    if l == 'chunksOff.csv' or l =='fits' or l[-4:] == '.txt':
                        pass
                    else:
                        corrFiles.append(l)
                corrs = Correlations()
                for l in corrFiles:
                    setattr(corrs, l[0:-4], csv2array(os.path.join(analysisfolder, l)))
                corrs.dwellTime = -1
                if len(corrFiles) > 0:
                    corrObj.corrs = corrs
                fileObj.analysis_list.append(corrObj)
                # check overall fit
                anObj = fileObj.get_analysis(k)
                fitfolder = os.path.join(analysisfolder, 'fits')
                fits = next(os.walk(fitfolder))[1]
                Nfits = len(fits)
                for l in range(Nfits):
                    cfitObj = CorrFit()
                    # check individual fits
                    fitsinglefolder = os.path.join(fitfolder, 'fit' + str(l))
                    singlefits = next(os.walk(fitsinglefolder))[1]
                    Nsinglefits = len(singlefits)
                    for m in range(Nsinglefits):
                        fitindvfolder = os.path.join(fitsinglefolder, 'fitSingleCurve' + str(m))
                        fdata = read_text(os.path.join(fitindvfolder, 'data.txt'))
                        ffitfunctionLabel = read_text(os.path.join(fitindvfolder, 'fitfunctionLabel.txt'))
                        ffitarray = np.array([int(i) for i in csv2array(os.path.join(fitindvfolder, 'fitarray.csv'))])
                        fstartvalues = csv2array(os.path.join(fitindvfolder, 'startvalues.csv'))
                        fparamFactors10 = csv2array(os.path.join(fitindvfolder, 'paramFactors10.csv'))
                        fitr = csv2array(os.path.join(fitindvfolder, 'fitrange.csv'))
                        fitstart = int(fitr[0])
                        fitstop = int(fitr[1])
                        ffitrange = [fitstart, fitstop]
                        indfitObj = FitSingleObj(fdata, FitModel(), ffitarray, fstartvalues, ffitrange)
                        indfitObj.data = fdata
                        indfitObj.fitfunction_label = ffitfunctionLabel
                        indfitObj.fitarray = ffitarray
                        indfitObj.startvalues = fstartvalues
                        indfitObj.param_factors10 = fparamFactors10
                        indfitObj.fitrange = ffitrange
                        indfitObj.D = read_text(os.path.join(fitindvfolder, 'D.txt'), float)
                        #indfitObj.fitfunction = read_text(os.path.join(fitindvfolder, 'fitfunction.txt'))
                        indfitObj.fitresult = csv2array(os.path.join(fitindvfolder, 'fitresult.csv'))
                        indfitObj.minbound = csv2array(os.path.join(fitindvfolder, 'minbound.csv'))
                        indfitObj.maxbound = csv2array(os.path.join(fitindvfolder, 'maxbound.csv'))
                        indfitObj.paramidx = [int(i) for i in csv2array(os.path.join(fitindvfolder, 'paramidx.csv'))]
                        
                        indfitObj.w0 = read_text(os.path.join(fitindvfolder, 'w0.txt'), float)
                        cfitObj.fit_all_curves.append(indfitObj)
                    anObj.fits.append(cfitObj)
    if root != 0:
        root.progressMessage = 'Almost there...'
    shutil.rmtree(folder)
    return lib

def restorelib_ffs(libfile, root=0):
    # restore session from .ffs lib file (h5)
    # unzip file
    if root != 0:
        root.progress = 0
        root.progressMessage = 'Opening file...'
    
    # create session library
    lib = FFSlib()
    with h5py.File(libfile) as f:
        lib.version = check_none(f['info'].attrs['version'])
        lib.notes = check_none(f['info'].attrs['notes'])
        
        active_im_str = 'active_image'
        if lib.version == '1.0':
            active_im_str = 'activeImage'
        lib.active_image = check_none(f['info'].attrs[active_im_str], dtype=int)
        
        if lib.version != '1.0':
            lib.date_created = check_none(f['info'].attrs['date_created'], dtype=str)
            lib.date_modified = check_none(f['info'].attrs['date_modified'], dtype=str)
        
        # walk through images
        imgList = [img for img in list(f.keys()) if img[0:5]=='image']
        Nimages = len(imgList)
        
        for i in range(Nimages):
            if root != 0:
                root.progress = i / Nimages
                root.progressMessage = 'Reading image ' + str(i+1) + '...'
            
            image = f['image' + str(i) + '/image'][:]
            
            im_name_str = 'image_name'
            if lib.version == '1.0':
                im_name_str = 'imageName'
            fname = f['image' + str(i)].attrs[im_name_str]
            
            # add image
            lib.add_image(image, fname)
            imageObj = lib.get_image(i)
            
            # active file
            active_file_str = 'active_ffs'
            if lib.version == '1.0':
                active_file_str = 'activeFFS'
            imageObj.active_ffs = check_none(f['image' + str(i)].attrs[active_file_str], dtype=int)
            
            # check for files in image
            fileList = list(f['image' + str(i)].keys())
            files = [file for file in fileList if file[0:4] == 'file']
            Nfiles = len(files)
            for j in range(Nfiles):
                currFile = 'image' + str(i) + '/file' + str(j)
                if root != 0:
                    root.progress = i / Nimages + j / (Nfiles * Nimages)
                    root.progressMessage = 'Reading FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                fileObj = FFSfile()
                
                args_str_new = ['fname', 'label', 'active_analysis']
                args_str_old = args_str_new
                if lib.version == '1.0':
                    args_str_old = ['fname', 'label', 'activeAnalysis']
                for k in range(len(args_str_new)):
                    setattr(fileObj, args_str_new[k], check_none(f[currFile + '/'].attrs[args_str_old[k]]))
                
                for arg in ['timetrace', 'airy']:
                    dummy = check_none(f[currFile + '/' + arg])
                    if dummy is not None:
                        dummy = np.array(dummy)
                    setattr(fileObj, arg, dummy)
                
                if -1 in fileObj.timetrace:
                    fileObj.timetrace = None
                
                if -1 in fileObj.airy:
                    fileObj.airy = None
                
                md = FFSmetadata()
                
                args_str_new = ['num_pixels', 'num_lines', 'num_frames', 'range_x', 'range_y', 'range_z', 'num_datapoints', 'hold_off_x5', 'hold_off']
                args_str_old = args_str_new
                if lib.version == '1.0':
                    args_str_old = ['numberOfPixels', 'numberOfLines', 'numberOfFrames', 'rangeX', 'rangeY', 'rangeZ', 'numberOfDataPoints', 'holdOffx5', 'holdOff']
                for k in range(len(args_str_new)):
                    setattr(md, args_str_new[k], check_none(f[currFile].attrs[args_str_old[k]], dtype=int))
                
                args_str_new = ['time_resolution', 'dwelltime', 'duration', 'pxsize']
                args_str_old = args_str_new
                if lib.version == '1.0':
                    args_str_old = ['timeResolution', 'dwellTime', 'duration', 'pxsize']                
                for k in range(len(args_str_new)):
                    setattr(md, args_str_new[k], check_none(f[currFile].attrs[args_str_old[k]], dtype=float))
                
                coords = check_none(f[currFile + '/coords'][:])
                if coords is not None:
                    coords = coords.astype(int)
                md.coords = coords
                fileObj.metadata = md
                imageObj.ffs_list.append(fileObj)
                
                # check for analysis in file
                fileObj = imageObj.get_ffs_file(j)
                
                analList = list(f[currFile].keys())
                analyses = [anal for anal in analList if anal[0:8] == 'analysis']
                Nanalyses = len(analyses)
                for k in range(Nanalyses):
                    if root != 0:
                        root.progressMessage = 'Reading correlation analysis ' + str(k+1) + ', FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                    currAnal = currFile + '/analysis' + str(k)
                    
                    corrObj = FFScorr()
                    corrObj.mode = check_none(f[currAnal].attrs['mode'], dtype=str)
                    
                    active_fit_str = 'activeFit' if lib.version == '1.0' else 'active_fit'
                    corrObj.active_fit = check_none(f[currAnal].attrs[active_fit_str], dtype=int)
                    corrSett = CorrSettings()
                    
                    elList = check_none(f[currAnal].attrs['elements'])
                    elListOut = []
                    for l in range(len(elList)):
                        elListOut.append(elList[l])
                    
                    glist_str = 'listOfG' if lib.version == '1.0' else 'list_of_g'
                    gList = check_none(f[currAnal].attrs[glist_str])
                   
                    try:
                        gListOut = []
                        for l in range(len(gList)):
                            gListOut.append(gList[l])
                    except:
                        gListOut = None
                    
                    
                    corrSett.elements = elListOut
                    corrSett.list_of_g = gListOut
                    corrSett.average = getattr(f[currAnal].attrs, 'average', None)
                    corrSett.algorithm = f[currAnal].attrs['algorithm']
                    corrSett.resolution = check_none(f[currAnal].attrs['resolution'], dtype=int)
                    corrSett.chunksize = check_none(f[currAnal].attrs['chunksize'], dtype=float)
                    
                    chunksoff_str = 'chunksOff' if lib.version == '1.0' else 'chunks_off'
                    corrSett.chunks_off = check_none(f[currAnal][chunksoff_str][:], dtype=None)
                    
                    corrObj.settings = corrSett
                    
                    Ncorrs = 0
                    corrs = Correlations()
                    corrs.dwellTime = -1 # not used
                    for corr in f[currAnal].keys():
                        if corr not in ['fits']:
                            Ncorrs += 1
                            setattr(corrs, corr, f[currAnal + '/' + corr][:])
                    
                    if Ncorrs > 1:
                        # if only 1 element found, this element is chunksoff and not used because already stored elsewhere
                        corrObj.corrs = corrs
                    
                    fileObj.analysis_list.append(corrObj)
                    
                    # check overall fit
                    anObj = fileObj.get_analysis(k)
                    fitFolder = 'image' + str(i) + '/file' + str(j) + '/analysis' + str(k) + '/fits'
                    
                    fitList = list(f[fitFolder].keys())
                    Nfits = len(fitList)
                    
                    for l in range(Nfits):
                        cfitObj = CorrFit()
                        
                        if root != 0:
                            root.progressMessage = 'Loading fit ' + str(l+1) + ', analysis' + str(k+1) +  ', FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                    
                        fitAllCurves5 = fitFolder + '/fit' + str(l)
                        
                        Nfits3 = len(f[fitAllCurves5].keys())
                        for m in range(Nfits3):
                            if root != 0:
                                root.progressMessage = 'Loading fit--- ' + str(m) +  ', FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                            
                            fitsingle_str = '/fitSingleCurve' if lib.version == '1.0' else '/fit_single_curve'
                            fitSingleCurveFolder = fitAllCurves5 + fitsingle_str + str(m)
                            fitSingleCurve = f[fitSingleCurveFolder]
                            
                            ffitarray = fitSingleCurve['fitarray'][:]
                            ffitresult = fitSingleCurve['fitresult'][:]
                            fmaxbound = fitSingleCurve['maxbound'][:]
                            fminbound = fitSingleCurve['minbound'][:]
                            fparamidx = fitSingleCurve['paramidx'][:]
                            fstartvalues = fitSingleCurve['startvalues'][:]
                            
                            fparam_str = 'paramFactors10' if lib.version == '1.0' else 'param_factors10'
                            fparamFactors10 = fitSingleCurve[fparam_str][:]
                            
                            fitf_str = 'fitfunctionLabel' if lib.version == '1.0' else 'fitfunction_label'
                            ffitfunctionLabel = f[fitSingleCurveFolder].attrs[fitf_str]
                            
                            ffitrange = f[fitSingleCurveFolder].attrs['fitrange']
                            fD = f[fitSingleCurveFolder].attrs['D']
                            fw0 = check_none(f[fitSingleCurveFolder].attrs['w0'])
                            fdata = f[fitSingleCurveFolder].attrs['data']
                            
                            indfitObj = FitSingleObj(fdata, FitModel(), ffitarray, fstartvalues, ffitrange)
                            indfitObj.data = fdata
                            indfitObj.fitfunction_label = ffitfunctionLabel
                            indfitObj.fitarray = ffitarray
                            indfitObj.startvalues = fstartvalues
                            indfitObj.param_factors10 = fparamFactors10
                            indfitObj.fitrange = ffitrange
                            indfitObj.D = fD
                            indfitObj.fitresult = ffitresult
                            indfitObj.minbound = fminbound
                            indfitObj.maxbound = fmaxbound
                            indfitObj.paramidx = fparamidx
                            indfitObj.w0 = fw0
                            
                            cfitObj.fit_all_curves.append(indfitObj)
                          
                            
                        anObj.fits.append(cfitObj)
    if root != 0:
        root.progressMessage = 'Almost there...'
    
    return lib

def savelib(FFSlib, root=0, fname=''):
    if fname == '':
        fname = save_ffs()
    # save session in .ffs or .ffz lib file
    if str(fname).endswith('.ffz'):
        savelib_ffz(FFSlib, root, fname)
    else:
        # make sure the file ends with .ffs
        fname = checkfname(fname, 'ffs')
        savelib_ffs(FFSlib, root, fname)
    return fname

def savelib_ffz(FFSlib, root=0, fname=''):
    if fname == '':
        fname = save_ffs()
    if fname is not None:
        if root != 0:
            root.progress = 0
            root.progressMessage = 'Saving file...'
        lib = FFSlib.lib
        # make folder to store data temporarilyy
        randString = str(int(np.random.rand()*100000))
        folderName = "Facts_tempData_" + randString
        create_dir(folderName)
        save_text("1.1", folderName + '/version.txt')
        save_text(FFSlib.notes, folderName + '/notes.txt')
        save_text(FFSlib.active_image, folderName + '/activeImage.txt')
        save_text(FFSlib.date_created, folderName + '/date_created.txt')
        save_text(str(datetime.now()), folderName + '/date_modified.txt')
        Nimages = len(lib)
        for i in range(Nimages):
            if root != 0:
                root.progress = i / Nimages
                root.progressMessage = 'Saving image ' + str(i+1) + '...'
            im = lib[i]
            # image found
            imageFolder = folderName + "/image" + str(i)
            create_dir(imageFolder)
            # store image name
            save_text(im.image_name, imageFolder + '/imageName.txt')
            # store image
            array2csv(im.image, imageFolder + '/image.csv', dtype=int)
            array2csv(im.image, imageFolder + '/imageRaw.csv', dtype=int)
            # store active file
            save_text(im.active_ffs, imageFolder + '/activeFFS.txt')
            # to through ffs files
            Nfiles = len(im.ffs_list)
            for j in range(Nfiles):
                if root != 0:
                    root.progress = i / Nimages + j / (Nimages * Nfiles)
                    root.progressMessage = 'Saving FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                file = im.ffs_list[j]
                fileFolder = imageFolder + "/file" + str(j)
                create_dir(fileFolder)
                # save file name
                save_text(file.fname, fileFolder + '/fname.txt')
                # save label
                save_text(file.label, fileFolder + '/label.txt')
                # save time trace
                tt = file.timetrace
                if tt is None:
                    tt = np.zeros((1)) - 1
                array2csv(tt, fileFolder + '/timetrace.csv')
                # save finger print
                fp = file.airy
                if fp is None:
                    fp = np.zeros((1)) - 1
                array2csv(fp, fileFolder + '/airy.csv')
                # save active analysis
                save_text(file.active_analysis, fileFolder + '/activeAnalysis.txt')
                # save metadata
                md = file.metadata
                for key in list(md.__dict__.keys()):
                    if key != 'coords':
                        save_text(check_none(getattr(md, key)), fileFolder + '/' + key + '.txt')
                    else:
                        array2csv(check_none(getattr(md, key)), fileFolder + '/' + key + '.csv')
                # go through each analysis
                Nanalysis = len(file.analysis_list)
                for k in range(Nanalysis):
                    if root != 0:
                        root.progressMessage = 'Saving analysis ' + str(k) +  ', FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                    analysis = file.analysis_list[k]
                    analysisFolder = fileFolder + "/analysis" + str(k)
                    create_dir(analysisFolder)
                    # save analysis mode
                    save_text(analysis.mode, analysisFolder + '/mode.txt')
                    # save elements
                    save_text(analysis.settings.elements, analysisFolder + '/elements.txt')
                    # save g list
                    save_text(analysis.settings.list_of_g, analysisFolder + '/listOfG.txt')
                    # save average list
                    save_text(analysis.settings.average, analysisFolder + '/average.txt')
                    # save algorithm
                    save_text(analysis.settings.algorithm, analysisFolder + '/algorithm.txt')
                    # save resolution
                    save_text(analysis.settings.resolution, analysisFolder + '/resolution.txt')
                    # save chunck size
                    save_text(analysis.settings.chunksize, analysisFolder + '/chunksize.txt')
                    # save chunks off
                    array2csv(analysis.settings.chunks_off, analysisFolder + '/chunksOff.csv')
                    # save active fit
                    save_text(analysis.active_fit, analysisFolder + '/activeFit.txt')
                    # save correlations
                    c = analysis.corrs
                    if c is not None:
                        for key in list(c.__dict__.keys()):
                            if key != "dwellTime":
                                # DWELLTIME IS ALREADY STORED IN METADATA
                                array2csv(getattr(c, key), analysisFolder + '/' + key + '.csv')
                    # save fits
                    fitsFolder = analysisFolder + "/fits"
                    create_dir(fitsFolder)
                    fits = analysis.fits
                    Nfits = len(fits)
                    for l in range(Nfits):
                        fitAllCurvesFolder = fitsFolder + "/fit" + str(l)
                        create_dir(fitAllCurvesFolder)
                        fitAllCurves = fits[l].fit_all_curves # contains 3 fits: central, sum3, sum5
                        Nfits3 = len(fitAllCurves)
                        for m in range(Nfits3):
                            fitSingleCurveFolder = fitAllCurvesFolder + "/fitSingleCurve" + str(m)
                            create_dir(fitSingleCurveFolder)
                            fitSingleCurve = fitAllCurves[m]
                            # save all fields
                            array2csv(fitSingleCurve.minbound, fitSingleCurveFolder + '/minbound.csv')
                            array2csv(fitSingleCurve.maxbound, fitSingleCurveFolder + '/maxbound.csv')
                            save_text(fitSingleCurve.data, fitSingleCurveFolder + '/data.txt')
                            save_text(fitSingleCurve.fitfunction_label, fitSingleCurveFolder + '/fitfunctionLabel.txt')
                            array2csv(fitSingleCurve.fitrange, fitSingleCurveFolder + '/fitrange.csv')
                            array2csv(fitSingleCurve.fitresult, fitSingleCurveFolder + '/fitresult.csv')
                            #save_text(fitSingleCurve.fitfunction, fitSingleCurveFolder + '/fitfunction.txt')
                            array2csv(fitSingleCurve.fitarray, fitSingleCurveFolder + '/fitarray.csv')
                            array2csv(fitSingleCurve.startvalues, fitSingleCurveFolder + '/startvalues.csv')
                            array2csv(fitSingleCurve.param_factors10, fitSingleCurveFolder + '/paramFactors10.csv')
                            array2csv(fitSingleCurve.paramidx, fitSingleCurveFolder + '/paramidx.csv')
                            save_text(check_none(fitSingleCurve.w0), fitSingleCurveFolder + '/w0.txt')
                            save_text(fitSingleCurve.D, fitSingleCurveFolder + '/D.txt')
        
        if root != 0:
            root.progressMessage = 'Almost there...'
        make_zip(fname, folderName)
        shutil.rmtree(folderName)
        

def savelib_ffs(FFSlib, root=0, fname=''):
    if fname == '':
        fname = save_ffs()

    if fname is not None:
        if root != 0:
            root.progress = 0
            root.progressMessage = 'Saving file...'
        
        with h5py.File(fname, "w") as f:
            # save version
            inf = f.create_group('info')                                                                   
            inf.attrs['version'] = check_none("1.1", dtype=str)
            inf.attrs['notes'] = check_none(FFSlib.notes, dtype=str)
            inf.attrs['active_image'] = check_none(FFSlib.active_image, dtype=int)
            inf.attrs['date_created'] = check_none(FFSlib.date_created, dtype=str)
            inf.attrs['date_modified'] = check_none(str(datetime.now()), dtype=str)
            
            lib = FFSlib.lib
            Nimages = len(lib)
            for i in range(Nimages):
                if root != 0:
                    root.progress = i / Nimages
                    root.progressMessage = 'Saving image ' + str(i+1) + '...'
                im = lib[i]
                
                # image found
                im5 = f.create_group("image" + str(i))
                im5.attrs["image_name"] = check_none(im.image_name, dtype=str)
                im5.create_dataset('image',data=im.image, compression="gzip")
                im5.create_dataset('image_raw',data=im.image, compression="gzip")
                
                # store active file
                im5.attrs["active_ffs"] = check_none(im.active_ffs, dtype=int)
                
                # go through ffs files
                Nfiles = len(im.ffs_list)
                for j in range(Nfiles):
                    if root != 0:
                        root.progress = i / Nimages + j / (Nimages * Nfiles)
                        root.progressMessage = 'Saving FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                    file = im.ffs_list[j]
                    
                    file5 = im5.create_group("file" + str(j))
                    file5.attrs["fname"] = check_none(file.fname, dtype=str)
                    file5.attrs["label"] = check_none(file.label, dtype=str)
                    
                    # save time trace
                    tt = file.timetrace
                    if tt is None:
                        tt = np.zeros((1)) - 1
                    file5.create_dataset('timetrace', data=tt, compression="gzip")
                    
                    # save finger print
                    fp = file.airy
                    if fp is None:
                        fp = np.zeros((1)) - 1
                    file5.create_dataset('airy', data=fp, compression="gzip")
                    
                    # save active analysis
                    file5.attrs["active_analysis"] = check_none(file.active_analysis)
                    
                    # save metadata
                    md = file.metadata
                    for key in list(md.__dict__.keys()):
                        if key != 'coords':
                            file5.attrs[key] = check_none(getattr(md, key))
                        else:
                            file5.create_dataset(key, data=getattr(md, key))
                        
                    # go through each analysis
                    Nanalysis = len(file.analysis_list)
                    for k in range(Nanalysis):
                        if root != 0:
                            root.progressMessage = 'Saving analysis ' + str(k+1) +  ', FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                        analysis = file.analysis_list[k]
                        
                        analysis5 = file5.create_group("analysis" + str(k))
                        
                        # save analysis mode
                        analysis5.attrs["mode"] = check_none(analysis.mode)
                        # save elements
                        analysis5.attrs["elements"] = check_none(analysis.settings.elements)
                        # save list of G
                        analysis5.attrs["list_of_g"] = check_none(analysis.settings.list_of_g)
                        # save average
                        analysis5.attrs["average"] = check_none(analysis.settings.average)
                        # save algorithm
                        analysis5.attrs["algorithm"] = check_none(analysis.settings.algorithm)
                        
                        # save resolution
                        analysis5.attrs["resolution"] = check_none(analysis.settings.resolution)
                        # save chunck size
                        analysis5.attrs["chunksize"] = check_none(analysis.settings.chunksize)
                        # save chunks off
                        analysis5.create_dataset("chunks_off",data=analysis.settings.chunks_off)
                        
                        # save active fit
                        analysis5.attrs["active_fit"] = check_none(analysis.active_fit)
                        # save correlations
                        c = analysis.corrs
                        if c is not None:
                            for key in list(c.__dict__.keys()):
                                if key != "dwellTime" and key != "chunks_off":
                                    # DWELLTIME IS ALREADY STORED IN METADATA
                                    analysis5.create_dataset(key, data=getattr(c, key))
        
                        # save fits
                        root.progressMessage = 'Saving fit ' + str(k+1) +  ', FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                        fits5 = analysis5.create_group("fits")
                        fits = analysis.fits
                        Nfits = len(fits)
                        
                        root.progressMessage = 'Saving fits ' + str(k+1) +  ', FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                        for l in range(Nfits):
                            fitAllCurves5 = fits5.create_group("fit" + str(l))
                            fitAllCurves = fits[l].fit_all_curves # contains 3 fits: central, sum3, sum5
                            Nfits3 = len(fitAllCurves)
                            for m in range(Nfits3):
                                root.progressMessage = 'Saving fit--- ' + str(m) +  ', FFS file ' + str(j+1) + ', image ' + str(i+1) + '...'
                                fitSingleCurve5 = fitAllCurves5.create_group("fit_single_curve" + str(m))
                                fitSingleCurve = fitAllCurves[m]
                                # save all fields
                                fitSingleCurve5.create_dataset('minbound',data=fitSingleCurve.minbound)
                                fitSingleCurve5.create_dataset('maxbound',data=fitSingleCurve.maxbound)
                                fitSingleCurve5.create_dataset('fitresult',data=fitSingleCurve.fitresult)
                                fitSingleCurve5.create_dataset('fitarray',data=fitSingleCurve.fitarray)
                                fitSingleCurve5.create_dataset('startvalues',data=fitSingleCurve.startvalues)
                                fitSingleCurve5.create_dataset('param_factors10',data=fitSingleCurve.param_factors10)
                                fitSingleCurve5.create_dataset('paramidx',data=fitSingleCurve.paramidx)
                                fitSingleCurve5.attrs['data'] = check_none(fitSingleCurve.data)
                                fitSingleCurve5.attrs['fitfunction_label'] = check_none(fitSingleCurve.fitfunction_label)
                                fitSingleCurve5.attrs['fitrange'] = check_none(fitSingleCurve.fitrange)
                                fitSingleCurve5.attrs['w0'] = check_none(fitSingleCurve.w0)
                                check_none(fitSingleCurve.D)
                                fD = fitSingleCurve.D
                                if fD is None:
                                    fD = 'None'
                                fitSingleCurve5.attrs['D'] = fD
                                
                                
            if root != 0:
                root.progressMessage = 'Almost there...'

def save_ffs(window_title='Save project as', ftype='FFS files (*.ffs)', directory=''):
    """
    Select name to save ffs lib
    ===========================================================================
    Input       Meaning
    ---------------------------------------------------------------------------
    dialog window is opened to choose file name
    ===========================================================================
    Output      Meaning
    ---------------------------------------------------------------------------
    the file name is returned
    ===========================================================================
    """
    
    fname, _ = QFileDialog.getSaveFileName(None, window_title, directory, ftype)
    
    return fname if fname else None

def make_zip(fname, zipfolder):
    if fname[-4:] != '.ffz':
        fname = fname + '.ffz'
    # make zip
    shutil.make_archive('tempfile', 'zip', zipfolder)
    # store zip as ffz
    try:
        os.rename('tempfile.zip', 'tempfile.ffz')
    except:
        os.remove('tempfile.ffz')
        os.rename('tempfile.zip', 'tempfile.ffz')
    # move ffs file do desired location
    try:
        shutil.move('tempfile.ffz', fname)
    except:
        os.remove(fname)
        shutil.move('tempfile.ffz', fname)

def unzip(fname):
    originalFilename = fname
    if originalFilename[-4:] == '.ffz':
        fname = originalFilename[:-4] + '.zip'
        os.rename(originalFilename, fname)
    randString = str(int(np.random.rand()*100000))
    folderName = "Facts_tempData_" + randString
    create_dir(folderName)
    shutil.unpack_archive(fname, folderName)
    if originalFilename[-4:] == '.ffz':
        os.rename(fname, originalFilename)
    return folderName

def create_dir(path):
    os.mkdir(path)

def check_none(inpt, dtype=None):
    if dtype is not None and inpt is not None and inpt != 'None':
        inpt = dtype(inpt)
    if inpt is None:
        inpt = 'None'
    elif type(inpt) is str and inpt == 'None':
        inpt = None
    return inpt

def to_text(text):
    if text is None:
        text = "None"
    text = str(text)
    return text

def save_image(image, fname):
    if image is not None:
        array2csv(image, fname, dtype=int)

def save_text(text, fname):
    text = to_text(text)
    with open(fname, "w") as text_file:
        text_file.write(text)

def read_text(fname, dtype=str):    
    try:
        with open(fname, "r") as text_file:
            text = text_file.read()
            if text == 'None':
                text = None
            else:
                text = dtype(text)
    except:
        text = None
    return text