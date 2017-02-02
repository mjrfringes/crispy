from tools.locate_psflets import locatePSFlets,PSFLets
from tools.image import Image
from tools.par_utils import Task, Consumer
from IFS import propagateIFS
from photutils import CircularAperture
import matplotlib as mpl
import numpy as np
from scipy import signal
import logging as log
from astropy.io import fits as pyf
import logging as log
import os
import time
import multiprocessing
from scipy import ndimage
import matplotlib.pyplot as plt



def createWavecalFiles(par,lamlist):
    '''
    Creates a set of monochromatic IFS images to be used in wavelength calibration step
    
    '''
    
    par.saveDetector=False
    lamoD = 3. # number of lenslets per lamoD at 660nm
    mperpix = par.pitch/lamoD
    par.pixperlenslet = par.pitch/mperpix
    par.mperpix = mperpix
    inputCube = np.ones((1,512,512),dtype=float)/lamoD**2
    filelist = []
    for wav in lamlist:
        detectorFrame = propagateIFS(par,[wav*1e-3],inputCube)
        filename = par.wavecalDir+'det_%3d.fits' % (wav)
        filelist.append(filename)
        Image(data=detectorFrame).write(filename)
    par.lamlist = lamlist
    par.filelist = filelist




def computeWavecal(par,lamlist=None,filelist=None,order = 3):
    '''
    Computes a wavelength calibration from a set of fits files
    Uses Tim Brandt's locate_PSFlets routine with an initial guess.
    The consecutive solutions use information from previous solutions to improve the
    initial guess.
    The process can be optimized somewhat by reducing the number of lenslets within
    the locatePSFlets function (could make this a parameter!)
    lamlist and filelist can be defined in the parameters in which case they don't need to be set
    
    Inputs
    1. par          Parameter instance with at least the key IFS parameters, interlacing and scale
    2. lamlist      list of wavelengths in nm
    3. filelist     list of fits files to open in corresponding order
    4. order        order of 2d polynomial to be fitted to lenslets
    
    '''
    # lamlist/filelist override
    if (filelist!=None) and (lamlist!=None):
        par.filelist = filelist
        par.lamlist = lamlist
    coef = None
    allcoef = []    
    xpos = []
    ypos = []
    good = []
#     polyimage = np.zeros((len(par.lamlist), par.npix, par.npix))

    for i in range(len(par.lamlist)):
        inImage = Image(filename=par.filelist[i])
        _x, _y, _good, coef = locatePSFlets(inImage, polyorder=order, coef=coef,phi=par.philens,scale=par.pitch/par.pixsize,nlens=par.nlens)
#         polyimage[i] = inImage.data
        xpos += [_x]
        ypos += [_y]
        good += [_good]
        allcoef += [[par.lamlist[i]] + list(coef)]
    log.info("Saving wavelength solution to " + par.wavecalDir + "lamsol.dat")
    outkey = pyf.HDUList(pyf.PrimaryHDU(np.asarray(par.lamlist)))
    outkey.append(pyf.PrimaryHDU(np.asarray(xpos)))
    outkey.append(pyf.PrimaryHDU(np.asarray(ypos)))
    outkey.append(pyf.PrimaryHDU(np.asarray(good).astype(np.uint8)))
    outkey.writeto(par.wavecalDir + par.wavecalName, clobber=True)
    
#     out = pyf.HDUList(pyf.PrimaryHDU(polyimage.astype(np.float32)))
#     out.writeto(par.wavecalDir + 'polychromeR%d.fits' % (par.R), clobber=True)

    allcoef = np.asarray(allcoef)
    np.savetxt(par.wavecalDir + "lamsol.dat", allcoef)

def createPolychrome(par):
    '''
    To be run after generating a wavelength calibration set.
    This function constructs a cube of (lam_max-lam_min)/par.dlam depth,
    in which each slice is a monochromatic map at that wavelength.
    If the wavelength calibration doesn't contain all the required wavelengths,
    interpolate between wavelengths using map_coordinates.
    In the end, one should be able to make a cutout at a given location and get a cube
    with all the psflets for that lenslet.
    
    For now, this only works with simulated, noiseless data in order to get
    almost perfect PSFs. In practice, we will have to create this cube from data.
    For now, already assumes that all wavelengths are available (need to compute these)
    '''
    
    polyimage = np.zeros((len(par.lamlist), par.npix, par.npix))

    # simply put all the ideal images into a cube (will need to develop this further)
    for i in range(len(par.lamlist)):
        inImage = Image(filename=par.filelist[i])
        polyimage[i] = inImage.data
    out = pyf.HDUList(pyf.PrimaryHDU(polyimage.astype(np.float32)))
    out.writeto(par.wavecalDir + 'polychromeR%d.fits' % (par.R), clobber=True)

def inspectWaveCal(par,slice=0,name='inspectWavecal'):
    '''
    Inspects a wavecal solution by overplotting apertures on top of the image
    
    Inputs:
    1. par          Parameter instance
    2. slice        which slice of the wavelength calibration to look at
    3. save         whether to save the figure or not
    
    '''
    calCube = pyf.open(par.wavecalDir+'polychromekeyR%d.fits' % (par.R))
    waveCalArray = calCube[0].data
    i = slice
    log.info('Displaying PSF centroid fits for wavelength %3d' % (waveCalArray[i]))
    xorig = calCube[1].data[i,:,:]
    yorig = calCube[2].data[i,:,:]
    xdim,ydim = yorig.shape

    xg,yg = xorig.shape
    vals = np.array([(xorig[m,n],yorig[m,n]) for m in range(xg) for n in range(yg)])
    pos = (vals[:,0],vals[:,1])
    aps = CircularAperture(pos, r=3)
    fig,ax = plt.subplots(figsize=(15,15))
    hdulist = pyf.open(par.filelist[i],ignore_missing_end=True)
    if hdulist[0].header['NAXIS']!=2:
        image = pyf.open(par.filelist[i],ignore_missing_end=True)[1].data
    else:
        image = pyf.open(par.filelist[i],ignore_missing_end=True)[0].data
    mean = np.mean(image)
    std = np.std(image)
    norm = mpl.colors.Normalize(vmin=mean,vmax=mean+5*std)
    ax.imshow(image, cmap='Greys',norm=norm,interpolation='nearest',origin='lower')
    aps.plot(ax=ax,color='blue', lw=1, alpha=0.5)
    fig.savefig(par.wavecalDir+name+'_%3d.png' % (waveCalArray[i]),dpi=300)


def do_inspection(par,image,xpos,ypos,lam):
    
    xg,yg = xpos.shape
    vals = np.array([(xpos[m,n],ypos[m,n]) for m in range(xg) for n in range(yg)])
    pos = (vals[:,0],vals[:,1])
    aps = CircularAperture(pos, r=3)
    fig,ax = plt.subplots(figsize=(15,15))
    mean = np.mean(image)
    std = np.std(image)
    norm = mpl.colors.Normalize(vmin=mean,vmax=mean+5*std)
    ax.imshow(image, cmap='Greys',norm=norm,interpolation='nearest',origin='lower')
    aps.plot(ax=ax,color='blue', lw=1, alpha=0.5)
    fig.savefig(par.wavecalDir+'inspection_%3d.png' % (lam),dpi=300)
    
# def inspectAllWavecal(par,slice_list = None):
#     
#     
#     # load up polychrome key cube
#     polychromekey = pyf.open(par.wavecalDir+'polychromekeyR%d.fits' % (par.R))
#     if slice_list is None:
        
    

def make_polychrome(lam1, lam2, hires_arrs, lam_arr, psftool, allcoef,
                     xindx, yindx, ydim,xdim, upsample=5, nlam=10):
    """
    """

    padding = 10
    image = np.zeros((ydim + 2*padding, xdim + 2*padding))
    x = np.arange(image.shape[0])
    x, y = np.meshgrid(x, x)
    npix = hires_arrs[0].shape[2]//upsample

    dloglam = (np.log(lam2) - np.log(lam1))/nlam
    loglam = np.log(lam1) + dloglam/2. + np.arange(nlam)*dloglam

    for lam in np.exp(loglam):

        ################################################################
        # Build the appropriate average hires image by averaging over
        # the nearest wavelengths.  Then apply a spline filter to the
        # interpolated high resolution PSFlet images to avoid having
        # to do this later, saving a factor of a few in time.
        ################################################################

        hires = np.zeros((hires_arrs[0].shape))
        if lam <= np.amin(lam_arr):
            hires[:] = hires_arrs[0]
        elif lam >= np.amax(lam_arr):
            hires[:] = hires_arrs[-1]
        else:
            i1 = np.amax(np.arange(len(lam_arr))[np.where(lam > lam_arr)])
            i2 = i1 + 1
            hires = hires_arrs[i1]*(lam - lam_arr[i1])/(lam_arr[i2] - lam_arr[i1])
            hires += hires_arrs[i2]*(lam_arr[i2] - lam)/(lam_arr[i2] - lam_arr[i1])

        for i in range(hires.shape[0]):
            for j in range(hires.shape[1]):
                hires[i, j] = ndimage.spline_filter(hires[i, j])

        ################################################################
        # Run through lenslet centroids at this wavelength using the
        # fitted coefficients in psftool to get the centroids.  For
        # each centroid, compute the weights for the four nearest
        # regions on which the high-resolution PSFlets have been made.
        # Interpolate the high-resolution PSFlets and take their
        # weighted average, adding this to the image in the
        # appropriate place.
        ################################################################

        xcen, ycen = psftool.return_locations(lam, allcoef, xindx, yindx)
        xcen += padding
        ycen += padding
        xcen = np.reshape(xcen, -1)
        ycen = np.reshape(ycen, -1)
        for i in range(xcen.shape[0]):
            if not (xcen[i] > npix//2 and xcen[i] < image.shape[0] - npix//2 and 
                    ycen[i] > npix//2 and ycen[i] < image.shape[0] - npix//2):
                continue
                
            # central pixel -> npix*upsample//2
            iy1 = int(ycen[i]) - npix//2
            iy2 = iy1 + npix
            ix1 = int(xcen[i]) - npix//2
            ix2 = ix1 + npix
            yinterp = (y[iy1:iy2, ix1:ix2] - ycen[i])*upsample + upsample*npix/2
            xinterp = (x[iy1:iy2, ix1:ix2] - xcen[i])*upsample + upsample*npix/2
            # Now find the closest high-resolution PSFs
            
            x_hires = xcen[i]*1./image.shape[1]
            y_hires = ycen[i]*1./image.shape[0]
            
            x_hires = x_hires*hires_arrs[0].shape[1] - 0.5
            y_hires = y_hires*hires_arrs[0].shape[0] - 0.5
            
            totweight = 0
            
            if x_hires <= 0:
                i1 = i2 = 0
            elif x_hires >= hires_arrs[0].shape[1] - 1:
                i1 = i2 = hires_arrs[0].shape[1] - 1
            else:
                i1 = int(x_hires)
                i2 = i1 + 1

            if y_hires < 0:
                j1 = j2 = 0
            elif y_hires >= hires_arrs[0].shape[0] - 1:
                j1 = j2 = hires_arrs[0].shape[0] - 1
            else:
                j1 = int(y_hires)
                j2 = j1 + 1
            
            ##############################################################
            # Bilinear interpolation by hand.  Do not extrapolate, but
            # instead use the nearest PSFlet near the edge of the
            # image.  The outer regions will therefore have slightly
            # less reliable PSFlet reconstructions.  Then take the
            # weighted average of the interpolated PSFlets.
            ##############################################################

            weight22 = max(0, (x_hires - i1)*(y_hires - j1))
            weight12 = max(0, (x_hires - i1)*(j2 - y_hires))
            weight21 = max(0, (i2 - x_hires)*(y_hires - j1))
            weight11 = max(0, (i2 - x_hires)*(j2 - y_hires))
            totweight = weight11 + weight21 + weight12 + weight22
            weight11 /= totweight*nlam
            weight12 /= totweight*nlam
            weight21 /= totweight*nlam
            weight22 /= totweight*nlam

            image[iy1:iy2, ix1:ix2] += weight11*ndimage.map_coordinates(hires[j1, i1], [yinterp, xinterp], prefilter=False)
            image[iy1:iy2, ix1:ix2] += weight12*ndimage.map_coordinates(hires[j1, i2], [yinterp, xinterp], prefilter=False)
            image[iy1:iy2, ix1:ix2] += weight21*ndimage.map_coordinates(hires[j2, i1], [yinterp, xinterp], prefilter=False)
            image[iy1:iy2, ix1:ix2] += weight22*ndimage.map_coordinates(hires[j2, i2], [yinterp, xinterp], prefilter=False)
     
    image = image[padding:-padding, padding:-padding]
    return image



def gethires(x, y, image, upsample=5, nsubarr=5, npix=13, renorm=True):
    """
    Build high resolution images of the undersampled PSF using the
    monochromatic frames.

    Inputs:
    1. 
    """

    ###################################################################
    # hires_arr has nsubarr x nsubarr high-resolution PSFlets.  Smooth
    # out the result very slightly to reduce the impact of poorly
    # sampled points.  The resolution on these images, which will be
    # passed to a multidimensional spline interpolator, is a factor of
    # upsample higher than the pixellation of the original image.
    ###################################################################

    hires_arr = np.zeros((nsubarr, nsubarr, upsample*(npix + 1), upsample*(npix + 1)))
    _x = np.arange(3*upsample) - (3*upsample - 1)/2.
    _x, _y = np.meshgrid(_x, _x)
    r2 = _x**2 + _y**2
    window = np.exp(-r2/(2*0.3**2*(upsample/5.)**2))
    
    ###################################################################
    # yreg and xreg denote the regions of the image.  Each region will
    # have roughly 20,000/nsubarr**2 PSFlets from which to construct
    # the resampled version.  For 5x5 (default), this is roughly 800.
    ###################################################################   

    for yreg in range(nsubarr):
        i1 = yreg*image.data.shape[0]//nsubarr
        i2 = i1 + image.data.shape[0]//nsubarr
        i1 = max(i1, npix)
        i2 = min(i2, image.data.shape[0] - npix)

        for xreg in range(nsubarr):
            j1 = xreg*image.data.shape[1]//nsubarr
            j2 = j1 + image.data.shape[1]//nsubarr
            j1 = max(j1, npix)
            j2 = min(j2, image.data.shape[1] - npix)

            ############################################################
            # subim holds the high-resolution images.  The first
            # dimension counts over PSFlet, and must hold roughly the
            # total number of PSFlets divided by upsample**2.  The
            # worst possible case is about 20,000/nsubarr**2.
            ############################################################

            k = 0
            subim = np.zeros((20000/nsubarr**2, upsample*(npix + 1), upsample*(npix + 1)))

            ############################################################
            # Now put the PSFlets in.  The pixel of index
            # [npix*upsample//2, npix*upsample//2] is the centroid.
            # The counter k keeps track of how many PSFlets contribute
            # to each resolution element.
            ############################################################

            for i in range(x.shape[0]):
                if x[i] > j1 and x[i] < j2 and y[i] > i1 and y[i] < i2:
                    xval = x[i] - 0.5/upsample
                    yval = y[i] - 0.5/upsample

                    ix = (1 + int(xval) - xval)*upsample
                    iy = (1 + int(yval) - yval)*upsample

                    if ix == upsample:
                        ix -= upsample
                    if iy == upsample:
                        iy -= upsample

                    iy1, ix1 = [int(yval) - npix//2, int(xval) - npix//2]
                    cutout = image.data[iy1:iy1 + npix + 1, ix1:ix1 + npix + 1]
                    subim[k, iy::upsample, ix::upsample] = cutout
                    k += 1

            meanpsf = np.zeros((upsample*(npix + 1), upsample*(npix + 1)))
            weight = np.zeros((upsample*(npix + 1), upsample*(npix + 1)))

            ############################################################
            # Take the trimmed mean (middle 60% of the data) for each
            # PSFlet to avoid contamination by bad pixels.  Then
            # convolve with a narrow Gaussian to mitigate the effects
            # of poor sampling.
            ############################################################

            for ii in range(3):

                window1 = np.exp(-r2/(2*1**2*(upsample/5.)**2))
                window2 = np.exp(-r2/(2*1**2*(upsample/5.)**2))
                if ii < 2:
                    window = window2
                else:
                    window = window1                    

                if ii > 0:
                    for kk in range(k):
                        mask = 1.*(subim[kk] != 0)
                        if np.sum(mask) > 0:
                            A = np.sum(subim[kk]*meanpsf*mask)
                            A /= np.sum(meanpsf**2*mask)

                            if A > 0.5 and A < 2:
                                subim[kk] /= A
                            else:
                                subim[kk] = 0

                            chisq = np.sum(mask*(meanpsf - subim[kk])**2)
                            chisq /= np.amax(meanpsf)**2

                            subim[kk] *= (chisq < 1e-2*upsample**2)
                            #mask2 = np.abs(meanpsf - subim[kk])/(np.abs(meanpsf) + 0.01*np.amax(meanpsf)) < 1
                            #subim[kk] *= mask2
                            subim[kk] *= subim[kk] > -1e-3*np.amax(meanpsf)

                subim2 = subim.copy()
                for i in range(subim.shape[1]):
                    for j in range(subim.shape[2]):

                        _i1 = max(i - upsample//4, 0)
                        _i2 = min(i + upsample//4 + 1, subim.shape[1] - 1)
                        _j1 = max(j - upsample//4, 0)
                        _j2 = min(j + upsample//4 + 1, subim.shape[2] - 1)
                        
                        data = subim2[:k, _i1:_i2, _j1:_j2][np.where(subim2[:k, _i1:_i2, _j1:_j2] != 0)]
                        if data.shape[0] > 10:
                            data = np.sort(data)[3:-3]
                            std = np.std(data) + 1e-10
                            mean = np.mean(data)
                        
                            subim[:k, i, j] *= np.abs(subim[:k, i, j] - mean)/std < 3.5
                        elif data.shape[0] > 5:
                            data = np.sort(data)[1:-1]
                            std = np.std(data) + 1e-10
                            mean = np.mean(data)
                        
                            subim[:k, i, j] *= np.abs(subim[:k, i, j] - mean)/std < 3.5
                        
                        data = subim[:k, i, j][np.where(subim[:k, i, j] != 0)]
                        #data = np.sort(data)
                        npts = data.shape[0]
                        if npts > 0:
                            meanpsf[i, j] = np.mean(data)
                            weight[i, j] = npts

                meanpsf = signal.convolve2d(meanpsf*weight, window, mode='same')
                meanpsf /= signal.convolve2d(weight, window, mode='same')

                val = meanpsf.copy()
                for jj in range(10):
                    tmp = val/signal.convolve2d(meanpsf, window, mode='same')
                    meanpsf *= signal.convolve2d(tmp, window[::-1, ::-1], mode='same')
                    
            
            ############################################################
            # Normalize all PSFs to unit flux when resampled with an
            # interpolator.
            ############################################################

            if renorm:
                meanpsf *= upsample**2/np.sum(meanpsf)
            hires_arr[yreg, xreg] = meanpsf
            
    return hires_arr


def buildcalibrations(par,filelist=None, lamlist=None,hires=True,
                      order=3, lam1=605, lam2=725,inspect=True,
                      genwavelengthsol=True, makehiresPSFlets=True,
                      savehiresimages=True,borderpix = 4,
                      upsample=3,nsubarr=3,parallel=True):
    """
    """
    outdir = par.wavecalDir
    R = par.R
    
    if filelist is None:
        if par.filelist is None:
            raise
        else:
            filelist = par.filelist
    if lamlist is None:
        if par.lamlist is None:
            raise
        else:
            lamlist = par.lamlist
    
    
    try: 
        os.makedirs(outdir)
    except OSError:
        if not os.path.isdir(outdir):
            raise

    log.info("Building calibration files, placing results in " + outdir)

    tstart = time.time()
    coef = None
    allcoef = []
    imlist = []
    ysize,xsize = Image(filename=filelist[0]).data.shape
    for i, ifile in enumerate(filelist):
        im = Image(filename=ifile)
        # this is just to keep while we use noiseless images. Remove when real images are used.
        im.data+=1e-9
        imlist += [im]
        if genwavelengthsol:
            x, y, good, coef = locatePSFlets(im, polyorder=order, sig=par.FWHM,coef=coef,phi=par.philens,scale=par.pitch/par.pixsize,nlens=par.nlens)
            allcoef += [[lamlist[i]] + list(coef)]
            if inspect:
                do_inspection(par,im.data,x,y,lamlist[i])

    
    if genwavelengthsol:
        log.info("Saving wavelength solution to " + outdir + "lamsol.dat")
        allcoef = np.asarray(allcoef)
        np.savetxt(outdir + "lamsol.dat", allcoef)
        lam = allcoef[:, 0]
        allcoef = allcoef[:, 1:]
        
    else:
        log.info("Loading wavelength solution from " + outdir + "lamsol.dat")
        lam = np.loadtxt(outdir + "lamsol.dat")[:, 0]
        allcoef = np.loadtxt(outdir + "lamsol.dat")[:, 1:]

    log.info("Computing wavelength values at pixel centers")
    psftool = PSFLets()
    psftool.genpixsol(par,lam, allcoef, lam1=lam1/1.04, lam2=lam2*1.03)
    psftool.savepixsol(outdir=outdir)

    xindx = np.arange(-par.nlens/2, par.nlens/2)
    xindx, yindx = np.meshgrid(xindx, xindx)

    if hires:

        hires_arrs = []
        allxpos = []
        allypos = []
    
        log.info('Making high-resolution PSFLet models')

        if parallel:
            log.info('Starting parallel computation')
            for i in range(len(lam)):

                xpos, ypos = psftool.return_locations(lam[i], allcoef, xindx, yindx)
                xpos = np.reshape(xpos, -1)
                ypos = np.reshape(ypos, -1)
                allxpos += [xpos]
                allypos += [ypos]

            tasks = multiprocessing.Queue()
            results = multiprocessing.Queue()
            ncpus = multiprocessing.cpu_count()
            consumers = [ Consumer(tasks, results)
                          for i in range(ncpus) ]
            for w in consumers:
                w.start()
        
            for i in range(len(lam)):
                tasks.put(Task(i, gethires, (allxpos[i], allypos[i],
                                                            imlist[i], upsample,nsubarr)))
        
            for i in range(ncpus):
                tasks.put(None)
            for i in range(len(lam)):
                index, hiresarr = results.get()
                hires_arrs += [hiresarr]
        
                if savehiresimages:
                    di, dj = hiresarr.shape[0], hiresarr.shape[2]
                    outim = np.zeros((di*dj, di*dj))
                    for ii in range(di):
                        for jj in range(di):
                            outim[ii*dj:(ii + 1)*dj, jj*dj:(jj + 1)*dj] = hiresarr[ii, jj]
                    out = pyf.HDUList(pyf.PrimaryHDU(hiresarr.astype(np.float32)))
                    out.writeto(outdir + 'hires_psflets_lam%d.fits' % (lamlist[index]), clobber=True)
        else:
            log.info('No parallel computation')
            for i in range(len(lam)):
                xpos, ypos = psftool.return_locations(lam[i], allcoef, xindx, yindx)
                xpos = np.reshape(xpos, -1)
                ypos = np.reshape(ypos, -1)
                hiresarr = gethires(xpos, ypos, imlist[i],upsample,nsubarr)
                hires_arrs += [hiresarr]

                if savehiresimages:
                    di, dj = hiresarr.shape[0], hiresarr.shape[2]
                    outim = np.zeros((di*dj, di*dj))
                    for ii in range(di):
                        for jj in range(di):
                            outim[ii*dj:(ii + 1)*dj, jj*dj:(jj + 1)*dj] = hiresarr[ii, jj]
                    out = pyf.HDUList(pyf.PrimaryHDU(hiresarr.astype(np.float32)))
                    out.writeto(par.wavecalDir + 'hires_psflets_lam%d.fits' % (lam[i]), clobber=True)

        Nspec = int(np.log(lam2*1./lam1)*R + 1.5)
        log.info('Reduced cube will have %d wavelength bins' % Nspec)
        loglam_endpts = np.linspace(np.log(lam1), np.log(lam2), Nspec)
        loglam_midpts = (loglam_endpts[1:] + loglam_endpts[:-1])/2
        lam_endpts = np.exp(loglam_endpts)
        lam_midpts = np.exp(loglam_midpts)
        polyimage = np.zeros((Nspec - 1, ysize, xsize))
        xpos = []
        ypos = []
        good = []

        log.info('Making polychrome cube')
    
        if parallel==False:
            for i in range(Nspec - 1):
                polyimage[i] = make_polychrome(lam_endpts[i], lam_endpts[i + 1],
                                                          hires_arrs, lam, psftool, 
                                                          allcoef, xindx, yindx,ysize,xsize,upsample=upsample)
                _x, _y = psftool.return_locations(lam_midpts[i], allcoef, xindx, yindx)
                _good = (_x > borderpix)*(_x < xsize-borderpix)*(_y > borderpix)*(_y < ysize-borderpix)
                xpos += [_x]
                ypos += [_y]
                good += [_good]
        else:
            tasks = multiprocessing.Queue()
            results = multiprocessing.Queue()
            ncpus = multiprocessing.cpu_count()
            consumers = [ Consumer(tasks, results)
                          for i in range(ncpus) ]
            for w in consumers:
                w.start()
        
            for i in range(Nspec - 1):
                tasks.put(Task(i, make_polychrome, (lam_endpts[i], lam_endpts[i + 1],
                                                          hires_arrs, lam, psftool, 
                                                          allcoef, xindx, yindx,ysize,xsize,upsample)))
        
            for i in range(ncpus):
                tasks.put(None)
            for i in range(Nspec - 1):
                index, poly = results.get()
                polyimage[index] = poly
                _x, _y = psftool.return_locations(lam_midpts[index], allcoef, xindx, yindx)
                _good = (_x > borderpix)*(_x < xsize-borderpix)*(_y > borderpix)*(_y < ysize-borderpix)
                xpos += [_x]
                ypos += [_y]
                good += [_good]
            
        log.info('Saving polychrome cube')

        out = pyf.HDUList(pyf.PrimaryHDU(polyimage.astype(np.float32)))
        out.writeto(outdir + 'polychromeR%d.fits' % (R), clobber=True)
        out = pyf.HDUList(pyf.PrimaryHDU(np.sum(polyimage,axis=0).astype(np.float32)))
        out.writeto(outdir + 'polychromeR%dstack.fits' % (R), clobber=True)
    
    else:
        lam_midpts = lam
        xpos = []
        ypos = []
        good = []

        for i in range(len(lam)):
            _x, _y = psftool.return_locations(lam_midpts[i], allcoef, xindx, yindx)
            _good = (_x > borderpix)*(_x < xsize-borderpix)*(_y > borderpix)*(_y < ysize-borderpix)
            xpos += [_x]
            ypos += [_y]
            good += [_good]
    
    log.info('Saving wavelength calibration cube')
    outkey = pyf.HDUList(pyf.PrimaryHDU(lam_midpts))
    outkey.append(pyf.PrimaryHDU(np.asarray(xpos)))
    outkey.append(pyf.PrimaryHDU(np.asarray(ypos)))
    outkey.append(pyf.PrimaryHDU(np.asarray(good).astype(np.uint8)))
    outkey.writeto(outdir + 'polychromekeyR%d.fits' % (R), clobber=True)
    
    print ("Total time elapsed: %.0f s" % (time.time() - tstart))
