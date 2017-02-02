from astropy.io import fits as pyf
import numpy as np
import logging as log
from scipy import signal
from scipy.interpolate import interp1d
from photutils import RectangularAperture
from photutils import aperture_photometry
from scipy import ndimage
from tools.locate_psflets import PSFLets
from tools.image import Image
from scipy import interpolate

def _smoothandmask(datacube, good):
    """
    Set bad spectral measurements to an inverse variance of zero.  The
    threshold for effectively discarding data is an inverse variance
    less than 10% of the smoothed inverse variance, i.e., a
    measurement much worse than the surrounding ones.  This rejection
    is done separately at each wavelength.  

    Then use a smoothed, inverse-variance-weighted map to replace the
    values of the masked spectral measurements.  Note that this last
    step is purely cosmetic as the inverse variances are, in any case,
    zero.

    Inputs:
    1. datacube: image class containing 3D arrays data and ivar
    2. good:     2D array, nonzero = good lenslet

    Output:
    1. datacube: input datacube modified in place

    """

    ivar = datacube.ivar
    cube = datacube.data

    x = np.arange(7) - 3
    x, y = np.meshgrid(x, x)
    widewindow = np.exp(-(x**2 + y**2))
    narrowwindow = np.exp(-2*(x**2 + y**2))
    widewindow /= np.sum(widewindow)
    narrowwindow /= np.sum(narrowwindow)

    for i in range(cube.shape[0]):
        ivar_smooth = signal.convolve2d(ivar[i], widewindow, mode='same')
        ivar[i] *= ivar[i] > ivar_smooth/10.
        
        mask = signal.convolve2d(cube[i]*ivar[i], narrowwindow, mode='same')
        mask /= signal.convolve2d(ivar[i], narrowwindow, mode='same') + 1e-100
        indx = np.where(np.all([ivar[i] == 0, good], axis=0))
        cube[i][indx] = mask[indx]

    return datacube


def simpleReduction(par,name,ifsimage):
    '''
    Basic cube reduction using an IFS image and a wavecal cube
    Equivalent to method 1 in the IDL primitive 'pisces_assemble_spectral_datacube'
    '''
    log.info('Simple reduction')
    calCube = pyf.open(par.wavecalDir+par.wavecalName)
    
    waveCalArray = calCube[0].data
    waveCalArray = waveCalArray/1000.
    
    xcenter = calCube[1].data
    nlens = xcenter.shape[1]
    ycenter = calCube[2].data
    good = calCube[3].data
    ydim,xdim = ifsimage.shape
    xdim-=2
    xcenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan
    ycenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan

    
#     sdpx = 26 # length of longest spectra in pixels; need to link this to the filter setting
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
#     
#     wavelengths = np.arange(sdpx+1)*(lam_long-lam_short)/float(sdpx)+lam_short
    wavelengths = np.arange(lam_short,lam_long,par.dlam)
    cube = np.zeros((len(wavelengths),nlens,nlens))

    X = np.zeros((nlens,nlens),dtype='i4')
    Y = np.zeros((nlens,nlens),dtype='i4')
    #X *= np.NaN
    #Y *= np.NaN
    
    for wav in range(len(wavelengths)):
        lam = wavelengths[wav]
        log.info('Wavelength = %3.1f' % (lam*1000.))
        #index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-lam))
        #print index
        for i in range(nlens):
            for j in range(nlens):
                if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
                    #print xcenter[:,i,j],np.isnan(xcenter[:,i,j]).any()
                    #print np.isnan(xcenter[0,i,j])
                    fx = interp1d(waveCalArray,xcenter[:,i,j])
                    fy = interp1d(waveCalArray,ycenter[:,i,j])
                    #print (lam,fx(lam),xcenter[:,i,j])
                    Y[j,i] = np.int(fx(lam))
                    X[j,i] = np.int(fy(lam))            
        cube[wav,:,:] = ifsimage[X,Y]+ifsimage[X,Y+1]+ \
            ifsimage[X,Y-1]+ifsimage[X,Y+2]+ifsimage[X,Y-2]
    
    cube[cube==0] = np.NaN
    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
    return cube


def densifiedSimpleReduction(par,name,ifsimage,ratio=10.):
    '''
    Basic cube reduction using an IFS image and a wavecal cube
    Equivalent to method 1 in the IDL primitive 'pisces_assemble_spectral_datacube'
    '''
    calCube = pyf.open(par.wavecalDir+par.wavecalName)
    
    nx = int(ifsimage.shape[0] * ratio)
    ny = int(ifsimage.shape[1] * ratio)
    
    x = np.arange(nx)/ratio 
    y = np.arange(ny)/ratio

    x, y = np.meshgrid(x, y)
    
    log.info('Densifying image by a factor %d' % ratio)
    ifsimageDense = ndimage.map_coordinates(ifsimage, [y, x],cval=np.nan)
    ifsimageDense /= ratio**2 # conserve energy

    
    waveCalArray = calCube[0].data
    waveCalArray = waveCalArray/1000.
    
    xcenter = calCube[1].data
    nlens = xcenter.shape[1]
    ycenter = calCube[2].data
    good = calCube[3].data
    #xcenter[~good] = np.NaN
    #ycenter[~good] = np.NaN
    ydim,xdim = ifsimageDense.shape
    xmargin = (ratio*5)//2
    ymargin = (ratio)//2
    xdim-=xmargin
    ydim-=ymargin
    xcenter[~((xcenter*ratio<xdim)*(ycenter*ratio<ydim)*(xcenter*ratio>xmargin)*(ycenter*ratio>ymargin))] = np.nan
    ycenter[~((xcenter*ratio<xdim)*(ycenter*ratio<ydim)*(xcenter*ratio>xmargin)*(ycenter*ratio>ymargin))] = np.nan

    
#     sdpx = 26 # length of longest spectra in pixels; need to link this to the filter setting
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
#     
#     wavelengths = np.arange(sdpx+1)*(lam_long-lam_short)/float(sdpx)+lam_short
    wavelengths = np.arange(lam_short,lam_long,par.dlam)
    
    cube = np.zeros((len(wavelengths),nlens,nlens))

    X = np.zeros((nlens,nlens),dtype='i4')
    Y = np.zeros((nlens,nlens),dtype='i4')
    #X *= np.NaN
    #Y *= np.NaN
    ratio = int(ratio)
    for wav in range(len(wavelengths)):
        lam = wavelengths[wav]
        log.info('Wavelength = %3.1f' % (lam*1000.))
        #index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-lam))
        #print index
        for i in range(nlens):
            for j in range(nlens):
                if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
                    #print xcenter[:,i,j],np.isnan(xcenter[:,i,j]).any()
                    #print np.isnan(xcenter[0,i,j])
                    fx = interp1d(waveCalArray,xcenter[:,i,j])
                    fy = interp1d(waveCalArray,ycenter[:,i,j])
                    #print (lam,fx(lam),xcenter[:,i,j])
                    Y[j,i] = np.floor(fx(lam)*ratio)
                    X[j,i] = np.floor(fy(lam)*ratio)
        for xcount in range(-ratio//2,-ratio//2+ratio):
            for ycount in range(-(ratio*5)//2,-(ratio*5)//2+ratio*5):
                cube[wav,:,:] += ifsimageDense[X+xcount,Y+ycount]
            #cube[wav,:,:] = ifsimage[X,Y]+ifsimage[X,Y+1]+ \
            #ifsimage[X,Y-1]+ifsimage[X,Y+2]+ifsimage[X,Y-2]
    
    cube[cube==0] = np.NaN
    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
    return cube


def apertureReduction(par,name,ifsimage):
    '''
    Basic cube reduction using an IFS image and a wavecal cube
    Equivalent to method 1 in the IDL primitive 'pisces_assemble_spectral_datacube'
    '''
    calCube = pyf.open(par.wavecalDir+par.wavecalName)
    
    waveCalArray = calCube[0].data#wavecal[0,:,:]
#     waveCalArray = waveCalArray[waveCalArray>0]/1000.
    waveCalArray = waveCalArray/1000.
    
#     xcenter = np.zeros((numWCal,nlens,nlens))
#     ycenter = np.zeros((numWCal,nlens,nlens))
#     for i in range(numWCal):
#         xcenter[i,:,:] = wavecal[i*2+1,:,:]
#         ycenter[i,:,:] = wavecal[i*2+2,:,:]
    xcenter = calCube[1].data
    nlens = xcenter.shape[1]
    ycenter = calCube[2].data
    good = calCube[3].data
    #xcenter[~good] = np.NaN
    #ycenter[~good] = np.NaN
    ydim,xdim = ifsimage.shape
    xdim-=2
    xcenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan
    ycenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan

    
#     sdpx = 26 # length of longest spectra in pixels; need to link this to the filter setting
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
#     
#     wavelengths = np.arange(sdpx+1)*(lam_long-lam_short)/float(sdpx)+lam_short
    wavelengths = np.arange(lam_short,lam_long,par.dlam)
    cube = np.zeros((len(wavelengths),nlens,nlens))

    psftool = PSFLets()
    lam = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    # lam in nm
    psftool.geninterparray(lam, allcoef)
    
    for wav in range(len(wavelengths)):
        lam = wavelengths[wav]
        log.info('Wavelength = %3.1f' % (lam*1000.))
        #index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-lam))
        #print index
        for i in range(nlens):
            for j in range(nlens):
                if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
                    #print xcenter[:,i,j],np.isnan(xcenter[:,i,j]).any()
                    #print np.isnan(xcenter[0,i,j])
                    #fx = interp1d(waveCalArray,xcenter[:,i,j])
                    #fy = interp1d(waveCalArray,ycenter[:,i,j])
                    #print (lam,fx(lam),xcenter[:,i,j])
                    #pos = (fx(lam),fy(lam))
                    _x,_y = psftool.return_locations(lam*1000., allcoef, j-nlens/2, i-nlens/2)
                    pos = (_x,_y)
                    ap = RectangularAperture(pos,1,5,0)
                    cube[wav,j,i] = aperture_photometry(ifsimage,ap)['aperture_sum'][0]
    
    cube[cube==0] = np.NaN
    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
    return cube


def testReduction(par,name,ifsimage):
    calCube = pyf.open(par.wavecalDir+par.wavecalName)
    
    waveCalArray = calCube[0].data#wavecal[0,:,:]
    waveCalArray = waveCalArray/1000.
    
    xcenter = calCube[1].data
    nlens = xcenter.shape[1]
    ydim,xdim = ifsimage.shape
    
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
    wavelengths = np.arange(lam_short,lam_long,par.dlam)
    cube = np.zeros((len(wavelengths),nlens,nlens))

    psftool = PSFLets()
    lam = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    # lam in nm
    psftool.geninterparray(lam, allcoef)
    xindx = np.arange(-nlens/2, nlens/2)
    xindx, yindx = np.meshgrid(xindx, xindx)

    for iwav in range(len(wavelengths)): 
        wav = wavelengths[iwav]
        log.info('Wavelength = %3.1f' % (wav*1000.))
        xcenter, ycenter = psftool.return_locations(wav*1000., allcoef, xindx, yindx)
        good = (xcenter > 2)*(xcenter < xdim-2)*(ycenter > 3)*(ycenter < ydim-3)
        xcenter = np.reshape(xcenter,-1)
        ycenter = np.reshape(ycenter,-1)
        good = np.reshape(good,-1)
        xcenter[~good] = xdim/2
        ycenter[~good] = ydim/2
    
        pos = zip(xcenter,ycenter)
        aps = RectangularAperture(pos,1,5,0)
        table = aperture_photometry(ifsimage,aps)['aperture_sum']
        for i in range(nlens):
            for j in range(nlens):
                if good[j+i*nlens]:
                    cube[iwav,j,i] = table[j+i*nlens]
                else:
                    cube[iwav,j,i] = np.NaN                

    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
    return cube


def lstsqExtract(par,name,ifsimage,ivar=False):
    '''
    Least squares extraction, inspired by T. Brandt and making use of some of his code.
    
    Parameters
    ----------
    par:   Parameter instance
    name: string
            Name that will be given to final image, without fits extension
    ifsimage: Image instance of IFS detector map, with optional inverse variance
            First dimension needs to be the same length as lamlist
                
    Returns
    -------
    detectorFrame : 2D array
            Return the detector frame

    '''
    polychromeR = pyf.open(par.wavecalDir + 'polychromeR%d.fits' % (par.R))
    psflets = polychromeR[0].data
    
    psftool = PSFLets()
    lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    # lam in nm
    psftool.geninterparray(lamlist, allcoef)
    
    if ivar:
        if ifsimage.ivar is None:
            ifsimage.ivar = ifsimage.data.copy()
            ifsimage.ivar = 1./ifsimage.ivar
        else:
            ifsimage.ivar = None
    else:
        ifsimage.ivar = None
        
    cube = np.zeros((psflets.shape[0],par.nlens,par.nlens))
    
    ydim,xdim = ifsimage.data.shape
    
    for i in range(par.nlens):
        for j in range(par.nlens):
            xlist = []
            ylist = []
            good = True
            for lam in lamlist:
                _x,_y = psftool.return_locations(lam, allcoef, j-par.nlens/2, i-par.nlens/2)
                good *= (_x > 2)*(_x < xdim-2)*(_y > 3)*(_y < ydim-3)
                xlist += [_x]    
                ylist += [_y]   
                 
            if good:
                subim, psflet_subarr, [x0, x1, y0, y1] = get_cutout(ifsimage,xlist,ylist,psflets)
#             if i==par.nlens/2 and j==par.nlens/2:
#                 out = pyf.HDUList(pyf.PrimaryHDU(psflet_subarr.astype(np.float32)))
#                 out.writeto(par.unitTestsOutputs + '/psflet_cutouti0j0.fits', clobber=True)

                cube[:,j,i] = fit_cutout(subim.copy(), psflet_subarr.copy(), mode='lstsq')
            else:
                cube[:,j,i] = np.NaN

    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
    return cube

def get_cutout(im, x, y, psflets, dy=3):
    
    """
    Cut out a microspectrum for fitting.  Return the inputs to 
    linalg.lstsq or to whatever regularization scheme we adopt.
    Assumes that spectra are dispersed in the -y direction.

    Inputs:
    1. im:      Image object containing data to be fit
    2. x:       float, list of x centroids
    3. y:       float, list of y centroids of that microspectrum
    4. psflets: list of 2D ndarrays, each of which should have the
                same shape as image. Typically generated from polychrome
         
    Optional inputs:
    1. dy:      vertical length to cut out, default 3.  This is
                the length to cut out in the +/-y direction; the 
                lengths cut out in the +x direction (beyond the 
                shortest and longest wavelengths) are also dy.

    Returns: 
    1. subim:   a flattened subimage to be fit
    2. psflet_subarr: a 2D ndarray, first dimension is wavelength,
                second dimension is spatial, and is the same shape
                as the flattened subimage.

    Note: both subim and psflet_subarr are scaled by the inverse
    standard deviation if it is given for the input Image.  This 
    will make the fit chi2 and properly handle bad/masked pixels.

    """

    x0, x1 = [int(np.amin(x)) - dy+1, int(np.amax(x)) + dy+1]
    y0, y1 = [int(np.amin(y)) - dy, int(np.amax(y)) + dy+1]

    subim = im.data[y0:y1, x0:x1]
    if im.ivar is not None:
        isig = np.sqrt(im.ivar[y0:y1, x0:x1])
        subim *= isig

    subarrshape = tuple([len(psflets)] + list(subim.shape))
    psflet_subarr = np.zeros(subarrshape)
    for i in range(len(psflets)):
        psflet_subarr[i] = psflets[i][y0:y1, x0:x1]
        if im.ivar is not None:
            psflet_subarr[i] *= isig

    return subim, psflet_subarr, [y0,y1, x0,x1]

def fit_cutout(subim, psflets, mode='lstsq'):
    """
    Fit a series of PSFlets to an image, recover the best-fit coefficients.
    This is currently little more than a wrapper for np.linalg.lstsq, but 
    could be more complex if/when we regularize the problem or adopt some
    other approach.

    Inputs:
    1. subim:   2D nadarray, microspectrum to fit
    2. psflets: 3D ndarray, first dimension is wavelength.  psflets[0] 
                must match the shape of subim.
    3. mode:    string, method to use.  Currently limited to lstsq (a 
                simple least-squares fit using linalg.lstsq), this can
                be expanded to include an arbitrary approach.
    
    Returns:
    1. coef:    the best-fit coefficients (i.e. the microspectrum).

    Note: this routine may also return the covariance matrix in the future.
    It will depend on the performance of the algorithms and whether/how we
    implement regularization.
    """

    try:
        if not subim.shape == psflets[0].shape:
            raise ValueError("subim must be the same shape as each psflet.")
    except:
        raise ValueError("subim must be the same shape as each psflet.")
    
    if mode == 'lstsq':
        subim_flat = np.reshape(subim, -1)
        psflets_flat = np.reshape(psflets, (psflets.shape[0], -1))
        coef = np.linalg.lstsq(psflets_flat.T, subim_flat)[0]
    elif mode == 'ext':
        coef = np.zeros(psflets.shape[0])
        for i in range(psflets.shape[0]):
            coef[i] = np.sum(psflets[i]*subim)/np.sum(psflets[i])
    elif mode == 'apphot':
        coef = np.zeros((subim.shape[1]))
        for i in range(subim.shape[1]):
            coef[i] = np.sum(subim[:,i])
    else:
        raise ValueError("mode " + mode + " to fit microspectra is not currently implemented.")

    return coef


def intOptimalExtract(par,name,IFSimage):

    loc = PSFLets(load=True, infiledir=par.wavecalDir)
    lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    
    Nspec = int(np.log(max(lamlist)/min(lamlist))*par.R + 1.5)
    log.info('Reduced cube will have %d wavelength bins' % Nspec)
    loglam_endpts = np.linspace(np.log(min(lamlist)), np.log(max(lamlist)), Nspec)
    loglam_midpts = (loglam_endpts[1:] + loglam_endpts[:-1])/2
    lam_endpts = np.exp(loglam_endpts)
    lam_midpts = np.exp(loglam_midpts)

    datacube = fitspec_intpix(par,IFSimage, loc, lam_midpts)
    datacube.write(name+'.fits',clobber=True)
    return datacube.data

def fitspec_intpix(par,im, PSFlet_tool, lamlist, delt_y=6, flat=None, 
                   smoothandmask=False, header=pyf.PrimaryHDU().header):
    """
    """

    loglam = np.log(lamlist)

    ########################################################################
    # x-locations of the centers of the microspectra.  The y locations
    # are integer pixels, and the wavelengths in PSFlet_tool.lam_indx
    # are given at the centers of the pixels.  Dispersion is in the
    # y-direction.  Make copies of all arrays to ensure that they are
    # in native byte order as required by Cython.
    ########################################################################

    xindx = np.zeros(PSFlet_tool.xindx.shape)
    xindx[:] = PSFlet_tool.xindx
    yindx = np.zeros(PSFlet_tool.yindx.shape)
    yindx[:] = PSFlet_tool.yindx
    loglam_indx = np.log(PSFlet_tool.lam_indx + 1e-100)
    nlam = np.zeros(PSFlet_tool.nlam.shape, np.int32)
    nlam[:] = PSFlet_tool.nlam
    Nmax = max(PSFlet_tool.nlam_max, lamlist.shape[0])

    data = np.zeros(im.data.shape)
    data[:] = im.data
    ivar = np.zeros(im.ivar.shape)
    ivar[:] = im.ivar
    #print np.exp(loglam_indx)
#     coefs, tot_ivar = optext(data, ivar, xindx, yindx, 
#                                       loglam_indx, nlam, loglam, Nmax, 
#                                       delt_y=6, sig=1.5)

#     header['cubemode'] = ('Optimal Extraction', 'Method used to extract data cube')
#     header['lam_min'] = (np.amin(lam), 'Minimum (central) wavelength of extracted cube')
#     header['lam_max'] = (np.amax(lam), 'Maximum (central) wavelength of extracted cube')
#     header['dloglam'] = (np.log(lam[1]/lam[0]), 'Log spacing of extracted wavelength bins')
    lamsol = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    # lam in nm
    PSFlet_tool.geninterparray(lamsol, allcoef)
    cube = np.zeros((len(lamlist),par.nlens,par.nlens))
    ydim,xdim = im.data.shape
    sig = par.FWHM/2.35
    for i in range(par.nlens):
        for j in range(par.nlens):
            good = True
            for lam in lamlist:
                _x,_y = PSFlet_tool.return_locations(lam, allcoef, j-par.nlens/2, i-par.nlens/2)
                good *= (_x > delt_y)*(_x < xdim-delt_y)*(_y > delt_y)*(_y < ydim-delt_y)
                 
            if good:
                ix = xindx[i, j, :PSFlet_tool.nlam[i, j]]
                y = yindx[i, j, :PSFlet_tool.nlam[i, j]]
                iy = np.mean(y)
                lams = PSFlet_tool.lam_indx[i,j,:PSFlet_tool.nlam[i, j]]
                i1 = int(iy) + 1 - delt_y/2.
                arr = np.arange(i1,i1 + delt_y)
                dy = arr-iy
                gaussian = np.exp(-dy**2/sig**2/2.)
                weights = np.sum(gaussian**2)
#                 pix_center_vals = [np.sum(im.data[i1:i1 + delt_y, val],axis=0) for val in ix]
                pix_center_vals = [np.sum(im.data[i1:i1 + delt_y, val]*gaussian) for val in ix]/weights
                func = interp1d(lams,pix_center_vals,kind='linear')
                cube[:,j,i] = func(lamlist)
            else:
                cube[:,j,i] = np.NaN
#     datacube = Image(data=coefs, ivar=tot_ivar, header=header)
# 
#     if flat is not None:
#         datacube.data /= flat + 1e-10
#         datacube.ivar *= flat**2
# 
#     if smoothandmask:
#         good = np.any(datacube.data != 0, axis=0)
#         datacube = _smoothandmask(datacube, good)
# 
#     return datacube
    return Image(data=cube)
    #pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)




def fitspec_intpix_np(par,im, PSFlet_tool, lamlist, delt_y=7):
    """
    """

    xindx = PSFlet_tool.xindx
    yindx = PSFlet_tool.yindx
    Nmax = PSFlet_tool.nlam_max
    print (Nmax)

    x = np.arange(im.data.shape[1])
    y = np.arange(im.data.shape[0])
    x, y = np.meshgrid(x, y)

    ydim,xdim = im.data.shape


    coefs = np.zeros(tuple([max(Nmax, lamlist.shape[0])] + list(yindx.shape)[:-1]))
    
    xarr, yarr = np.meshgrid(np.arange(Nmax),np.arange(delt_y))

    loglam = np.log(lamlist)
    lamsol = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]
    PSFlet_tool.geninterparray(lamsol, allcoef)
    
    for i in range(xindx.shape[0]):
        for j in range(yindx.shape[1]):
            good = True
            for lam in lamlist:
                _x,_y = PSFlet_tool.return_locations(lam, allcoef, j-par.nlens/2, i-par.nlens/2)
                good *= (_x > 2)*(_x < xdim-2)*(_y > 3)*(_y < ydim-3)
                 
            if good:
                _x = xindx[i, j, :PSFlet_tool.nlam[i, j]]
#                 _x = _x[_x>0]
                _y = yindx[i, j, :PSFlet_tool.nlam[i, j]]
#                 _y = _y[_x>0]
                _lam = PSFlet_tool.lam_indx[i, j, :PSFlet_tool.nlam[i, j]]
#                 _lam = _lam[_x>0]
                print (len(_x),len(_y),len(_lam))

                if not (np.all(_x > x[0, 10]) and np.all(_x < x[0, -10]) and 
                        np.all(_y > y[10, 0]) and np.all(_y < y[-10, 0])):
                    continue


                i1 = int(np.mean(_y) - delt_y/2.)
                dy = _y[yarr[:,:len(_lam)]] - y[i1:i1 + delt_y,_x[0]:_x[-1] + 1]
                #var = _var[yarr[:len(_lam)]] - x[_y[0]:_y[-1] + 1, i1:i1 + delt_x]
                sig = par.FWHM/2.35
                weight = np.exp(-dy**2/2./sig**2)
                data = im.data[i1:i1 + delt_y,_x[0]:_x[-1] + 1]
                if im.ivar is not None:
                    ivar = im.ivar[i1:i1 + delt_y,_x[0]:_x[-1] + 1]
                else:
                    ivar = np.ones(data.shape)

                coefs[:len(_lam), i, j] = np.sum(weight*data*ivar, axis=0)[::-1]
                coefs[:len(_lam), i, j] /= np.sum(weight**2*ivar, axis=0)[::-1]

                tck = interpolate.splrep(np.log(_lam[::-1]), coefs[:len(_lam), i, j], s=0, k=3)
                coefs[:loglam.shape[0], i, j] = interpolate.splev(loglam, tck, ext=1)

#     header['cubemode'] = ('Optimal Extraction', 'Method used to extract data cube')
#     header['lam_min'] = (np.amin(lam), 'Minimum (central) wavelength of extracted cube')
#     header['lam_max'] = (np.amax(lam), 'Maximum (central) wavelength of extracted cube')
#     header['dloglam'] = (np.log(lam[1]/lam[0]), 'Log spacing of extracted wavelength bins')
#     header['nlam'] = (lam.shape[0], 'Number of extracted wavelengths')

#     datacube = Image(data=coefs[:loglam.shape[0]], header=header)
    datacube = Image(data=coefs[:loglam.shape[0]])
    return datacube

def fitspec_intpix_np_old(im, PSFlet_tool, lam, delt_x=7, header=pyf.PrimaryHDU().header):
    """
    """

    xindx = PSFlet_tool.xindx
    yindx = PSFlet_tool.yindx
    Nmax = PSFlet_tool.nlam_max

    x = np.arange(im.data.shape[1])
    y = np.arange(im.data.shape[0])
    x, y = np.meshgrid(x, y)

    coefs = np.zeros(tuple([max(Nmax, lam.shape[0])] + list(xindx.shape)[:-1]))
    
    xarr, yarr = np.meshgrid(np.arange(delt_x), np.arange(Nmax))

    loglam = np.log(lam)

    for i in range(xindx.shape[0]):
        for j in range(yindx.shape[1]):
            _x = xindx[i, j, :PSFlet_tool.nlam[i, j]]
            _y = yindx[i, j, :PSFlet_tool.nlam[i, j]]
            _lam = PSFlet_tool.lam_indx[i, j, :PSFlet_tool.nlam[i, j]]

            if not (np.all(_x > x[0, 10]) and np.all(_x < x[0, -10]) and 
                    np.all(_y > y[10, 0]) and np.all(_y < y[-10, 0])):
                continue


            i1 = int(np.mean(_x) - delt_x/2.)
            dx = _x[yarr[:len(_lam)]] - x[_y[0]:_y[-1] + 1, i1:i1 + delt_x]
            #var = _var[yarr[:len(_lam)]] - x[_y[0]:_y[-1] + 1, i1:i1 + delt_x]
            sig = 0.7
            weight = np.exp(-dx**2/2./sig**2)
            data = im.data[_y[0]:_y[-1] + 1, i1:i1 + delt_x]
            if im.ivar is not None:
                ivar = im.ivar[_y[0]:_y[-1] + 1, i1:i1 + delt_x]
            else:
                ivar = np.ones(data.shape)

            coefs[:len(_lam), i, j] = np.sum(weight*data*ivar, axis=1)[::-1]
            coefs[:len(_lam), i, j] /= np.sum(weight**2*ivar, axis=1)[::-1]

            tck = interpolate.splrep(np.log(_lam[::-1]), coefs[:len(_lam), i, j], s=0, k=3)
            coefs[:loglam.shape[0], i, j] = interpolate.splev(loglam, tck, ext=1)

    header['cubemode'] = ('Optimal Extraction', 'Method used to extract data cube')
    header['lam_min'] = (np.amin(lam), 'Minimum (central) wavelength of extracted cube')
    header['lam_max'] = (np.amax(lam), 'Maximum (central) wavelength of extracted cube')
    header['dloglam'] = (np.log(lam[1]/lam[0]), 'Log spacing of extracted wavelength bins')
    header['nlam'] = (lam.shape[0], 'Number of extracted wavelengths')

    datacube = Image(data=coefs[:loglam.shape[0]], header=header)
    return datacube
