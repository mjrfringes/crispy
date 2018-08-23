
import numpy as np
from crispy.tools.initLogger import getLogger
log = getLogger('crispy')
import matplotlib.pyplot as plt
from crispy.tools.image import Image
try:
    from astropy.io import fits
except:
    import pyfits as fits
from crispy.tools.locate_psflets import PSFLets
from crispy.tools.reduction import get_cutout,fit_cutout,calculateWaveList
from crispy.IFS import polychromeIFS
from crispy.tools.spectrograph import selectKernel,loadKernels
from crispy.tools.plotting import plotKernels
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy import interpolate



def testLoadKernels(par):
    '''
    Make sure the kernel interpolation with wavelength makes sense
    '''
    
    log.info('Import all kernels and rescale them to same plate scale')
    kernels890,locations = loadKernels(par,890)
    kernels770,loc = loadKernels(par,770)
    kernels660,loc = loadKernels(par,660)
    refWaveList = [660,770,890]
    kernelList = np.array([kernels660,kernels770,kernels890])

    for lam in np.arange(0.6,0.9,0.05):
        kernels = selectKernel(par,lam,refWaveList,kernelList)
        allkernels = plotKernels(par,kernels,locations)
        Image(data=allkernels).write(par.unitTestsOutputs+'/kernels%.3f.fits' % (lam))



def testCutout(par,fname,lensX = 0,lensY = 0, dy=2.5):
    '''
    Testing the cutout function
    
    '''
    
    # first load polychrome
    polychromeR = fits.open(par.wavecalDir + 'polychromeR%d.fits.gz' % (par.R))
    psflets = polychromeR[0].data
    
    psftool = PSFLets()
    lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    # lam in nm
    psftool.geninterparray(lamlist, allcoef)
    
    xlist = []
    ylist = []
    for lam in lamlist:
        _x,_y = psftool.return_locations(lam, allcoef, lensX, lensY)
        xlist += [_x]    
        ylist += [_y]    
    
    if isinstance(fname,basestring):
        im = Image(filename = fname)
    else:
        im = Image(data=fname)
    subim, psflet_subarr, [x0, x1, y0, y1] = get_cutout(im,xlist,ylist,psflets, dy=dy)
#     Image(data=subim).write(par.unitTestsOutputs+'/cutout.fits')
#     out = fits.HDUList(fits.PrimaryHDU(subim.astype(np.float32)))
#     out.writeto(par.unitTestsOutputs + '/subim.fits', clobber=True)
    return subim, psflet_subarr, [x0, x1, y0, y1]



def testFitCutout(par,fname,lensX, lensY, mode='lstsq',ivar=False, niter=3, pixnoise=0.0, dy=2.5):
    '''
    Testing the fit_cutout function
    
    '''
    # first load polychrome
    polychromeR = fits.open(par.wavecalDir + 'polychromeR%d.fits.gz' % (par.R))
    psflets = polychromeR[0].data
    
    psftool = PSFLets()
    lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    # lam in nm
    psftool.geninterparray(lamlist, allcoef)
    
    xlist = []
    ylist = []
    for lam in lamlist:
        _x,_y = psftool.return_locations(lam, allcoef, lensX, lensY)
        xlist += [_x]    
        ylist += [_y]    
    
    if isinstance(fname,basestring):
        im = Image(filename = fname)
    else:
        im = Image(data=fname)
    if ivar:
        im.ivar = im.data.copy()
        im.ivar = 1./im.ivar
    subim, psflet_subarr, [x0, x1, y0, y1] = get_cutout(im,xlist,ylist,psflets, dy=dy)
#     subim -= np.amin(np.sum(subim)
    return fit_cutout(subim, psflet_subarr, mode=mode, niter=3, pixnoise=pixnoise)


def testOptExt(par,im, lensX, lensY, smoothandmask=True, delt_y=5):
    """
    """


    PSFlet_tool = PSFLets(load=True, infiledir=par.wavecalDir)
    #Nspec = int(par.BW*par.npixperdlam*par.R)
    lamlist,scratch = calculateWaveList(par,method='optext')

    xindx = PSFlet_tool.xindx
    yindx = PSFlet_tool.yindx
    Nmax = PSFlet_tool.nlam_max
    try:
        sig = fits.open(par.wavecalDir + 'PSFwidths.fits')[0].data
    except:
        log.warning("No PSFLet widths found - assuming critical samping at central wavelength")
        sig=par.FWHM/2.35*np.ones(xindx.shape)
    
    x = np.arange(im.data.shape[1])
    y = np.arange(im.data.shape[0])
    x, y = np.meshgrid(x, y)

    ydim,xdim = im.data.shape

    coefs = np.zeros(tuple([max(Nmax, lamlist.shape[0])] + list(yindx.shape)[:-1]))
    xarr, yarr = np.meshgrid(np.arange(Nmax),np.arange(delt_y))

    #loglam = np.log(lamlist)
    lamsol = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]
    PSFlet_tool.geninterparray(lamsol, allcoef)
    
    good = PSFlet_tool.good

    
    i =lensX
    j =lensX
    
    if good[i,j]:
        _x = xindx[i, j, :PSFlet_tool.nlam[i, j]]
        _y = yindx[i, j, :PSFlet_tool.nlam[i, j]]
        _sig = sig[i, j, :PSFlet_tool.nlam[i, j]]
        _lam = PSFlet_tool.lam_indx[i, j, :PSFlet_tool.nlam[i, j]]
        iy = np.nanmean(_y)
        if ~np.isnan(iy):
            i1 = int(iy - delt_y/2.)
            dy = _y[xarr[:,:len(_lam)]] - y[i1:i1 + delt_y,int(_x[0]):int(_x[-1]) + 1]
            lams,_ = np.meshgrid(_lam,np.arange(delt_y))
            
            weight = np.exp(-dy**2/2./_sig**2)/_sig/np.sqrt(2.*np.pi)
            data = im.data[i1:i1 + delt_y,int(_x[0]):int(_x[-1]) + 1]
            
            if im.ivar is not None:
                ivar = im.ivar[i1:i1 + delt_y,int(_x[0]):int(_x[-1]) + 1]
            else:
                ivar = np.ones(data.shape)

            coefs[:len(_lam), i, j] = np.sum(weight*data*ivar, axis=0)
            coefs[:len(_lam), i, j] /= np.sum(weight**2*ivar, axis=0)
            tck = interpolate.splrep(_lam, coefs[:len(_lam), i, j], s=0, k=3)
            outspec = interpolate.splev(lamlist, tck, ext=1)
            tck = interpolate.splrep(_lam, np.sum(weight**2*ivar, axis=0)/np.sum(weight**2, axis=0), s=0, k=3)
            outvar = interpolate.splev(lamlist, tck, ext=1)
                

    return outspec,outvar



def testGenPixSol(par):
    psftool = PSFLets()
    lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    psftool.geninterparray(lamlist, allcoef)
    psftool.genpixsol(lamlist,allcoef)
    psftool.savepixsol(outdir = par.exportDir)


def testCreateFlatfield(par,pixsize = 0.1,
                        npix = 512, 
                        pixval = 1.,
                        Nspec=45,
                        outname='flatfield.fits',
                        useQE=True,
                        method='optext',
                        maxflux=400,
                        bg=10):
    '''
    Creates a polychromatic flatfield
    
    Parameters
    ----------
    par :   Parameter instance
        Contains all IFS parameters
    pixsize:   float
       Pixel scale (lam/D)
    npix: int
        Each input frame has a pixel size npix x npix
    pixval: float
        Each input frame has a unform value pixval in photons per second per nm of bandwidth
    Nspec: float
        Optional input forcing the number of wavelengths bins used
    outname: string
        Name of flatfield image
    useQE: boolean 
        Whether to take into account the wavelength-dependent QE of the detector
    
    '''
    
    lam_midpts,lam_endpts = calculateWaveList(par,Nspec=Nspec,method=method)
    inputCube = np.ones((len(lam_midpts),npix,npix),dtype=np.float32)
    
    for i in range(len(lam_midpts)):
        inputCube[i,:,:]*=pixval #/lam_midpts[i]
        
    par.saveDetector=False
    inCube = fits.HDUList(fits.PrimaryHDU(inputCube.astype(np.float32)))
    inCube[0].header['LAM_C'] = np.median(lam_midpts)/1000.
    inCube[0].header['PIXSIZE'] = pixsize
    inCube.writeto(par.unitTestsOutputs+'/flatfield_input.fits',clobber=True)
    detectorFrame = polychromeIFS(par,lam_midpts,inCube[0],parallel=True,wavelist_endpts=lam_endpts,QE=useQE)
    detectorFrame = np.random.poisson(detectorFrame*maxflux/np.amax(detectorFrame)+bg)-bg
    Image(data=detectorFrame,header=par.hdr).write(par.unitTestsOutputs+'/'+outname,clobber=True)
    

import scipy
from scipy.ndimage.filters import gaussian_filter1d
def testCrosstalk(par,pixsize = 0.1, npix = 512, pixval = 1.,Nspec=45,outname='crosstalk.fits',useQE=True,method='optext'):
    '''
    Creates a polychromatic flatfield
    
    Parameters
    ----------
    par :   Parameter instance
        Contains all IFS parameters
    pixsize:   float
       Pixel scale (lam/D)
    npix: int
        Each input frame has a pixel size npix x npix
    pixval: float
        Each input frame has a unform value pixval in photons per second per nm of bandwidth
    Nspec: float
        Optional input forcing the number of wavelengths bins used
    outname: string
        Name of flatfield image
    useQE: boolean 
        Whether to take into account the wavelength-dependent QE of the detector
    
    '''
    
    lam_midpts,lam_endpts = calculateWaveList(par,Nspec=Nspec,method=method)
    inputCube = np.zeros((len(lam_midpts),npix,npix),dtype=np.float32)
    
    for i in range(len(lam_midpts)):
        inputCube[i,npix//2,npix//2]+=pixval #/lam_midpts[i]
    
#     lam_midpts_nom,_ = calculateWaveList(par,method=method)

#     FWHM=Nspec/len(lam_midpts_nom)
#     inputCube[:,npix//2,npix//2] = gaussian_filter1d(inputCube[:,npix//2,npix//2],sigma=FWHM/2.35)
            
    par.saveDetector=False
    inCube = fits.HDUList(fits.PrimaryHDU(inputCube.astype(np.float32)))
    inCube[0].header['LAM_C'] = np.median(lam_midpts)/1000.
    inCube[0].header['PIXSIZE'] = pixsize
    inCube.writeto(par.unitTestsOutputs+'/crosstalk_input.fits',clobber=True)
    detectorFrame = polychromeIFS(par,lam_midpts,inCube[0],parallel=True,wavelist_endpts=lam_endpts,QE=useQE,noRot=True)
    Image(data=detectorFrame,header=par.hdr).write(par.unitTestsOutputs+'/'+outname,clobber=True)

