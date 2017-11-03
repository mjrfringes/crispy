
import numpy as np
import tools
from tools.initLogger import getLogger
log = getLogger('crispy')
import matplotlib.pyplot as plt
from tools.image import Image
try:
    from astropy.io import fits
except:
    import pyfits as fits
from tools.locate_psflets import PSFLets
from tools.reduction import get_cutout,fit_cutout,calculateWaveList
from IFS import polychromeIFS
from tools.spectrograph import selectKernel,loadKernels
from tools.plotting import plotKernels
from scipy import ndimage
from scipy.interpolate import interp1d



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



def testCutout(par,fname,lensX = 0,lensY = 0):
    '''
    Testing the cutout function
    
    '''
    
    # first load polychrome
    polychromeR = fits.open(par.wavecalDir + 'polychromeR%d.fits' % (par.R))
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
    subim, psflet_subarr, [x0, x1, y0, y1] = get_cutout(im,xlist,ylist,psflets)
    print [x0, x1, y0, y1]
    Image(data=subim).write(par.unitTestsOutputs+'/cutout.fits')
    out = fits.HDUList(fits.PrimaryHDU(subim.astype(np.float32)))
    out.writeto(par.unitTestsOutputs + '/subim.fits', clobber=True)
    return subim


def testFitCutout(par,fname,lensnum, mode='lstsq',ivar=False):
    '''
    Testing the fit_cutout function
    
    '''
    # first load polychrome
    polychromeR = fits.open(par.wavecalDir + 'polychromeR%d.fits' % (par.R))
    psflets = polychromeR[0].data
    
    psftool = PSFLets()
    lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    # lam in nm
    psftool.geninterparray(lamlist, allcoef)
    
    xlist = []
    ylist = []
    for lam in lamlist:
        _x,_y = psftool.return_locations(lam, allcoef, lensnum, lensnum)
        xlist += [_x]    
        ylist += [_y]    
    
    if isinstance(fname,basestring):
        im = Image(filename = fname)
    else:
        im = Image(data=fname)
    if ivar:
        im.ivar = im.data.copy()
        im.ivar = 1./im.ivar
    subim, psflet_subarr, [x0, x1, y0, y1] = get_cutout(im,xlist,ylist,psflets)
    return fit_cutout(subim, psflet_subarr, mode=mode)


def testGenPixSol(par):
    psftool = PSFLets()
    lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    psftool.geninterparray(lamlist, allcoef)
    psftool.genpixsol(lamlist,allcoef)
    psftool.savepixsol(outdir = par.exportDir)


def testCreateFlatfield(par,pixsize = 0.1, npix = 512, pixval = 1.,Nspec=45,outname='flatfield.fits',useQE=True,method='optext'):
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

