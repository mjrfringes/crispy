
import numpy as np
import tools
from tools.initLogger import getLogger
log = getLogger('crispy')
import matplotlib.pyplot as plt
from tools.image import Image
from params import Params
try:
    from astropy.io import fits
except:
    import pyfits as fits
from tools.locate_psflets import PSFLets
from tools.reduction import get_cutout,fit_cutout,calculateWaveList
from IFS import propagateIFS,polychromeIFS
from tools.spectrograph import selectKernel,loadKernels
from tools.plotting import plotKernels
from scipy import ndimage


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



def testCutout(par,fname,lensnum = 0):
    '''
    Testing the cutout function
    
    '''
    
    # first load polychrome
    polychromeR = pyf.open(par.wavecalDir + 'polychromeR%d.fits' % (par.R))
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
    
    im = Image(filename=fname)
    subim, psflet_subarr, [x0, x1, y0, y1] = get_cutout(im,xlist,ylist,psflets)
    print [x0, x1, y0, y1]
    Image(data=subim).write(par.unitTestsOutputs+'/cutout.fits')
    out = pyf.HDUList(pyf.PrimaryHDU(psflet_subarr.astype(np.float32)))
    out.writeto(par.unitTestsOutputs + '/psflet_cutout.fits', clobber=True)


def testFitCutout(par,fname,lensnum, mode='lstsq',ivar=False):
    '''
    Testing the fit_cutout function
    
    '''
    # first load polychrome
    polychromeR = pyf.open(par.wavecalDir + 'polychromeR%d.fits' % (par.R))
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
    
    im = Image(filename=fname)
    if ivar:
        im.ivar = im.data.copy()
        im.ivar = 1./im.ivar
    subim, psflet_subarr, [x0, x1, y0, y1] = get_cutout(im,xlist,ylist,psflets)
    return fit_cutout(subim, psflet_subarr, mode=mode)


#def testIntPixSol(par,fname)


def testGenPixSol(par):
    psftool = PSFLets()
    lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    psftool.geninterparray(lamlist, allcoef)
    psftool.genpixsol(lamlist,allcoef)
    psftool.savepixsol(outdir = par.exportDir)

def testCreateFlatfield(par,pixsize = 0.1, npix = 512, pixval = 1.,Nspec=45,outname='flatfield.fits'):
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
    
    '''
    
    lam_midpts,lam_endpts = calculateWaveList(par,Nspec=Nspec)
    inputCube = np.ones((len(lam_midpts),npix,npix),dtype=np.float32)
    for i in range(len(lam_midpts)):
        inputCube[i,:,:]*=pixval #/lam_midpts[i]
    
    par.saveDetector=False
    inCube = fits.HDUList(fits.PrimaryHDU(inputCube.astype(np.float32)))
    inCube[0].header['LAM_C'] = np.median(lam_midpts)/1000.
    inCube[0].header['PIXSIZE'] = pixsize
#     Image(data=inCube[0].data,header=inCube[0].header).write(par.unitTestsOutputs+'/inputcube.fits',clobber=True)
    #inCube.writeto(par.unitTestsOutputs+'/inputcube.fits', clobber=True)

#     detectorFrame = propagateIFS(par,lamlist/1000.,inCube[0])
    detectorFrame = polychromeIFS(par,lam_midpts,inCube[0],parallel=True,wavelist_endpts=lam_endpts)
    Image(data=detectorFrame,header=par.hdr).write(par.unitTestsOutputs+'/'+outname,clobber=True)
    
    
