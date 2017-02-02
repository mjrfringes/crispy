
import numpy as np
import tools
import logging as log
import matplotlib.pyplot as plt
from tools.image import Image
from params import Params
from astropy.io import fits as pyf
from tools.locate_psflets import PSFLets
from tools.reduction import get_cutout,fit_cutout
from IFS import propagateIFS
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

def testCreateFlatfield(par,lam1=605.,lam2=725.,nlam=26,parallel=False):

    # generate new wavelength array, with resolution R
#     lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]/1000.
    
    loglam_endpts = np.linspace(np.log(lam1), np.log(lam2), nlam)
    loglam_midpts = (loglam_endpts[1:] + loglam_endpts[:-1])/2
    loglam_width = (loglam_endpts[1:] - loglam_endpts[:-1])

    photon_flux_density_pr_lenslet = 0.2 # flux (W.m-2) per lenslet per nm
    lamlist=np.exp(loglam_midpts)
    #lamlist = [0.660]
    lamoD = 3. # number of lenslets per lamoD at 660nm
    mperpix = par.pitch/lamoD
    par.pixperlenslet = par.pitch/mperpix
    par.mperpix = mperpix
    inputCube = photon_flux_density_pr_lenslet*np.ones((len(lamlist),512,512),dtype=float)/lamoD**2
    for i in range(len(lamlist)):
        inputCube[i,:,:] *= np.exp(loglam_width[i])
    par.saveDetector=False
    detectorFrame = propagateIFS(par,lamlist/1000.,inputCube,parallel=parallel)
    Image(data=detectorFrame).write(par.unitTestsOutputs+'/flatfield.fits',clobber=True)
    
    
if __name__ == '__main__':
    testLoadKernels()
