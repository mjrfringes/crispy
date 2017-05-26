try:
    from astropy.io import fits as pyf
except:
    import pyfits as pyf

import numpy as np
from initLogger import getLogger
log = getLogger('crispy')
from scipy import signal
from scipy.interpolate import interp1d
from scipy import ndimage
from locate_psflets import PSFLets
from image import Image
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")

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

    Parameters
    ----------
    datacube: image instance
            containing 3D arrays data and ivar
    good:     2D array
            nonzero = good lenslet

    Returns
    -------
    datacube: input datacube modified in place

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

    Parameters
    ----------
    par:    Parameter instance
            Contains all IFS parameters
    name: string
            Name that will be given to final image, without fits extension
    ifsimage: Image instance of IFS detector map, with optional inverse variance
                    
    Returns
    -------
    cube :  3D array
            Return the reduced cube from the original IFS image


    '''
    log.info('Simple reduction')
    calCube = pyf.open(par.wavecalDir + 'polychromekeyR%d.fits' % (par.R))
    
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

    
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
    wavelengths = np.arange(lam_short,lam_long,par.dlam)
    cube = np.zeros((len(wavelengths),nlens,nlens))

    X = np.zeros((nlens,nlens),dtype='i4')
    Y = np.zeros((nlens,nlens),dtype='i4')
    
    for wav in range(len(wavelengths)):
        lam = wavelengths[wav]
        log.info('Wavelength = %3.1f' % (lam*1000.))
        for i in range(nlens):
            for j in range(nlens):
                if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
                    fx = interp1d(waveCalArray,xcenter[:,i,j])
                    fy = interp1d(waveCalArray,ycenter[:,i,j])
                    Y[j,i] = np.int(fx(lam))
                    X[j,i] = np.int(fy(lam))            
        cube[wav,:,:] = ifsimage[X,Y]+ifsimage[X,Y+1]+ \
            ifsimage[X,Y-1]+ifsimage[X,Y+2]+ifsimage[X,Y-2]
    
    cube[cube==0] = np.NaN
    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
    return cube


def densifiedSimpleReduction(par,name,ifsimage,ratio=10.):
    '''
    Use the same method as the 'simple' method but interpolate the original IFS map over
    a grid with finer sampling
    
    Parameters
    ----------
    par:    Parameter instance
            Contains all IFS parameters
    name: string
            Name that will be given to final image, without fits extension
    ifsimage: Image instance of IFS detector map, with optional inverse variance
    ratio:  int
            Ratio by which the original image is densified.
                    
    Returns
    -------
    cube :  3D array
            Return the reduced cube from the original IFS image
    
    '''
    calCube = pyf.open(par.wavecalDir + 'polychromekeyR%d.fits' % (par.R))
    
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
    ydim,xdim = ifsimageDense.shape
    xmargin = (ratio*5)//2
    ymargin = (ratio)//2
    xdim-=xmargin
    ydim-=ymargin
    xcenter[~((xcenter*ratio<xdim)*(ycenter*ratio<ydim)*(xcenter*ratio>xmargin)*(ycenter*ratio>ymargin))] = np.nan
    ycenter[~((xcenter*ratio<xdim)*(ycenter*ratio<ydim)*(xcenter*ratio>xmargin)*(ycenter*ratio>ymargin))] = np.nan

    
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
    wavelengths = np.arange(lam_short,lam_long,par.dlam)
    
    cube = np.zeros((len(wavelengths),nlens,nlens))

    X = np.zeros((nlens,nlens),dtype='i4')
    Y = np.zeros((nlens,nlens),dtype='i4')
    ratio = int(ratio)
    for wav in range(len(wavelengths)):
        lam = wavelengths[wav]
        log.info('Wavelength = %3.1f' % (lam*1000.))
        for i in range(nlens):
            for j in range(nlens):
                if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
                    fx = interp1d(waveCalArray,xcenter[:,i,j])
                    fy = interp1d(waveCalArray,ycenter[:,i,j])
                    Y[j,i] = np.floor(fx(lam)*ratio)
                    X[j,i] = np.floor(fy(lam)*ratio)
        for xcount in range(-ratio//2,-ratio//2+ratio):
            for ycount in range(-(ratio*5)//2,-(ratio*5)//2+ratio*5):
                cube[wav,:,:] += ifsimageDense[X+xcount,Y+ycount]
    
    cube[cube==0] = np.NaN
    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
    return cube


# def apertureReduction(par,name,ifsimage):
#     '''
#     Reduction using aperture photometry package from photutils
# 
#     Parameters
#     ----------
#     par:    Parameter instance
#     name: string
#             Name that will be given to final image, without fits extension
#     ifsimage: Image instance of IFS detector map, with optional inverse variance
#                     
#     Returns
#     -------
#     cube :  3D array
#             Return the reduced cube from the original IFS image
#     
#     '''
#     calCube = pyf.open(par.wavecalDir + 'polychromekeyR%d.fits' % (par.R))
#     
#     waveCalArray = calCube[0].data#wavecal[0,:,:]
#     waveCalArray = waveCalArray/1000.
#     
#     xcenter = calCube[1].data
#     nlens = xcenter.shape[1]
#     ycenter = calCube[2].data
#     good = calCube[3].data
#     ydim,xdim = ifsimage.shape
#     xdim-=2
#     xcenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan
#     ycenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan
# 
#     
#     lam_long = max(waveCalArray)
#     lam_short = min(waveCalArray)
#     wavelengths = np.arange(lam_short,lam_long,par.dlam)
#     cube = np.zeros((len(wavelengths),nlens,nlens))
# 
#     psftool = PSFLets()
#     lam = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
#     allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]
# 
#     # lam in nm
#     psftool.geninterparray(lam, allcoef)
#     
#     for wav in range(len(wavelengths)):
#         lam = wavelengths[wav]
#         log.info('Wavelength = %3.1f' % (lam*1000.))
#         for i in range(nlens):
#             for j in range(nlens):
#                 if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
#                     _x,_y = psftool.return_locations(lam*1000., allcoef, j-nlens/2, i-nlens/2)
#                     pos = (_x,_y)
#                     ap = RectangularAperture(pos,1,5,0)
#                     cube[wav,j,i] = aperture_photometry(ifsimage,ap)['aperture_sum'][0]
#     
#     cube[cube==0] = np.NaN
#     pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
#     return cube


def GPImethod2(par,name,ifsimage):
    '''
    Basic cube reduction using an IFS image and a wavecal cube
    Equivalent to method 2 in the IDL primitive 'pisces_assemble_spectral_datacube'

    Parameters
    ----------
    par:    Parameter instance
            Contains all IFS parameters
    name: string
            Name that will be given to final image, without fits extension
    ifsimage: Image instance of IFS detector map, with optional inverse variance
                    
    Returns
    -------
    cube :  3D array
            Return the reduced cube from the original IFS image


    '''
    log.info('GPI Method 2 reduction')
    calCube = pyf.open(par.wavecalDir + 'polychromekeyR%d.fits' % (par.R))
    
    waveCalArray = calCube[0].data
    waveCalArray = waveCalArray/1000.
    
    xcenter = calCube[1].data
    nlens = xcenter.shape[1]
    ycenter = calCube[2].data
    good = calCube[3].data
    ydim,xdim = ifsimage.shape
    xmargin = 5
    ymargin = 5
    xdim-=xmargin
    ydim-=ymargin
    xcenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>xmargin)*(ycenter>ymargin))] = np.nan
    ycenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>xmargin)*(ycenter>ymargin))] = np.nan

    sdpx = 26
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
    wavelengths = np.linspace(lam_short,lam_long,sdpx+1)
    cube = np.zeros((len(wavelengths),nlens,nlens))

    X = np.zeros((nlens,nlens))
    X0= np.zeros((nlens,nlens))
    Y = np.zeros((nlens,nlens))

        
    for wav in range(sdpx):
        lam = wavelengths[wav]
        log.info('Wavelength = %3.1f' % (lam*1000.))
        #index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-lam))
        #print index
        for i in range(nlens):
            for j in range(nlens):
                if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
                    if good[:,i,j].all():
                        fx = interp1d(waveCalArray,xcenter[:,i,j])
                        fy = interp1d(waveCalArray,ycenter[:,i,j])
                        Y[j,i] = fx(wavelengths[wav+1])
                        X0[j,i] = fy(wavelengths[wav])
                        X[j,i] = fy(wavelengths[wav+1])


        x_long = np.floor(X)
#             ;Y_long[np.where(Y_long == floor(!np.nan))] = np.nan
        x_long[np.where(X > 10000)] = np.nan
        x_long[np.where(x_long < -10000)] = np.nan
        x_long_rest = (X-np.floor(X)) #abs(Y_pix_arr - round(Y_pix_arr+0.5))
        x_short = np.floor(X0)
        x_short[np.where(x_short > 10000)] = np.nan
        x_short[np.where(x_short < -10000)] = np.nan
        x_short_rest = (x_long-X0) #abs(Y_long - Yo_pix_arr) ;- floor(Yo_pix_arr)

        index = np.where(x_long_rest < 0)
        plot_number = x_long_rest.shape
        for k in range(plot_number[0] - 1):
            for l in range(plot_number[1] - 1):
                if (x_long_rest[k,l] < 0):
                    x_short_rest[k,l] = 0.
                if (x_long_rest[k,l] < 0):
                    yo_pix_arr = floor(Y[k,l])
                    x_long_rest[k,l] = Yo_pix_arr[k,l] - Y[k,l]  #! Yo_pix_arr nt defined! equivalent to y spacing - just lam?, defined yo_pix_arr here as defined later

        #Fix the pixels that are longer than one pixel in the current spectral channel
        x_middle = x_short - 1.0
        x_middle_rest = np.zeros((nlens,nlens))
        x_middle_rest[np.where(x_long_rest > 1)] = 1.0#Y_long_rest[where(Y_long_rest gt 1)]
        x_long_rest[np.where(x_long_rest > 1)] = x_long_rest[np.where(x_long_rest > 1)] - 1
  
        diff = (Y - np.floor(Y)) - 0.5
        yo_pix_arr = np.floor(Y)
        east1 = np.zeros((nlens,nlens))
        east2 = np.zeros((nlens,nlens))
        west1 = np.zeros((nlens,nlens))
        west2 = np.zeros((nlens,nlens))
  
        east1[np.where(diff >= 0)] = 1.0
        east1[np.where(diff < 0)] = 1.0 + diff[np.where(diff < 0)]
        east2[np.where(diff >= 0)] = diff[np.where(diff >= 0)]
        west1[np.where(diff >= 0)] = 1.0 - diff[np.where(diff >= 0)]
        west1[np.where(diff < 0)] = 1.0
        west2[np.where(diff < 0)] = - diff[np.where(diff < 0)]
        x_long_rest[np.where(x_long_rest < 0)] = 0
        x_short_rest[np.where(x_short_rest < 0)] = 0
        x_middle_rest[np.where(x_middle_rest < 0)] = 0
        
        yo_pix_arr = yo_pix_arr.astype(int)
        x_long = x_long.astype(int)
        x_short = x_short.astype(int)
        x_middle = x_middle.astype(int)
        cube[wav,:,:]=(ifsimage[x_long,yo_pix_arr] + ifsimage[x_long,yo_pix_arr+1] + ifsimage[x_long,yo_pix_arr-1] + east1*ifsimage[x_long,yo_pix_arr+2] + west1*ifsimage[x_long,yo_pix_arr-2]    +   east2*ifsimage[x_long,yo_pix_arr+3] + west2*ifsimage[x_long,yo_pix_arr-3]) * x_long_rest + (ifsimage[x_short,yo_pix_arr] + ifsimage[x_short,yo_pix_arr+1] + ifsimage[x_short,yo_pix_arr-1] + east1*ifsimage[x_short,yo_pix_arr+2] + west1*ifsimage[x_short,yo_pix_arr-2]   +    east2*ifsimage[x_short,yo_pix_arr+3] + west2*ifsimage[x_short,yo_pix_arr-3]) * x_short_rest + (ifsimage[x_middle,yo_pix_arr] + ifsimage[x_middle,yo_pix_arr+1] + ifsimage[x_middle,yo_pix_arr-1] + east1*ifsimage[x_middle,yo_pix_arr+2] + west1*ifsimage[x_middle,yo_pix_arr-2]    +   east2*ifsimage[x_middle,yo_pix_arr+3] + west2*ifsimage[x_middle,yo_pix_arr-3]) * x_middle_rest

  ### cc not declared - need to figure this out before testing for off pixel lenslet          
  ###
  ###
  #declare as NaN mlens not on the detector (or on the reference pixel area, i.e. 4 pixels on each side):
  #bordy= np.where(~finite(Y) OR (Round(Y) < 4.0) OR (Round(Y) > 1019.0),cc) ;: size of the detector
  #if (cc ne 0) then cubef[bordy]=!VALUES.F_NAN
  #;; we expand the border region by 2 pixels in X, so that we flag as NaN
  #;; any 3x1 pixel box that has at least one pixel off the edge...
  #bordx=where(~finite(x3) OR (x3 LT 6.0) OR (x3 GT 1017.0),cc) ;: size of the detector
  #if (cc ne 0) then cubef[bordx]=!VALUES.F_NAN


        Yo_pix_arr=Y #?
    cube[cube==0] = np.NaN
    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)



def testReduction(par,name,ifsimage):
    '''
    Scratch routine to test various things.

    Parameters
    ----------
    par:    Parameter instance
            Contains all IFS parameters
    name: string
            Name that will be given to final image, without fits extension
    ifsimage: Image instance of IFS detector map, with optional inverse variance
                    
    Returns
    -------
    cube :  3D array
            Return the reduced cube from the original IFS image
    
    '''
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

def calculateWaveList(par,lam_list=None,Nspec=None,method='lstsq'):

    '''
    Computes the wavelength lists corresponding to the center and endpoints of each
    spectral bin. Wavelengths are separated by a constant value in log space. Number of
    wavelengths depends on spectral resolution.
    
    Parameters
    ----------
    par:    Parameter instance
            Contains all IFS parameters
    lam_list:   list of floats
            Usually this is left to None. If so, we use the wavelengths used for wavelength
            calibration. Otherwise, we could decide to focus on a smaller/larger region of
            the spectrum to retrieve. The final processed cubes will have bins centered
            on lam_midpts
    Nspec: int
            If specified, forces the number of bins in the final cube (uses np.linspace)
              
    Returns
    -------
    lam_midpts: list of floats
            Wavelengths at the midpoint of each bin
    lam_endpts: list of floats
            Wavelengths at the edges of each bin
    '''
    if lam_list is None:
        lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    else:
        lamlist = lam_list
    if Nspec is None:
        if method=='lstsq':
            Nspec = int(np.log(max(lamlist)/min(lamlist))*par.R*par.nspecperpix_lstsq+2)
        else:
            Nspec = int(np.log(max(lamlist)/min(lamlist))*par.R*par.npixperdlam+2)
    log.info('Reduced cube will have %d wavelength bins' % (Nspec-1))
    lam_endpts = np.linspace(min(lamlist), max(lamlist), Nspec)
    lam_midpts = (lam_endpts[1:]+lam_endpts[:-1])/2.
    return lam_midpts,lam_endpts

def lstsqExtract(par,name,ifsimage,smoothandmask=True,ivar=True,dy=3,refine=False):
    '''
    Least squares extraction, inspired by T. Brandt and making use of some of his code.
    
    Parameters
    ----------
    par:    Parameter instance
            Contains all IFS parameters
    name: string
            Name that will be given to final image, without fits extension
    ifsimage: Image
            Image instance of IFS detector map, with optional inverse variance
                    
    Returns
    -------
    cube :  3D array
            Return the reduced cube from the original IFS image

    '''
    polychromeR = pyf.open(par.wavecalDir + 'polychromeR%d.fits' % (par.R))
    psflets = polychromeR[0].data
    
    psftool = PSFLets()
    lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]

    # lam in nm
    psftool.geninterparray(lamlist, allcoef)
    
#     n_add = 1
#     psflets = _add_row(psflets, n=n_add, dtype=np.float64)
#     psflets[-n_add:] = 0
#     psflets[-1, 4:-4, 4:-4] = 1

    
    polychromekey = pyf.open(par.wavecalDir + 'polychromekeyR%d.fits' % (par.R))
    lams = polychromekey[0].data
    xindx = polychromekey[1].data
    yindx = polychromekey[2].data
    good = polychromekey[3].data
    
    if ivar:
        if ifsimage.ivar is None:
            ifsimage.ivar = np.ones(ifsimage.data.shape)
    else:
        ifsimage.ivar = None
        
    cube = np.zeros((psflets.shape[0],par.nlens,par.nlens))
    ivarcube = np.zeros((psflets.shape[0],par.nlens,par.nlens))

    resid = np.empty(ifsimage.data.shape)
    resid = ifsimage.data.copy()
    
    
    ydim,xdim = ifsimage.data.shape
    for i in range(par.nlens):
        for j in range(par.nlens):
#             xlist = []
#             ylist = []
#             good = True
#             for lam in lams:
#                 _x,_y = psftool.return_locations(lam, allcoef, j-par.nlens//2, i-par.nlens//2)
#                 good *= (_x > dy)*(_x < xdim-dy)*(_y > dy)*(_y < ydim-dy)
#                 xlist += [_x]    
#                 ylist += [_y]   
            
            if np.prod(good[:,i,j],axis=0):
                subim, psflet_subarr, [y0, y1, x0, x1] = get_cutout(ifsimage,xindx[:,i,j],yindx[:,i,j],psflets,dy)
                cube[:,j,i] = fit_cutout(subim.copy(), psflet_subarr.copy(), mode='lstsq')
                ivarcube[:,i,j] = 1.
            else:
                cube[:,j,i] = np.NaN
                ivarcube[:,i,j] = 0.
    
    xindx = np.arange(-par.nlens/2, par.nlens/2)
    xindx, yindx = np.meshgrid(xindx, xindx)
    lam_midpts,lam_endpts = calculateWaveList(par)
    for i in range(len(psflets)):
        ydim,xdim=ifsimage.data.shape
        _x,_y = psftool.return_locations(lam_midpts[i], allcoef, xindx, yindx)
        good = (_x > dy)*(_x < xdim-dy)*(_y > dy)*(_y < ydim-dy)
        psflet_indx = _tag_psflets(ifsimage.data.shape, _x, _y, good,dy=10)
        coefs_flat = np.reshape(cube[i].transpose(), -1)
        resid -= psflets[i]*coefs_flat[psflet_indx]

    if 'cubemode' not in par.hdr:
        par.hdr.append(('cubemode','Least squares', 'Method used to extract data cube'), end=True)
        par.hdr.append(('lam_min',np.amin(lams), 'Minimum mid wavelength of extracted cube'), end=True)
        par.hdr.append(('lam_max',np.amax(lams), 'Maximum mid wavelength of extracted cube'), end=True)
        par.hdr.append(('dlam',(np.amax(lams)-np.amin(lams))/lams.shape[0], 'Spacing of extracted wavelength bins'), end=True)
        par.hdr.append(('nlam',lams.shape[0], 'Number of extracted wavelengths'), end=True)

    if hasattr(par,'lenslet_flat'):
        lenslet_flat = pyf.open(par.lenslet_flat)[1].data
        lenslet_flat = lenslet_flat[np.newaxis,:]
        if "FLAT" not in par.hdr:
            par.hdr.append(('FLAT',True, 'Applied lenslet flatfield'), end=True)
        cube *= lenslet_flat
        ivarcube /= lenslet_flat**2
    else:
        lenslet_flat = np.ones(cube.shape)
    if hasattr(par,'lenslet_mask'):
        if "MASK" not in par.hdr:
            par.hdr.append(('MASK',True, 'Applied lenslet mask'), end=True)
        lenslet_mask = pyf.open(par.lenslet_mask)[1].data
        lenslet_mask = lenslet_mask[np.newaxis,:]
        ivarcube *= lenslet_mask
    else:
        lenslet_mask = np.ones(cube.shape)
    
    
    if 'SMOOTHED' not in par.hdr:
        par.hdr.append(('SMOOTHED',smoothandmask, 'Cube smoothed over bad lenslets'), end=True)
    if smoothandmask:
        cube = Image(data=cube*lenslet_mask,ivar=ivarcube)
        #good = np.any(cube.data != 0, axis=0)
        cube = _smoothandmask(cube, np.ones(good.shape))
    else:
        cube = Image(data=cube,ivar=ivarcube)


#     if smoothandmask:
#         par.hdr.append(('SMOOTHED',True, 'Cube smoothed over bad lenslets'), end=True)
#         cube = Image(data=cube,ivar=None)
#         good = np.any(cube.data != 0, axis=0)
#         cube = _smoothandmask(cube, good)
#     else:
#         par.hdr.append(('SMOOTHED',False, 'Cube NOT smoothed over bad lenslets'), end=True)
#         cube = Image(data=cube,ivar=None)

    #cube = Image(data=cube.data,ivar=cube.ivar,header=par.hdr,extraheader=ifsimage.extraheader)


    #pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
    #pyf.PrimaryHDU(resid).writeto(name+'_resid.fits',clobber=True)
    Image(data=cube.data,ivar=ivarcube,header=par.hdr,extraheader=ifsimage.extraheader).write(name+'.fits',clobber=True)
    Image(data=resid,header=par.hdr,extraheader=ifsimage.extraheader).write(name+'_resid.fits',clobber=True)
#     pyf.PrimaryHDU(newresid).writeto(name+'_newresid.fits',clobber=True)
    return cube


def _add_row(arr, n=1, dtype=None):
    """

    """

    if n < 1:
        return arr
    newshape = list(arr.shape)
    newshape[0] += n
    if dtype is None:
        outarr = np.zeros(tuple(newshape), arr.dtype)
    else:
        outarr = np.zeros(tuple(newshape), dtype)
    outarr[:-n] = arr
    meanval = (arr[0] + arr[-1])/2
    for i in range(1, n + 1):
        outarr[-i] = meanval
    return outarr



def get_cutout(im, x, y, psflets, dy=3):
    
    """
    Cut out a microspectrum for fitting.  Return the inputs to 
    linalg.lstsq or to whatever regularization scheme we adopt.
    Assumes that spectra are dispersed in the -y direction.

    Parameters
    ----------
    im: Image intance
            Image containing data to be fit
    x: float
            List of x centroids for each microspectrum
    y: float
            List of y centroids for each microspectrum
    psflets: PSFLet instance
            Typically generated from polychrome step in wavelength calibration routine
    dy: int
            vertical length to cut out, default 3.  This is the length to cut out in the
            +/-y direction; the lengths cut out in the +x direction (beyond the shortest 
            and longest wavelengths) are also dy.

    Returns
    -------
    subim:  2D array
            A flattened subimage to be fit
    psflet_subarr: 2D ndarray
            first dimension is wavelength, second dimension is spatial, and is the same 
            shape as the flattened subimage.

    Notes
    -----
    Both subim and psflet_subarr are scaled by the inverse
    standard deviation if it is given for the input Image.  This 
    will make the fit chi2 and properly handle bad/masked pixels.

    """

    x0, x1 = [int(np.amin(x)) - dy+1, int(np.amax(x)) + dy+1]
    y0, y1 = [int(np.amin(y)) - dy+1, int(np.amax(y)) + dy+1]

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

    Parameters
    ----------
    subim:   2D nadarray
        Microspectrum to fit
    psflets: 3D ndarray
        First dimension is wavelength.  psflets[0] 
                must match the shape of subim.
    mode:    string
        Method to use.  Currently limited to lstsq (a 
                simple least-squares fit using linalg.lstsq), this can
                be expanded to include an arbitrary approach.
    
    Returns
    -------
    coef:    array
        The best-fit coefficients (i.e. the microspectrum).

    Notes
    -----
    This routine may also return the covariance matrix in the future.
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
#     elif mode == 'ext':
#         coef = np.zeros(psflets.shape[0])
#         for i in range(psflets.shape[0]):
#             coef[i] = np.sum(psflets[i]*subim)/np.sum(psflets[i])
#     elif mode == 'apphot':
#         coef = np.zeros((subim.shape[1]))
#         for i in range(subim.shape[1]):
#             coef[i] = np.sum(subim[:,i])
    else:
        raise ValueError("mode " + mode + " to fit microspectra is not currently implemented.")

    return coef

def _tag_psflets(shape, x, y, good, dx=10, dy=10):
    """
    Create an array with the index of each lenslet at a given
    wavelength.  This will make it very easy to remove the best-fit
    spectra accounting for nearest-neighbor crosstalk.

    Parameters
    ----------
    shape:  tuple
        Shape of the image and psflet arrays.
    x:      ndarray
        x indices of the PSFlet centroids at a given wavelength
    y:      ndarray
        y indices of the PSFlet centroids
    good:  boolean ndarray
        True if the PSFlet falls on the detector

    Returns
    -------
    psflet_indx: ndarray
        Has the requested input shape (matching the PSFlet image). 
        The array contains the indices of the closest lenslet to each
        pixel at the wavelength of x and y

    Notes
    -----
    The output, psflet_indx, is to be used as follows:
    coefs[psflet_indx] will give the scaling of the monochromatic PSF-let
    frame.

    """

    psflet_indx = np.zeros(shape, np.int)
    oldshape = x.shape
    x_int = (np.reshape(x + 0.5, -1)).astype(int)
    y_int = (np.reshape(y + 0.5, -1)).astype(int)
    
    good = np.reshape(good, -1)
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)

    x_i = np.arange(shape[1])
    y_i = np.arange(shape[0])
    x_i, y_i = np.meshgrid(x_i, y_i)

    mindist = np.ones(shape)*1e10

    for i in range(x_int.shape[0]):
        if good[i]:
            #psflet_indx[y_int[i] - 6:y_int[i] + 7, x_int[i] - 6:x_int[i] + 7] = i
            iy1, iy2 = [y_int[i] - dy, y_int[i] + dy + 1]
            ix1, ix2 = [x_int[i] - dx, x_int[i] + dx + 1]

            dist = (y[i] - y_i[iy1:iy2, ix1:ix2])**2 
            dist += (x[i] - x_i[iy1:iy2, ix1:ix2])**2
            indx = np.where(dist < mindist[iy1:iy2, ix1:ix2])

            psflet_indx[iy1:iy2, ix1:ix2][indx] = i
            mindist[iy1:iy2, ix1:ix2][indx] = dist[indx]
            
    good = np.reshape(good, oldshape)
    x = np.reshape(x, oldshape)
    y = np.reshape(y, oldshape)

    return psflet_indx




def intOptimalExtract(par,name,IFSimage,smoothandmask=True):
    """
    Calls the optimal extraction routine
    
    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    name: string
            Path & name of the output file
    IFSimage: Image instance 
            Image instance of input image. Can have a .ivar field for a variance map.

    Return
    ------
    datacube.data:  3D ndarray
            Datacube of reduced IFS image. The corresponding wavelengths can be found in
            using calculateWaveList(par)
            
    Notes
    -----
    A cube is also written at par.SimResults/name.fits
    
    """

    loc = PSFLets(load=True, infiledir=par.wavecalDir)
    #Nspec = int(par.BW*par.npixperdlam*par.R)
    lam_midpts,scratch = calculateWaveList(par,method='optext')

    
    datacube = fitspec_intpix_np(par,IFSimage, loc, lam_midpts,smoothandmask=smoothandmask)
    datacube.write(name+'.fits',clobber=True)
    return datacube

def fitspec_intpix(par,im, PSFlet_tool, lamlist,  delt_y=6, flat=None, 
                   mode = 'gaussvar'):
    """
    Optimal extraction routine
    
    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    im:     Image instance
            IFS image to be processed.
    PSFlet_tool: PSFLet instance
            Inverse wavelength solution that is constructed during wavelength calibration
    lamlist: list of floats
            List of wavelengths to which each microspectrum is interpolated.    
    delt_y: int
            Width in pixels of each microspectrum in the cross-dispersion direction       
    flat:
            Whether a lenslet flatfield is used (not implemented yet)
    smoothandmask: Boolean
            Whether to smooth and mask bad pixels
    
    Returns
    -------
    image:  Image instance
            Reduced cube in the image.data field
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
                iy = np.nanmean(y)
                if ~np.isnan(iy):
                    lams = PSFlet_tool.lam_indx[i,j,:PSFlet_tool.nlam[i, j]]
                    i1 = int(iy) - delt_y/2.
                    arr = np.arange(i1,i1 + delt_y)
                    dy = arr-iy
                    if mode=='sum':
#                     gaussian = np.exp(-dy**2/eff_sig**2/2.)
#                     weights = np.sum(gaussian**2)
                        pix_center_vals = [np.sum(im.data[i1:i1 + delt_y, val],axis=0) for val in ix]
                    elif mode=='gaussvar':
                        weights = np.array([np.sum((np.exp(-dy**2/(sig*wav/par.FWHMlam)**2/2.))**2) for wav in lams])
                        pix_center_vals = np.array([np.sum(im.data[i1:i1 + delt_y, ix[ii]]*np.exp(-dy**2/(sig*lams[ii]/par.FWHMlam)**2/2.)) for ii in range(PSFlet_tool.nlam[i, j])])/weights
                    elif mode=='gaussnovar':
                        weights = np.array([np.sum((np.exp(-dy**2/(sig)**2/2.))**2) for lam in lams])
                        pix_center_vals = np.array([np.sum(im.data[i1:i1 + delt_y, ix[ii]]*np.exp(-dy**2/(sig)**2/2.)) for ii in range(PSFlet_tool.nlam[i, j])])/weights
                    tck = interpolate.splrep(np.log(lams), pix_center_vals, s=0, k=3)
                    cube[:,j,i] = interpolate.splev(loglam, tck, ext=1)

#                     func = interp1d(lams,pix_center_vals,kind='linear')
#                     cube[:,j,i] = func(lamlist)
                else:
                    cube[:,j,i] = np.NaN
            else:
                cube[:,j,i] = np.NaN
# 
#     if flat is not None:
#         datacube.data /= flat + 1e-10
#         datacube.ivar *= flat**2
# 
    if smoothandmask:
        good = np.any(cube.data != 0, axis=0)
        datacube = _smoothandmask(cube, good)
        
    par.hdr.append(('cubemode','Optimal Extraction', 'Method used to extract data cube'), end=True)
    par.hdr.append(('lam_min',np.amin(lamlist), 'Minimum (central) wavelength of extracted cube'), end=True)
    par.hdr.append(('lam_max',np.amax(lamlist), 'Maximum (central) wavelength of extracted cube'), end=True)
    par.hdr.append(('dloglam',loglam[1]-loglam[0], 'Log spacing of extracted wavelength bins'), end=True)
    par.hdr.append(('nlam',lamlist.shape[0], 'Number of extracted wavelengths'), end=True)

    return Image(data=cube,header=par.hdr,extraheader=im.extraheader)




def fitspec_intpix_np(par,im, PSFlet_tool, lamlist,smoothandmask=True, delt_y=6):
    """
    Original optimal extraction routine in Numpy from T. Brand
    
    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    im:     Image instance
            IFS image to be processed.
    PSFlet_tool: PSFLet instance
            Inverse wavelength solution that is constructed during wavelength calibration
    lamlist: list of floats
            List of wavelengths to which each microspectrum is interpolated.    
    delt_y: int
            Width in pixels of each microspectrum in the cross-dispersion direction       
    
    Returns
    -------
    image:  Image instance
            Reduced cube in the image.data field
    """

    xindx = PSFlet_tool.xindx
    yindx = PSFlet_tool.yindx
    Nmax = PSFlet_tool.nlam_max

    x = np.arange(im.data.shape[1])
    y = np.arange(im.data.shape[0])
    x, y = np.meshgrid(x, y)

    ydim,xdim = im.data.shape

    coefs = np.zeros(tuple([max(Nmax, lamlist.shape[0])] + list(yindx.shape)[:-1]))
    cube = np.zeros((len(lamlist),par.nlens,par.nlens))    
    ivarcube = np.zeros((len(lamlist),par.nlens,par.nlens))    
    xarr, yarr = np.meshgrid(np.arange(Nmax),np.arange(delt_y))

    #loglam = np.log(lamlist)
    lamsol = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
    allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]
    PSFlet_tool.geninterparray(lamsol, allcoef)
    
    #polychromekey = pyf.open(par.wavecalDir + 'polychromekeyR%d.fits' % (par.R))
#     lams = polychromekey[0].data
#     xindx = polychromekey[1].data+0.5
#     yindx = polychromekey[2].data+0.5
#     good = polychromekey[3].data
    good = PSFlet_tool.good

    
    for i in range(xindx.shape[0]):
        for j in range(yindx.shape[1]):
#             good = True
#             for lam in lamlist:
#                 _x,_y = PSFlet_tool.return_locations(lam, allcoef, j-par.nlens//2, i-par.nlens//2)
#                 good *= (_x > delt_y)*(_x < xdim-delt_y)*(_y > delt_y)*(_y < ydim-delt_y)
#                  
#             if good:
            if good[i,j]:
                _x = xindx[i, j, :PSFlet_tool.nlam[i, j]]
                _y = yindx[i, j, :PSFlet_tool.nlam[i, j]]
                _lam = PSFlet_tool.lam_indx[i, j, :PSFlet_tool.nlam[i, j]]
                iy = np.nanmean(_y)
                if ~np.isnan(iy):
                    i1 = int(iy - delt_y/2.)
                    dy = _y[xarr[:,:len(_lam)]] - y[i1:i1 + delt_y,int(_x[0]):int(_x[-1]) + 1]
                    lams,tmp = np.meshgrid(_lam,np.arange(delt_y))
                    
                    sig = par.FWHM/2.35*lams/par.FWHMlam
                    weight = np.exp(-dy**2/2./sig**2)/sig/np.sqrt(2.*np.pi)
                    data = im.data[i1:i1 + delt_y,int(_x[0]):int(_x[-1]) + 1]
                    
                    if im.ivar is not None:
                        ivar = im.ivar[i1:i1 + delt_y,int(_x[0]):int(_x[-1]) + 1]
                    else:
                        ivar = np.ones(data.shape)

                    coefs[:len(_lam), i, j] = np.sum(weight*data*ivar, axis=0)
                    coefs[:len(_lam), i, j] /= np.sum(weight**2*ivar, axis=0)
                    tck = interpolate.splrep(_lam, coefs[:len(_lam), i, j], s=0, k=3)
                    cube[:,j,i] = interpolate.splev(lamlist, tck, ext=1)
                    tck = interpolate.splrep(_lam, np.sum(weight**2*ivar, axis=0)/np.sum(weight**2, axis=0), s=0, k=3)
                    ivarcube[:,j,i] = interpolate.splev(lamlist, tck, ext=1)
                else:
                    cube[:,j,i] = np.NaN
                    ivarcube[:,j,i] = 0.
            else:
                cube[:,j,i] = np.NaN
                ivarcube[:,j,i] = 0.
                
    if 'cubemode' not in par.hdr:
        par.hdr.append(('cubemode','Optimal Extraction', 'Method used to extract data cube'), end=True)
        par.hdr.append(('lam_min',np.amin(lamlist), 'Minimum mid wavelength of extracted cube'), end=True)
        par.hdr.append(('lam_max',np.amax(lamlist), 'Maximum mid wavelength of extracted cube'), end=True)
        par.hdr.append(('dlam',lamlist[1]-lamlist[0], 'Spacing of extracted wavelength bins'), end=True)
        par.hdr.append(('nlam',lamlist.shape[0], 'Number of extracted wavelengths'), end=True)
    
    if hasattr(par,'lenslet_flat'):
        lenslet_flat = pyf.open(par.lenslet_flat)[1].data
        lenslet_flat = lenslet_flat[np.newaxis,:]
        if "FLAT" not in par.hdr:
            par.hdr.append(('FLAT',True, 'Applied lenslet flatfield'), end=True)
        cube *= lenslet_flat
        ivarcube /= lenslet_flat**2
    else:
        lenslet_flat = np.ones(cube.shape)
    if hasattr(par,'lenslet_mask'):
        if "MASK" not in par.hdr:
            par.hdr.append(('MASK',True, 'Applied lenslet mask'), end=True)
        lenslet_mask = pyf.open(par.lenslet_mask)[1].data
        lenslet_mask = lenslet_mask[np.newaxis,:]
        ivarcube *= lenslet_mask
    else:
        lenslet_mask = np.ones(cube.shape)
    
    
    if smoothandmask:
        if 'SMOOTHED' not in par.hdr:
            par.hdr.append(('SMOOTHED',True, 'Cube smoothed over bad lenslets'), end=True)
        cube = Image(data=cube*lenslet_mask,ivar=ivarcube)
        #good = np.any(cube.data != 0, axis=0)
        cube = _smoothandmask(cube, np.ones(good.shape))
    else:
        if 'SMOOTHED' not in par.hdr:
            par.hdr.append(('SMOOTHED',False, 'Cube NOT smoothed over bad lenslets'), end=True)
        cube = Image(data=cube,ivar=ivarcube)

    cube = Image(data=cube.data,ivar=cube.ivar,header=par.hdr,extraheader=im.extraheader)

    return cube

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
