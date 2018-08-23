#!/usr/bin/env python

import numpy as np
from scipy import interpolate, ndimage, signal
import glob
try:
    from astropy.io import fits as pyf
except BaseException:
    import pyfits as pyf

import re
from crispy.tools.initLogger import getLogger
log = getLogger('crispy')
import matplotlib.pyplot as plt
import codecs


def distort(fx, fy, lam):
    '''
    Apply the distortion and dispersion from pre-determined polynomial
    This was estimated from Zemax by Qian Gong and Jorge Llop and needs to be revisited

    Parameters
    ----------
    fx,fy :   float
            Distance between a lenslet and the center of the detector in millimeter.
    lam : float
            Wavelength in microns.

    Returns
    -------
    x,y : float
            Distance from center of detector at which image from a lenslet falls in mm.

    '''
    cx = np.zeros(20)
    cy = np.zeros(20)
    cx[0] = -0.00000000036463819
    cx[1] = -0.96764187
    cx[2] = 2.9532635E-14
    cx[3] = 0.0000000016635338
    cx[4] = -2.0986347E-15
    cx[5] = 0.000117021
    cx[6] = -2.2672132E-15
    cx[7] = -0.063628939
    cx[8] = -8.187448500000000E-14
    cx[9] = -2.524745200000000E-09
    cx[10] = -3.346803500000000E-04
    cx[11] = 3.312787500000000E-17
    cx[12] = -3.582555400000000E-04
    cx[13] = 2.204885100000000E-17
    cx[14] = 3.043899600000000E-15
    cx[15] = -1.187753400000000E-04
    cx[16] = 3.301599300000000E-15
    cx[17] = 4.349654500000000E-02
    cx[18] = 5.381647600000000E-14
    cx[19] = 1.274761900000000E-09
    cy[0] = -2.962499600000000E+00
    cy[1] = -7.983890700000000E-14
    cy[2] = -9.907069600000000E-01
    cy[3] = 6.343124200000000E+00
    cy[4] = -2.579091100000000E-03
    cy[5] = -5.548179600000000E-16
    cy[6] = -3.231052700000000E-03
    cy[7] = 2.416302700000000E-13
    cy[8] = -2.116432700000000E-02
    cy[9] = -2.979901200000000E+00
    cy[10] = 8.941723000000000E-18
    cy[11] = -3.690345100000000E-04
    cy[12] = 1.272463800000000E-17
    cy[13] = -3.495699500000000E-04
    cy[14] = -8.848836700000000E-05
    cy[15] = 7.928802600000000E-16
    cy[16] = -9.424257500000000E-05
    cy[17] = -1.844725700000000E-13
    cy[18] = 2.163655100000000E-02
    cy[19] = 2.627917300000000E-01

    staticArray = np.array([1.,
                            fx,
                            fy,
                            lam,
                            fx**2.,
                            fx * fy,
                            fy**2.,
                            fx * lam,
                            fy * lam,
                            lam**2.,
                            fx**3.,
                            fx**2. * fy,
                            fx * fy**2.,
                            fy**3.,
                            fx**2. * lam,
                            fx * fy * lam,
                            fy**2. * lam,
                            fx * lam**2.,
                            fy * lam**2.,
                            lam**3.])

    x = np.sum(cx * staticArray)
    y = np.sum(cy * staticArray)
    return x, y


def createAllWeightsArray(par, locations):
    '''
    Creates weights for bilinear interpolation

    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    locations : 2D ndarray, Nx2
            Array of normalized locations on the detector, .

    Returns
    -------
    detectorFrame : 2D array
            Return the detector frame with correct pixel scale.

    '''

    # requires square array for now
    npix = par.nlens
    xfrac = np.linspace(0, npix, npix) / npix
    yfrac, xfrac = np.meshgrid(xfrac, xfrac)

    allweights = np.ones((npix, npix, len(locations))) * 1 / 0.25
    for i in range(len(locations)):
        allweights[:, :, i] *= (np.abs(xfrac - locations[i, 0]) < 0.5)
        allweights[:, :, i] *= (np.abs(yfrac - locations[i, 1]) < 0.5)

    for i in range(npix):
        for k in range(len(locations)):
            if xfrac[i, 0] > locations[k, 0]:
                allweights[i, :, k] *= locations[k, 0] + 0.5 - xfrac[i, :]
            else:
                allweights[i, :, k] *= xfrac[i, :] - (locations[k, 0] - 0.5)

    for j in range(npix):
        for k in range(len(locations)):
            if yfrac[0, j] > locations[k, 1]:
                allweights[:, j, k] *= locations[k, 1] + 0.5 - yfrac[:, j]
            else:
                allweights[:, j, k] *= yfrac[:, j] - (locations[k, 1] - 0.5)
    return allweights


def selectKernel(par, lam, refWaveList, kernelList):
    '''
    Select the correct kernel for the current wavlength

    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    lam : float
            Wavelength at which we want the get the kernel, in microns
    refWaveList : list of floats
            Wavelengths at which the kernels are defined
    kernelList : list of 3D ndarrays
            List of the kernels cubes at the wavelengths above

    Returns
    -------
    kernels: array of 2D arrays
            Represents each 2D kernel at each location for that wavlength
    locations: Nx2 ndarray
            Location coordinates in detector position ratio (0,0) is bottom left, (1,1) is
            top right
    '''

    # find lowest index that is greater than lam
    ind = 0
    for nwave in range(len(refWaveList)):
        if refWaveList[ind] >= lam * 1000:
            break
        else:
            ind += 1

    if ind == 0:
        kernels = kernelList[0]
        log.warning('Wavelength out of range of reference kernels')
    elif ind == len(refWaveList):
        kernels = kernelList[-1]
        log.warning('Wavelength out of range of reference kernels')
    else:
        # ind is the index of the wavelength immediately greater than lam
        wavelUp = refWaveList[ind]
        wavelDown = refWaveList[ind - 1]
        kernels = (wavelUp - lam * 1000) * \
            kernelList[ind] + (lam * 1000 - wavelDown) * kernelList[ind - 1]
        kernels /= (wavelUp - wavelDown)

    if par.convolve:
        newkernels = np.zeros(kernels.shape)
        # convolve the clipped kernel by a Gaussian to simulate defocus
        # this is inaccurate but is a placeholder for using the real defocussed kernels
        # scale in kernel is (par.pixsize/par.pxperdetpix)
        # we want kernel to be ~2 detector pixel FWHM so
        # par.pixsize/(par.pixsize/par.pxperdetpix)
        sigma = par.FWHM / 2.35 * par.pxperdetpix
        for k in range(kernels.shape[0]):
            newkernels[k] = ndimage.filters.gaussian_filter(
                kernels[k], sigma, order=0, mode='constant')
    if par.gaussian:
        sigma = par.FWHM / 2.35 * par.pxperdetpix
        for k in range(kernels.shape[0]):
            x = np.arange(kernels[0].shape[0]) - kernels[0].shape[0] / 2
            _x, _y = np.meshgrid(x, x)
            newkernels[k] = np.exp(-(_x**2 + _y**2) /
                                   (2 * (sigma * lam * 1000 / par.FWHMlam)**2))
            newkernels[k] /= np.sum(newkernels[k])
    return newkernels


def loadKernels(par, wavel):
    '''
    Loads the kernels that represent the PSFs at different locations on the detector

    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    wavel : float
            Wavelength at which the kernels are needed

    Returns
    -------
    kernels: array of 2D arrays
            Represents each 2D kernel at each location
    locations: Nx2 ndarray
            Location coordinates in detector position ratio (0,0) is bottom left, (1,1) is
            top right
    '''

    log.info('Loading spot diagrams.')
    # first, select which wavelength PSF to use
    spotfields = glob.glob(par.prefix + '/simpsf/%dPSF_*.fits' % wavel)
    spotsizefiles = glob.glob(par.prefix + '/simpsf/%d_*.txt' % wavel)
    kernels = [pyf.open(ifile)[1].data for ifile in spotfields]
    locations = np.zeros((len(spotfields), 2))

    if len(spotfields) != len(spotsizefiles):
        log.error('Number of kernels should match number of textfiles')

    ##################################################################
    # Reading kernel scale from Zemax textfile
    ##################################################################
    kernelscale = 0.0
    for spotsizefile in spotsizefiles:
        readFile = codecs.open(spotsizefile, encoding='utf-16-le')
        for i, line in enumerate(readFile):
            # physical scale is located on line 9, element 4
            if i == 9:
                kernelscale += float(line.split(' ')[3])
            elif i > 9:
                break
    kernelscale /= len(spotsizefiles)
    log.info(
        'kernel scale average is %.3f micron per pixel at %d nm' %
        (kernelscale, wavel))
    kernelscale *= 1e-6

    ##################################################################
    # Reading locations
    ##################################################################
    for i in range(len(spotfields)):
        name = re.sub('.fits', '', re.sub('.*PSF_', '', spotfields[i]))
        locations[i, 0] = float(name.split('_')[0])
        locations[i, 1] = float(name.split('_')[1])

    locations /= 2.
    locations += 0.5

    ##################################################################
    # Resample the kernels to the appropriate resolution
    ##################################################################
    log.info('Resampling kernels to match input')
    plateScale = par.pixsize / par.pxperdetpix
    log.debug(
        'Lenslet plane plate scale is %.3f micron per pixel' %
        (plateScale * 1e6))
    for i in range(len(locations)):
        # the scale in the incoming plane is par.pitch/par.pxprlens
        # the scale in the kernels is kernelscale
        # remap kernel to match incoming plate scale

        ratio = kernelscale / plateScale
        nx = int(kernels[i].shape[0] * ratio)
        ny = int(kernels[i].shape[1] * ratio)

        x = (np.arange(nx) - nx // 2) / ratio
        y = (np.arange(ny) - ny // 2) / ratio

        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        x = r * np.cos(theta + par.philens) + kernels[i].shape[0] // 2
        y = r * np.sin(theta + par.philens) + kernels[i].shape[1] // 2

        kernels[i] = ndimage.map_coordinates(kernels[i], [y, x])

    if hasattr(par, 'pxprlens'):
        log.info('Padding smaller kernels')
        newkernels = np.zeros((len(kernels), par.pxprlens, par.pxprlens))
        for k in range(len(locations)):
            kxm = par.pxprlens // 2 - kernels[k].shape[0] // 2
            kxp = kxm + kernels[k].shape[0]
            #print kxm,kxp,kernels[k].shape
            newkernels[k, kxm:kxp, kxm:kxp] += kernels[k]
    else:
        par.pxprlens = kernels[0].shape[0]
        log.info('pxprlens: %.3f' % par.pxprlens)
        newkernels = kernels

    for k in range(len(locations)):
        newkernels[k] /= np.sum(newkernels[k])
        if par.pinhole:
            #             if kernels[k].shape[0]<2*par.pxprlens+par.pin_dia/plateScale:
            #                 log.warning('Kernel too small to capture crosstalk')
            x = np.linspace(-1.5, 1.5, 3 * par.pxprlens) % 1
            x[np.where(x > 0.5)] -= 1
            x, y = np.meshgrid(x, x)

            mask = 1. * (np.sqrt(x**2 + y**2) <= 0.5 * par.pin_dia / par.pitch)
            xc = newkernels[k].shape[0] // 2
            yc = newkernels[k].shape[1] // 2
            mx = mask.shape[0] // 2
            my = mask.shape[1] // 2

            if xc < mx:
                xlow = mx - xc
                xhigh = xlow + newkernels[k].shape[0]
                ylow = my - yc
                yhigh = ylow + newkernels[k].shape[1]
                newkernels[k] *= mask[xlow:xhigh, ylow:yhigh]
            else:
                xlow = xc - mx
                xhigh = xlow + mask.shape[0]
                ylow = yc - my
                yhigh = ylow + mask.shape[1]
                newkernels[k, xlow:xhigh, ylow:yhigh] *= mask

    return newkernels, locations
