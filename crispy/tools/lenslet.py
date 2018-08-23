#!/usr/bin/env python

import numpy as np
try:
    from astropy.io import fits as pyf
except BaseException:
    import pyfits as pyf
from crispy.tools.rotate import Rotate
from crispy.tools.initLogger import getLogger
log = getLogger('crispy')
import matplotlib.pyplot as plt
from crispy.tools.detutils import frebin
from scipy import ndimage
from scipy.special import erf
from crispy.tools.spectrograph import distort
from crispy.tools.locate_psflets import initcoef, transform, PSFLets


def processImagePlane(par, imagePlane, noRot=False):
    '''
    Function processImagePlane

    Rotates an image or slice, and rebins in a flux-conservative way
    on an array of lenslets, using the plate scale provided in par.pixperlenslet.
    Each pixel represents the flux within a lenslet. Starts by padding the original
    image to avoid cropping edges when rotating. This step necessarily involves an
    interpolation, so one needs to be cautious.

    Parameters
    ----------
    par :   Parameters instance
            Contains all IFS parameters
    imagePlane : 2D array
            Input slice to IFS sim, first dimension of data is wavelength

    Returns
    -------
    imagePlaneRot : 2D array
            Rotated image plane on same sampling as original.
    '''

    paddedImagePlane = np.zeros(
        (int(imagePlane.shape[0] * np.sqrt(2)), int(imagePlane.shape[1] * np.sqrt(2))))

    xdim, ydim = paddedImagePlane.shape
    xpad = xdim - imagePlane.shape[0]
    ypad = ydim - imagePlane.shape[1]
    xpad //= 2
    ypad //= 2
    paddedImagePlane[xpad:-xpad, ypad:-ypad] = imagePlane

    if noRot:
        imagePlaneRot = paddedImagePlane.copy()
    else:
        imagePlaneRot = Rotate(paddedImagePlane, par.philens, clip=False)

    ######################################################################
    # Flux conservative rebinning
    ######################################################################
    newShape = (int(imagePlaneRot.shape[0] /
                    par.pixperlenslet), int(imagePlaneRot.shape[1] /
                                            par.pixperlenslet))
    imagePlaneRot = frebin(imagePlaneRot, newShape)
    log.debug('Input plane is %dx%d' % imagePlaneRot.shape)

    return imagePlaneRot


def propagateLenslets(
        par,
        imageplane,
        lam1,
        lam2,
        hires_arrs=None,
        lam_arr=None,
        upsample=3,
        nlam=10,
        npix=13,
        order=3,
        x0=0.0):
    """
    Function propagateLenslets

    This is the main propagation function. It puts the PSFLets where they belong on the detector.
    It uses template PSFLets given in hires_arrs, and can use also a pre-determined wavelength
    solution through the allcoef argument.

    Parameters
    ----------

    par: Params instance
        Parameters instance for crispy
    imageplane: 2D array
        Flux map where each pixel corresponds to one lenslet
    lam1: float
        Minimum wavelength in IFS band
    lam2: float
        Maximum wavelength in IFS band
    hires_arr: 4D ndarray
        For each wavelength, for each location on the detector, a 2D array of the oversampled PSFLet
    lam_arr: 1D array
        Wavelength array corresponding to the hires_arr array
    upsample: int
        Factor by which the PSFLets are oversampled
    nlam: int
        Number of wavelengths to oversample a given wavelength bin
    npix: int
       PSFLet will be put on npix*npix detector pixels, models will be (npix*upsample)^2
    order: int
        Order used in the polynomial fit of the wavelength solution
    x0: float
        Offset from the center of the detector in the vertical direction (x)
    """

    padding = 10
    ydim, xdim = imageplane.shape

    xindx = np.arange(-xdim // 2, -xdim // 2 + xdim)
    xindx, yindx = np.meshgrid(xindx, xindx)

    image = np.zeros((par.npix + 2 * padding, par.npix + 2 * padding))
    x = np.arange(image.shape[0])
    x, y = np.meshgrid(x, x)

    dloglam = (np.log(lam2) - np.log(lam1)) / nlam
    loglam = np.log(lam1) + dloglam / 2. + np.arange(nlam) * dloglam

    # load external PSFLet positions
    if par.PSFLetPositions:
        psftool = PSFLets()
        lamlist = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]
        allcoef = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 1:]
        psftool.geninterparray(lamlist, allcoef)

    for lam in np.exp(loglam):

        ################################################################
        # Build the appropriate average hires image by averaging over
        # the nearest wavelengths.  Then apply a spline filter to the
        # interpolated high resolution PSFlet images to avoid having
        # to do this later, saving a factor of a few in time.
        ################################################################

        if (hires_arrs is None) or (lam_arr is None):
            log.error('No template PSFLets given!')
            return
        else:
            hires = np.zeros((hires_arrs[0].shape))
            if lam <= np.amin(lam_arr):
                hires[:] = hires_arrs[0]
            elif lam >= np.amax(lam_arr):
                hires[:] = hires_arrs[-1]
            else:
                i1 = np.amax(np.arange(len(lam_arr))[np.where(lam > lam_arr)])
                i2 = i1 + 1
                hires = hires_arrs[i1] * \
                    (lam - lam_arr[i1]) / (lam_arr[i2] - lam_arr[i1])
                hires += hires_arrs[i2] * \
                    (lam_arr[i2] - lam) / (lam_arr[i2] - lam_arr[i1])

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

        ################################################################
        # here is where one could import any kind of polynomial mapping
        # and introduce distortions
        ################################################################
        if par.PSFLetPositions:
            xcen, ycen = psftool.return_locations(
                lam, allcoef, xindx, yindx, order=order)
        else:
            dispersion = par.npixperdlam * par.R * np.log(lam / par.FWHMlam)
            coef = initcoef(
                order,
                scale=par.pitch / par.pixsize,
                phi=par.philens,
                x0=par.npix // 2 + dispersion,
                y0=par.npix // 2 + x0)
            xcen, ycen = transform(xindx, yindx, order, coef)

        xcen += padding
        ycen += padding
        xindx = np.reshape(xindx, -1)
        yindx = np.reshape(yindx, -1)
        xcen = np.reshape(xcen, -1)
        ycen = np.reshape(ycen, -1)
        for i in range(xcen.shape[0]):
            if not (
                    xcen[i] > npix //
                    2 and xcen[i] < image.shape[0] -
                    npix //
                    2 and ycen[i] > npix //
                    2 and ycen[i] < image.shape[0] -
                    npix //
                    2):
                continue

            # these are the coordinates of the lenslet within the image plane
            Ycoord = yindx[i] + imageplane.shape[0] // 2
            Xcoord = xindx[i] + imageplane.shape[1] // 2

            if not (Xcoord > 0 and Xcoord <
                    imageplane.shape[1] and Ycoord > 0 and Ycoord < imageplane.shape[0]):
                continue

            val = imageplane[Xcoord, Ycoord]

            # if the value is 0, don't waste time
            if val == 0.0:
                continue

            # central pixel -> npix*upsample//2
            iy1 = int(ycen[i]) - npix // 2
            iy2 = iy1 + npix
            ix1 = int(xcen[i]) - npix // 2
            ix2 = ix1 + npix

            yinterp = (y[iy1:iy2, ix1:ix2] - ycen[i]) * \
                upsample + upsample * npix / 2.
            xinterp = (x[iy1:iy2, ix1:ix2] - xcen[i]) * \
                upsample + upsample * npix / 2.

            # Now find the closest high-resolution PSFs from a library
            if hires.shape[0] == 1 and hires.shape[1] == 1:
                image[iy1:iy2,
                      ix1:ix2] += val * ndimage.map_coordinates(hires[0,
                                                                      0],
                                                                [yinterp,
                                                                 xinterp],
                                                                prefilter=False) / nlam
            else:
                x_hires = xcen[i] * 1. / image.shape[1]
                y_hires = ycen[i] * 1. / image.shape[0]

                x_hires = x_hires * hires_arrs[0].shape[1] - 0.5
                y_hires = y_hires * hires_arrs[0].shape[0] - 0.5

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
                weight22 = max(0, (x_hires - i1) * (y_hires - j1))
                weight12 = max(0, (x_hires - i1) * (j2 - y_hires))
                weight21 = max(0, (i2 - x_hires) * (y_hires - j1))
                weight11 = max(0, (i2 - x_hires) * (j2 - y_hires))
                totweight = weight11 + weight21 + weight12 + weight22
                weight11 /= totweight * nlam
                weight12 /= totweight * nlam
                weight21 /= totweight * nlam
                weight22 /= totweight * nlam

                image[iy1:iy2, ix1:ix2] += val * weight11 * \
                    ndimage.map_coordinates(hires[j1, i1], [yinterp, xinterp], prefilter=False)
                image[iy1:iy2, ix1:ix2] += val * weight12 * \
                    ndimage.map_coordinates(hires[j1, i2], [yinterp, xinterp], prefilter=False)
                image[iy1:iy2, ix1:ix2] += val * weight21 * \
                    ndimage.map_coordinates(hires[j2, i1], [yinterp, xinterp], prefilter=False)
                image[iy1:iy2, ix1:ix2] += val * weight22 * \
                    ndimage.map_coordinates(hires[j2, i2], [yinterp, xinterp], prefilter=False)

    image = image[padding:-padding, padding:-padding]
    return image
