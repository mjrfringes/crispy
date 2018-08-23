#!/usr/bin/env python

import numpy as np
try:
    from astropy.io import fits
except BaseException:
    import pyfits as fits
import copy
from scipy import signal, ndimage, optimize, interpolate
from crispy.tools.image import Image
import glob
import re
import os

from crispy.tools.initLogger import getLogger
log = getLogger('crispy')


class PSFLets:
    """
    Helper class to deal with the PSFLets on the detector. Does most of the heavy lifting
    during the wavelength calibration step.
    """

    def __init__(self, load=False, infile=None, infiledir='.'):
        '''
        Initialize the class

        Parameters
        ----------
        load: Boolean
            Whether to load an already-existing wavelength calibration file
        infile: String
            If load is True, this is the name of the file
        infiledir: String
            If load is True, this is the directory in which the file resides
        '''
        self.xindx = None
        self.yindx = None
        self.lam_indx = None
        self.nlam = None
        self.nlam_max = None
        self.interp_arr = None
        self.order = None

        if load:
            self.loadpixsol(infile, infiledir)

    def loadpixsol(self, infile=None, infiledir='./calibrations'):
        '''
        Loads existing wavelength calibration file

        Parameters
        ----------
        infile: String
            Name of the file
        infiledir: String
            Directory in which the file resides
        '''
        if infile is None:
            infile = re.sub('//', '/', infiledir + '/PSFloc.fits')
        hdulist = fits.open(infile)

        try:
            self.lam_indx = hdulist[0].data
            self.xindx = hdulist[1].data
            self.yindx = hdulist[2].data
            self.nlam = hdulist[3].data.astype(int)
            self.good = hdulist[4].data.astype(int)
        except BaseException:
            raise RuntimeError(
                "File " +
                infile +
                " does not appear to contain a CHARIS wavelength solution in the appropriate format.")
        self.nlam_max = np.amax(self.nlam)

    def savepixsol(self, outdir="calibrations/"):
        '''
        Saves wavelength calibration file

        Parameters
        ----------
        outdir: String
            Directory in which to put the file. The file is name PSFloc.fits and is a
            multi-extension FITS file, each extension corresponding to:
            0. the list of wavelengths at which the calibration is done
            1. a 2D ndarray with the X position of all lenslets
            2. a 2D ndarray with the Y position of all lenslets
            3. a 2D ndarray with the number of valid wavelengths for a given lenslet (some wavelengths fall outside of the detector area)

        '''
        if not os.path.isdir(outdir):
            raise IOError(
                "Attempting to save pixel solution to directory " +
                outdir +
                ".  Directory does not exist.")
        outfile = re.sub('//', '/', outdir + '/PSFloc.fits')
        out = fits.HDUList(fits.PrimaryHDU(self.lam_indx))
        out.append(fits.PrimaryHDU(self.xindx))
        out.append(fits.PrimaryHDU(self.yindx))
        out.append(fits.PrimaryHDU(self.nlam.astype(int)))
        out.append(fits.PrimaryHDU(self.good.astype(int)))
        try:
            out.writeto(outfile, clobber=True)
        except BaseException:
            raise

    def geninterparray(self, lam, allcoef, order=3):
        '''
        Set up array to solve for best-fit polynomial fits to the
        coefficients of the wavelength solution.  These will be used
        to smooth/interpolate the wavelength solution, and
        ultimately to compute its inverse.

        Parameters
        ----------
        lam: float
            Wavelength in nm
        allcoef: list of lists floats
            Polynomial coefficients of wavelength solution
        order: int
            Order of polynomial wavelength solution

        Notes
        -----
        Populates the attribute interp_arr in PSFLet class
        '''

        self.interp_arr = np.zeros((order + 1, allcoef.shape[1]))
        self.order = order
        xarr = np.ones((lam.shape[0], order + 1))
        for i in range(1, order + 1):
            xarr[:, i] = np.log(lam)**i
        for i in range(self.interp_arr.shape[1]):
            coef = np.linalg.lstsq(xarr, allcoef[:, i])[0]
            self.interp_arr[:, i] = coef

    def return_locations_short(self, coef, xindx, yindx):
        '''
        Returns the x,y detector location of a given lenslet for a given polynomial fit

        Parameters
        ----------
        coef: lists floats
            Polynomial coefficients of fit for a single wavelength
        xindx: int
            X index of lenslet in lenslet array
        yindx: int
            Y index of lenslet in lenslet array

        Returns
        -------
        interp_x: float
            X coordinate on the detector
        interp_y: float
            Y coordinate on the detector
        '''
        interp_x, interp_y = transform(xindx, yindx, coeforder, coef)
        return interp_x, interp_y

    def return_res(self, lam, allcoef, xindx, yindx,
                   order=3, lam1=None, lam2=None):
        '''
        Returns the spectral resolution and interpolated wavelength array

        Parameters
        ----------
        lam: float
            Wavelength in nm
        allcoef: list of lists floats
            Polynomial coefficients of wavelength solution
        xindx: int
            X index of lenslet in lenslet array
        yindx: int
            Y index of lenslet in lenslet array
        order: int
            Order of polynomial wavelength solution
        lam1: float
            Shortest wavelength in nm
        lam2: float
            Longest wavelength in nm

        Returns
        -------
        interp_lam: array
            Array of wavelengths
        R: float
            Effective spectral resolution
        '''

        if lam1 is None:
            lam1 = np.amin(lam) / 1.04
        if lam2 is None:
            lam2 = np.amax(lam) * 1.03

        interporder = order

        if self.interp_arr is None:
            self.geninterparray(lam, allcoef, order=order)

        coeforder = int(np.sqrt(allcoef.shape[1])) - 1
        n_spline = 100

        interp_lam = np.linspace(lam1, lam2, n_spline)
        dy = []
        dx = []

        for i in range(n_spline):
            coef = np.zeros((coeforder + 1) * (coeforder + 2))
            for k in range(1, interporder + 1):
                coef += k * self.interp_arr[k] * np.log(interp_lam[i])**(k - 1)
            _dx, _dy = transform(xindx, yindx, coeforder, coef)

            dx += [_dx]
            dy += [_dy]

        R = np.sqrt(np.asarray(dy)**2 + np.asarray(dx)**2)

        return interp_lam, R

    def monochrome_coef(self, lam, alllam=None, allcoef=None, order=3):
        if self.interp_arr is None:
            if alllam is None or allcoef is None:
                raise ValueError(
                    "Interpolation array has not been computed.  Must call monochrome_coef with arrays.")
            self.geninterparray(alllam, allcoef, order=order)

        coef = np.zeros(self.interp_arr[0].shape)
        for k in range(self.order + 1):
            coef += self.interp_arr[k] * np.log(lam)**k
        return coef

    def return_locations(self, lam, allcoef, xindx, yindx, order=3):
        '''
        Calculates the detector coordinates of lenslet located at `xindx`, `yindx`
        for desired wavelength `lam`

        Parameters
        ----------
        lam: float
            Wavelength in nm
        allcoef: list of lists floats
            Polynomial coefficients of wavelength solution
        xindx: int
            X index of lenslet in lenslet array
        yindx: int
            Y index of lenslet in lenslet array
        order: int
            Order of polynomial wavelength solution

        Returns
        -------
        interp_x: float
            X coordinate on the detector
        interp_y: float
            Y coordinate on the detector
        '''
        if len(allcoef.shape) == 1:
            coeforder = int(np.sqrt(allcoef.shape[0])) - 1
            interp_x, interp_y = transform(xindx, yindx, coeforder, allcoef)
            return interp_x, interp_y

        if self.interp_arr is None:
            self.geninterparray(lam, allcoef, order=order)

        coeforder = int(np.sqrt(allcoef.shape[1])) - 1
        if not (coeforder + 1) * (coeforder + 2) == allcoef.shape[1]:
            raise ValueError(
                "Number of coefficients incorrect for polynomial order.")

        coef = np.zeros((coeforder + 1) * (coeforder + 2))
        for k in range(self.order + 1):
            coef += self.interp_arr[k] * np.log(lam)**k
        interp_x, interp_y = transform(xindx, yindx, coeforder, coef)

        return interp_x, interp_y
        
#     def return_fine_locations(self, lam, xindx, yindx, xlistarr, ylistarr):
#         '''
#         Calculates the detector coordinates of lenslet located at `xindx`, `yindx`
#         for desired wavelength `lam`, using the fine calibration method
# 
#         Parameters
#         ----------
#         lam: float
#             Wavelength in nm
#         xindx: int
#             X index of lenslet in lenslet array
#         yindx: int
#             Y index of lenslet in lenslet array
#         order: int
#             Order of polynomial wavelength solution
# 
#         Returns
#         -------
#         interp_x: float
#             X coordinate on the detector
#         interp_y: float
#             Y coordinate on the detector
#         '''
#         interp_x, interp_y = fine_transform(xindx, yindx, lam, xlistarr, ylistarr)
        

    def genpixsol(
            self,
            par,
            lam,
            allcoef,
            order=3,
            lam1=None,
            lam2=None,
            borderpix=4,
            finexy=None):
        '''
        Calculates the wavelength at the center of each pixel within a microspectrum

        Parameters
        ----------
        lam: float
            Wavelength in nm
        allcoef: list of floats
            List describing the polynomial coefficients that best fit the lenslets,
            for all wavelengths
        order: int
            Order of the polynomical fit
        lam1: float
            Lowest wavelength in nm
        lam2: float
            Highest wavelength in nm

        Notes
        -----
        This functions fills in most of the fields of the PSFLet class: the array
        of xindx, yindx, nlam, lam_indx and nlam_max
        '''

        ###################################################################
        # Read in wavelengths of spots, coefficients of wavelength
        # solution.  Obtain extrapolated limits of wavlength solution
        # to 4% below and 3% above limits of the coefficient file by
        # default.
        ###################################################################

        if lam1 is None:
            lam1 = np.amin(lam)# * 0.98
        if lam2 is None:
            lam2 = np.amax(lam)# * 1.02
        interporder = order

        if self.interp_arr is None:
            self.geninterparray(lam, allcoef, order=order)

        coeforder = int(np.sqrt(allcoef.shape[1])) - 1
        if not (coeforder + 1) * (coeforder + 2) == allcoef.shape[1]:
            raise ValueError(
                "Number of coefficients incorrect for polynomial order.")

        xindx = np.arange(-par.nlens / 2, par.nlens / 2)
        xindx, yindx = np.meshgrid(xindx, xindx)

        n_spline = 100

        interp_x = np.zeros(tuple([n_spline] + list(xindx.shape)))
        interp_y = np.zeros(interp_x.shape)
        interp_lam = np.linspace(lam1, lam2, n_spline)

        for i in range(n_spline):
            coef = np.zeros((coeforder + 1) * (coeforder + 2))
            for k in range(interporder + 1):
                coef += self.interp_arr[k] * np.log(interp_lam[i])**k
            interp_x[i], interp_y[i] = transform(xindx, yindx, coeforder, coef)
        if finexy is not None:
            interp_x += finexy[0]
            interp_y += finexy[1]
        

        x = np.zeros(tuple(list(xindx.shape) + [1000]))
        y = np.zeros(x.shape)
        nlam = np.zeros(xindx.shape, np.int)
        lam_out = np.zeros(y.shape)
        good = np.ones(xindx.shape)
        
        # SNR threshold
        if finexy is not None: good *= finexy[2]>10


        for ix in range(xindx.shape[0]):
            for iy in range(xindx.shape[1]):
                pix_y = interp_x[:, ix, iy]
                pix_x = interp_y[:, ix, iy]

                if np.any(
                    pix_x < borderpix) or np.any(
                    pix_x > par.npix -
                    borderpix) or np.any(
                    pix_y < borderpix) or np.any(
                    pix_y > par.npix -
                    borderpix):
                    good[ix, iy] = 0
                    continue

                if pix_y[-1] < pix_y[0]:
                    try:
                        tck_y = interpolate.splrep(
                            pix_y[::-1], interp_lam[::-1], k=1, s=0)
                    except BaseException:
                        good[ix, iy] = 0
                        raise
                else:
                    try:
                        tck_y = interpolate.splrep(pix_y, interp_lam, k=1, s=0)
                    except:
                        good[ix, iy] = 0
                        log.error('Error on wavelength calibration for lenslet ({:})'.format((ix,iy)))

                y1, y2 = [int(np.amin(pix_y)) + 1, int(np.amax(pix_y))]
                tck_x = interpolate.splrep(interp_lam, pix_x, k=1, s=0)

                nlam[ix, iy] = y2 - y1 + 1
                y[ix, iy, :nlam[ix, iy]] = np.arange(y1, y2 + 1)
                # evaluate wavelengths at integer pixel values
                lam_out[ix, iy, :nlam[ix, iy]] = interpolate.splev(
                    y[ix, iy, :nlam[ix, iy]], tck_y) 
                x[ix, iy, :nlam[ix, iy]] = interpolate.splev(
                    lam_out[ix, iy, :nlam[ix, iy]], tck_x)

        for nlam_max in range(x.shape[-1]):
            #             if np.all(y[:, :, nlam_max] == 0):
            if np.all(y[:, :, nlam_max] == 0):
                break

        self.xindx = y[:, :, :nlam_max] # array of integer pixel indices along dispersion
        self.yindx = x[:, :, :nlam_max] # array of floats indicating the cross. disp. axis
        self.nlam = nlam
        self.lam_indx = lam_out[:, :, :nlam_max] # wavelengths at int. pixel indices
        self.nlam_max = np.amax(nlam)
        self.good = good


def initcoef(order, scale, phi, x0=0, y0=0):
    """
    Create a set of coefficients including a rotation matrix plus zeros.

    Parameters
    ----------
    order: int
        The polynomial order of the grid distortion
    scale: float
        The linear separation in pixels of the PSFlets. Default 13.88.
    phi:   float
        The pitch angle of the lenslets.  Default atan(2)
    x0:    float
        x offset to apply to the central pixel. Default 0
    y0:    float
        y offset to apply to the central pixel. Default 0

    Returns
    -------
    coef: list of floats
        A list of length (order+1)*(order+2) to be optimized.

    Notes
    -----
    The list of coefficients has space for a polynomial fit of the
    input order (i.e., for order 3, up to terms like x**3 and x**2*y,
    but not x**3*y).  It is all zeros in the output apart from the
    rotation matrix given by scale and phi.
    """

    try:
        if not order == int(order):
            raise ValueError("Polynomial order must be integer")
        else:
            if order < 1 or order > 5:
                raise ValueError("Polynomial order must be >0, <=5")
    except BaseException:
        raise ValueError("Polynomial order must be integer")

    n = (order + 1) * (order + 2)
    coef = np.zeros((n))

    coef[0] = x0
    coef[1] = scale * np.cos(phi)
    coef[order + 1] = -scale * np.sin(phi)
    coef[n // 2] = y0
    coef[n // 2 + 1] = scale * np.sin(phi)
    coef[n // 2 + order + 1] = scale * np.cos(phi)

    return list(coef)


def transform(x, y, order, coef):
    """
    Apply the coefficients given to transform the coordinates using
    a polynomial.

    Parameters
    ----------
    x:     ndarray
        Rectilinear grid
    y:     ndarray of floats
        Rectilinear grid
    order: int
        Order of the polynomial fit
    coef:  list of floats
        List of the coefficients.  Must match the length required by
        order = (order+1)*(order+2)

    Returns
    -------
    _x:    ndarray
        Transformed coordinates
    _y:    ndarray
        Transformed coordinates

    """

    try:
        if not len(coef) == (order + 1) * (order + 2):
            raise ValueError(
                "Number of coefficients incorrect for polynomial order.")
    except BaseException:
        raise AttributeError("order must be integer, coef should be a list.")

    try:
        if not order == int(order):
            raise ValueError("Polynomial order must be integer")
        else:
            if order < 1 or order > 5:
                raise ValueError("Polynomial order must be >0, <=5")
    except BaseException:
        raise ValueError("Polynomial order must be integer")

    _x = np.zeros(np.asarray(x).shape)
    _y = np.zeros(np.asarray(y).shape)

    i = 0
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            _x += coef[i] * x**ix * y**iy
            i += 1
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            _y += coef[i] * x**ix * y**iy
            i += 1

    return [_x, _y]

def revealCoefs(coef,order):

    i = 0
    s = 'i and j are the integer coordinates of the lenslets in the array\n'
    s+='X coordinates:\n'
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            s += '{:} * i^{:} j^{:} + \n'.format(coef[i],ix,iy)
            i += 1
    s += 'Y coordinates:\n'
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            s += '{:} * i^{:} j^{:} + \n'.format(coef[i],ix,iy)
            i += 1
    return s



def fine_transform(lam, x, y, reflam, xlistarr, ylistarr):
    """
    Apply the coefficients given to transform the coordinates using
    a polynomial.

    Parameters
    ----------
    lam: float or 1D ndarray
        Desired wavelength (or array of wavelength)
    x:     ndarray
        Rectilinear grid
    y:     ndarray
        Rectilinear grid
    reflam: float or 1D ndarray
        Reference wavelength array at which xlistarr and ylistarr were computed
    xlistarr:     ndarray
        Centroids
    ylistarr:     ndarray
        Centroids

    Returns
    -------
    _x:    ndarray
        Transformed coordinates
    _y:    ndarray
        Transformed coordinates

    """

    if hasattr(lam, "__len__"):
        _x = np.zeros((len(lam),x.shape[0],x.shape[1]))
        _y = np.zeros((len(lam),y.shape[0],y.shape[1]))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tck = interpolate.splrep(reflam, xlistarr[:,i,j])
                _x[:,i,j] = interpolate.splev(lam, tck, der=0)
                tck = interpolate.splrep(reflam, ylistarr[:,i,j])
                _y[:,i,j] = interpolate.splev(lam, tck, der=0)
    
    else:
        _x = np.zeros((x.shape[0],x.shape[1]))
        _y = np.zeros((y.shape[0],y.shape[1]))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tck = interpolate.splrep(reflam, xlistarr[:,i,j])
                _x[i,j] = interpolate.splev(lam, tck, der=0)
                tck = interpolate.splrep(reflam, ylistarr[:,i,j])
                _y[i,j] = interpolate.splev(lam, tck, der=0)
        
    
    
    return [_x, _y]
    
def new_transform(x, y, order, coef):
    """
    Apply the coefficients given to transform the coordinates using
    a polynomial.

    Parameters
    ----------
    x:     ndarray
        Rectilinear grid
    y:     ndarray of floats
        Rectilinear grid
    order: int
        Order of the polynomial fit
    coef:  list of floats
        List of the coefficients.  Must match the length required by
        order = (order+1)*(order+2)

    Returns
    -------
    _x:    ndarray
        Transformed coordinates
    _y:    ndarray
        Transformed coordinates

    """
    Xlist = []
    Ylist = []
    i = 0
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            Xlist.append(x**ix * y**iy)
            Ylist.append(x**ix * y**iy)

    Xlist = np.array(Xlist)
    Ylist = np.array(Ylist)
    ncoefs = (order + 1) * (order + 2) // 2
    Xcoefs = coef[:ncoefs]
    Ycoefs = coef[-ncoefs:]
    X = np.dot(Xlist.T,Xcoefs)
    Y = np.dot(Ylist.T,Ycoefs)
    return X,Y


def corrval(coef, x, y, filtered, order, trimfrac=0.1):
    """
    Return the negative of the sum of the middle XX% of the PSFlet
    spot fluxes (disregarding those with the most and the least flux
    to limit the impact of outliers).  Analogous to the trimmed mean.

    Parameters
    ----------
    coef:     list of floats
        coefficients for polynomial transformation
    x: ndarray
        coordinates of lenslets
    y: ndarray
        coordinates of lenslets
    filtered: ndarray
        image convolved with gaussian PSFlet
    order: int
        order of the polynomial fit
    trimfrac: float
        fraction of outliers (high & low combined) to trim
        Default 0.1 (5% trimmed on the high end, 5% on the low end)

    Returns
    -------
    score:    float
        Negative sum of PSFlet fluxes, to be minimized
    """

    #################################################################
    # Use np.nan for lenslet coordinates outside the FOV,
    # discard these from the calculation before trimming.
    #################################################################

    _x, _y = transform(x, y, order, coef)
    vals = ndimage.map_coordinates(filtered, [_y, _x], mode='constant',
                                   cval=np.nan, prefilter=False)
    vals_ok = vals[np.where(np.isfinite(vals))]
    
    if trimfrac>0.0:
        iclip = int(vals_ok.shape[0] * trimfrac / 2)
        vals_sorted = np.sort(vals_ok)
        score = -1 * np.sum(vals_sorted[iclip:-iclip])
    else:
        vals_sorted = np.sort(vals_ok)
        score = -1 * np.sum(vals_sorted)
       
    return score


def corrvalsum(coef, x, y, filtered, order, trimfrac=0.1,gsize=2):
    _x, _y = transform(x, y, order, coef)
    ydim,xdim = filtered.shape
    s = 0.0
    ry = np.reshape(_y,-1)
    rx = np.reshape(_x,-1)
    for i in range(len(ry)):
        yi = ry[i]
        xi = rx[i]
        xmin = int(xi)-gsize
        xmax = xmin+2*gsize
        ymin = int(yi)-gsize
        ymax = ymin+2*gsize
        if ymin>2*gsize and xmin>2*gsize and xmax<xdim-2*gsize and ymax<ydim-2*gsize:
            dx = xi - int(xi)
            dy = yi - int(yi)
#             s+=np.sum(simplepsf(size=2*gsize,fwhm=fwhm,offx=dx,offy=dy)*filtered[ymin:ymax,xmin:xmax])
#             s+=np.sum(gausspsf(size=2*gsize,fwhm=fwhm,offx=dx,offy=dy)*filtered[ymin:ymax,xmin:xmax])
            s+=np.sum(filtered[ymin:ymax,xmin:xmax])
    return -s



def locatePSFlets(inImage, mask, polyorder=2, sig=0.7, coef=None, trimfrac=0.1,
                  phi=np.arctan2(1.926, -1), scale=15.02, nlens=108):
    """
    function locatePSFlets takes an Image class, assumed to be a
    monochromatic grid of spots with read noise and shot noise, and
    returns the esimated positions of the spot centroids.  This is
    designed to constrain the domain of the PSF-let fitting later in
    the pipeline.

    Parameters
    ----------
    imImage: Image class
        Assumed to be a monochromatic grid of spots
    polyorder: float
        order of the polynomial coordinate transformation. Default 2.
    sig: float
        standard deviation of convolving Gaussian used
        for estimating the grid of centroids.  Should be close
        to the true value for the PSF-let spots.  Default 0.7.
    coef: list
        initial guess of the coefficients of polynomial
        coordinate transformation
    trimfrac: float
        fraction of lenslet outliers (high & low
        combined) to trim in the minimization.  Default 0.1
        (5% trimmed on the high end, 5% on the low end)

    Returns
    -------
    x: 2D ndarray
        Estimated spot centroids in x.
    y: 2D ndarray
        Estimated spot centroids in y.
    good:2D boolean ndarray
        True for lenslets with spots inside the detector footprint
    coef: list of floats
        List of best-fit polynomial coefficients

    Notes
    -----
    the coefficients, if not supplied, are initially set to the
    known pitch angle and scale.  A loop then does a quick check to find
    reasonable offsets in x and y.  With all of the first-order polynomial
    coefficients set, the optimizer refines these and the higher-order
    coefficients.  This routine seems to be relatively robust down to
    per-lenslet signal-to-noise ratios of order unity (or even a little
    less).

    """

    #############################################################
    # Convolve with a Gaussian, centroid the filtered image.
    #############################################################

    x = np.arange(-1 * int(3 * sig + 1), int(3 * sig + 1) + 1)
    x, y = np.meshgrid(x, x)
    gaussian = np.exp(-(x**2 + y**2) / (2 * sig**2))

#     if mask is None:
#         unfiltered = signal.convolve2d(inImage.data, gaussian, mode='same')
#     else:
#         unfiltered = signal.convolve2d(
#             inImage.data * inImage.ivar, gaussian, mode='same')
#         unfiltered /= signal.convolve2d(inImage.ivar,
#                                         gaussian, mode='same') + 1e-10
    unfiltered = signal.convolve2d(
        inImage.data * mask, gaussian, mode='same')

    filtered = ndimage.interpolation.spline_filter(unfiltered)

    #############################################################
    # x, y: Grid of lenslet IDs, Lenslet (0, 0) is the center.
    #############################################################

    #gridfrac = 10
    ydim, xdim = inImage.data.shape
    #x = np.arange(-(ydim//gridfrac), ydim//gridfrac + 1)
    x = np.arange(-nlens // 2, nlens // 2)
    x, y = np.meshgrid(x, x)

    #############################################################
    # Set up polynomial coefficients, convert from lenslet
    # coordinates to coordinates on the detector array.
    # Then optimize the coefficients.
    # We want to start with a decent guess, so we use a grid of
    # offsets.  Seems to be robust down to SNR/PSFlet ~ 1
    # Create slice indices for subimages to perform the intial
    # fits on. The new dimensionality in both x and y is 2*subsize
    #############################################################

    if coef is None:

        log.info("Initializing PSFlet location transformation coefficients")
        bestval = 0
        subshape = xdim // 3
        _s = x.shape[0] // 3
        subfiltered = ndimage.interpolation.spline_filter(
            unfiltered[subshape:-subshape, subshape:-subshape])
#         for ix in np.arange(0, 14, 0.5):
#             for iy in np.arange(0, 25, 0.5):
        for ix in np.arange(-7, 7, 0.5):
            for iy in np.arange(-9, 9, 0.5):
                coef = initcoef(
                    polyorder,
                    x0=ix +
                    xdim /
                    2. -
                    subshape,
                    y0=iy +
                    ydim /
                    2. -
                    subshape,
                    scale=scale,
                    phi=phi)
                newval = corrval(coef, x[_s:-_s, _s:-_s], y[_s:-_s, _s:-_s],
                                 subfiltered, polyorder, trimfrac)
                if newval < bestval:
                    bestval = newval
                    coefbest = copy.deepcopy(coef)
        coef_opt = coefbest
        

        log.info(
            "Performing initial optimization of PSFlet location transformation coefficients for frame " +
            inImage.filename)
        res = optimize.minimize(corrval, coef_opt, args=(
            x[_s:-_s, _s:-_s], y[_s:-_s, _s:-_s], subfiltered, polyorder, trimfrac), method='Powell')
        coef_opt = res.x

        coef_opt[0] += subshape
        coef_opt[(polyorder + 1) * (polyorder + 2) // 2] += subshape
        log.info('Array origin: {:}'.format((coef_opt[0],coef_opt[(polyorder + 1) * (polyorder + 2) // 2])))

    #############################################################
    # If we have coefficients from last time, we assume that we
    # are now at a slightly higher wavelength, so try out offsets
    # that are slightly to the right to get a good initial guess.
    #############################################################

    else:
        log.info("Initializing transformation coefficients with previous values")
        bestval = 0
        coefsave = list(coef[:])

        for ix in np.arange(-1, 3.1, 0.2):
            for iy in np.arange(-1, 3.1, 0.2):
                coef = coefsave[:]
                coef[0] += ix
                coef[(polyorder + 1) * (polyorder + 2) // 2] += iy

                newval = corrval(coef, x, y, filtered, polyorder, trimfrac)
                if newval < bestval:
                    bestval = newval
                    coefbest = copy.deepcopy(coef)
        coef_opt = coefbest

    log.info(
        "Performing final optimization of PSFlet location transformation coefficients for frame " +
        inImage.filename)

    res = optimize.minimize(
        corrval,
        coef_opt,
        args=(
            x,
            y,
            filtered,
            polyorder,
            trimfrac),
        method='Powell')

    coef_opt = res.x
    log.info('Array origin: {:}'.format((coef_opt[0],coef_opt[(polyorder + 1) * (polyorder + 2) // 2])))

    if not res.success:
        log.info(
            "Optimizing PSFlet location transformation coefficients may have failed for frame " +
            inImage.filename)
    _x, _y = transform(x, y, polyorder, coef_opt)

    #############################################################
    # Boolean: do the lenslet PSFlets lie within the detector?
    #############################################################

    good = (_x > 5) * (_x < xdim - 5) * (_y > 5) * (_y < ydim - 5)

    return [_x, _y, good, coef_opt]
