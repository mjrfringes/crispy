#!/usr/bin/env python

import os
import numpy as np
from numpy import sqrt, arcsin
try:
    from astropy.io import fits
except BaseException:
    import pyfits as fits


class Params(object):

    def __init__(self, codeRoot='../'):
        '''
        Main class containing all the sim parameters
        '''

        self.saveRotatedInput = False
        self.saveLensletPlane = False
        self.saveDetector = True
        self.savePoly = True  # save polychromatic cubes
        self.codeRoot = codeRoot
        self.prefix = self.codeRoot + '/ReferenceFiles'
        self.exportDir = self.codeRoot + '/SimResults'
        self.unitTestsOutputs = self.codeRoot + '/unitTestsOutputs'
#         self.wavecalDir = self.prefix+'/VARIAwavecal/'
#         self.wavecalDir = self.prefix+'/HighSNRWavecal/'
#         self.wavecalDir = self.prefix+'/wavecal/'
#         self.wavecalDir = self.prefix + '/wavecalR50_660/'
        self.wavecalDir = self.prefix + '/wavecalHCIFS/'
#         self.wavecalDir = self.prefix+'/wavecalR50_660_SC/'
#         self.wavecalName = 'VARIAcalCube.fits'
#         self.wavecalName = 'LLTFcalCube.fits'
#         self.wavecalName = 'simCalCube.fits'
        self.filelist = []
        self.lamlist = []

        ######################################################################
        # Basic resolution/configuration parameters
        ######################################################################

        # Number of lenslets across array (account for rotation)
        self.nlens = 108
        self.pitch = 85.72e-6         # Lenslet pitch (meters)
        self.interlace = 2.          # Interlacing
        # Rotation angle of the lenslets (radians)
        self.philens = arcsin(1. / sqrt(self.interlace**2 + 1))
        self.lensletsampling = 1. / 2.  # lenslet size in lambda/D
        self.lensletlam = 660.     # Wavelength at which this is defined (nm)

        ######################################################################
        # Detector stuff
        ######################################################################

        self.npix = 1024            # Number of pixels in final detector
        self.pixsize = 6.45e-6        # Pixel size (meters)
        self.pxperdetpix = 1       # Oversampling of the final detector pixels
        self.convolve = True        # whether to convolve the existing kernels with
        # gaussian kernel (simulating defocus)
        self.FWHM = 2.               # FWHM of gaussian kernel
        self.FWHMlam = 660.         # Lam at which FWHM is defined
        self.gaussian = True        # Use standard Gaussian kernels instead of library
        self.gaussian_hires = True  # Use Gaussians for hires PSFLet matching, instead
        # of Lucy-Richardson deconvolution
        # use the wavelength calibration to place the PSFLets on the detector
        self.PSFLetPositions = False

        self.QE = "QE_CCD201_wl.txt"  # detector QE, including wavelength-dependent losses
        # to not include the losses, use simply "QE_CCD201.txt" and set
        # self.losses parameters below
        # whether to disable all the noises (except QE)
        self.nonoise = True
        self.poisson = True         # Use Poisson statistics?
        self.RN = 100.0             # Gain register read noise (e/px/fr)
        self.CIC = 1e-2             # Clock-induced charge (e/px/fr)
        self.dark = 2e-4            # Dark current noise (e/px/s)
        self.Traps = False          # Use traps or not (not yet implemented)
        # total losses for on-axis PSF (obsolete, now contained directly in QE
        # file)
        self.losses = 1.
        self.PhCountEff = 1.0		# Photon counting efficiency
        # fraction of lifetime (using Bijan's empirical model)
        self.lifefraction = 0.0
        self.pol = 1.		        # Polarization losses
        self.EMStats = True         # Calculate EM stats with Gamma function?
        self.EMGain = 2500.          # Gain of the EM stage
        self.PCmode = True          # Photon counting mode?
        self.PCbias = 200           # In order to allow the RN to be negative
        # if PCmode is True, this is photon detection threshold (sigmas)
        self.threshold = 5
        self.Nreads = 3			# number of reads for a target frame
        self.timeframe = 1000		# time in second for a frame (from file)

        ######################################################################
        # Spectrograph stuff
        ######################################################################

        self.BW = 0.18              # Spectral bandwidth
        self.npixperdlam = 2.0       # Number of pixels per spectral resolution element
        self.nchanperspec_lstsq = 2.0  # Nspec per pixel for least squares
        # Spectral resolving power (extracted cubes have twice)
        self.R = 50

        # carry-over old parameter names
        # Wavelength at which this is defined (nm)
        self.lenslet_wav = self.lensletlam
        self.lenslet_sampling = self.lensletsampling  # lenslet size in lambda/D

        self.makeHeader()

    def makeHeader(self):
        self.hdr = fits.PrimaryHDU().header
        self.hdr.append(('comment', ''), end=True)
        self.hdr.append(('comment', '*' * 60), end=True)
        self.hdr.append(
            ('comment',
             '*' *
             22 +
             ' General parameters ' +
             '*' *
             18),
            end=True)
        self.hdr.append(('comment', '*' * 60), end=True)
        self.hdr.append(('comment', ''), end=True)
        self.hdr.append(
            ('NLENS',
             self.nlens,
             '# lenslets across array'),
            end=True)
        self.hdr.append(
            ('PITCH',
             self.pitch,
             'Lenslet pitch (meters)'),
            end=True)
        self.hdr.append(('INTERLAC', self.interlace, 'Interlacing'), end=True)
        self.hdr.append(
            ('PHILENS',
             self.philens * 180. / np.pi,
             'Rotation angle of the lenslets (deg)'),
            end=True)
        self.hdr.append(
            ('PIXSIZE',
             self.pixsize,
             'Pixel size (meters)'),
            end=True)
        self.hdr.append(
            ('LENSAMP',
             self.lenslet_sampling,
             'Lenslet sampling (lam/D)'),
            end=True)
        self.hdr.append(
            ('LSAMPWAV',
             self.lenslet_wav,
             'Lenslet sampling wavelength (nm)'),
            end=True)
        self.hdr.append(
            ('FWHM',
             self.FWHM,
             'FHWM of PSFLet at detector (pixels)'),
            end=True)
        self.hdr.append(
            ('FWHMLAM',
             self.FWHMlam,
             'Wavelength at which FWHM is defined (nm)'),
            end=True)
        self.hdr.append(
            ('NPIX',
             self.npix,
             'Number of detector pixels'),
            end=True)
        self.hdr.append(('BW', self.BW, 'Bandwidth'), end=True)
        self.hdr.append(
            ('PIXPRLAM',
             self.npixperdlam,
             'Pixels per resolution element'),
            end=True)
        self.hdr.append(
            ('RESLSTSQ',
             self.nchanperspec_lstsq,
             'Nspec per Nyq. sample for lstsq extraction'),
            end=True)
        self.hdr.append(('R', self.R, 'Spectral resolution'), end=True)
