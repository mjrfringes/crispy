import numpy as np
import astropy.units as u
import astropy.constants as c
from crispy.tools.inputScene import convert_krist_cube, calc_contrast, calc_contrast_Bijan, zodi_cube, adjust_krist_header
import glob
from crispy.IFS import reduceIFSMap, polychromeIFS
from crispy.tools.initLogger import getLogger
log = getLogger('crispy')
from crispy.tools.image import Image
try:
    from astropy.io import fits
except BaseException:
    import pyfits as fits
from time import time
import os
from crispy.tools.detector import averageDetectorReadout, averageDetectorReadoutParallel, readDetector, calculateDark
import matplotlib.pyplot as plt

import multiprocessing
from crispy.tools.par_utils import Task, Consumer
import seaborn as sns
from crispy.tools.inputScene import calc_contrast
from crispy.tools.reduction import calculateWaveList
from crispy.tools.imgtools import bowtie, scale2imgs, circularMask
from crispy.tools.rotate import shiftCube
from scipy import ndimage
from scipy.interpolate import interp1d
import scipy


def process_SPC_IFS(par,
                    psf_time_series_folder,
                    offaxis_psf_filename,
                    planet_radius=1.27 * c.R_jup,
                    planet_AU=3.6, planet_dist_pc=14.1,
                    ref_star_T=9377 * u.K, ref_star_Vmag=2.37,
                    target_star_T=5887 * u.K, target_star_Vmag=5.03,
                    lamc=770., BW=0.18, n_ref_star_imgs=30,
                    tel_pupil_area=3.650265060424805 * u.m**2,
                    IWA=3, OWA=9,
                    xshift=0.0, yshift=0.0,  # need to tie this to planet distance
                    pp_fact=0.05,
                    t_zodi=0.09,
                    useQE=True,
                    subtract_ref_psf=True,
                    outdir_time_series='OS5',
                    outdir_detector='OS5/OS5_detector',
                    outdir_average='OS5/OS5_average',
                    process_cubes=True,
                    process_offaxis=True,
                    process_detector=True,
                    process_noiseless=False,
                    take_averages=True,
                    parallel=True):
    '''
    Process SPC PSF cubes from J. Krist through the IFS

    Parameters
    ----------
    par: Params instance
        Contains all the parameters of the IFS
    psf_time_series_folder: string
        Where the files from Krist are located
    offaxis_psf_filename: string
        Where the off-axis PSF is located
    mean_contrast: float
        Mean contrast of the planet spectrum
    ref_star_T: `u.K` float
        Reference star temperature, float multiplied by astropy.units.K
    ref_star_Vmag: float
        Vmag of the reference star
    target_star_T: `u.K` float
        Target star temperature, float multiplied by astropy.units.K
    target_star_T_star_Vmag: float
        Vmag of the target star
    lamc: float
        Band's central wavelength in nm
    BW: float
        Bandwidth
    n_ref_star_imgs: int
        Number of reference star images in the list
    tel_pupil_area: `u.m**2` float
        Effective surface area of telescope, including all geometric losses. Float
        multiplied by astropy.units.m**2
    IWA: float
        Inner working angle defined at par.lenslet_wav
    OWA: float
        Outer working angle defined at par.lenslet_wav
    pp_fact: float
        Post-processing factor - multiplies the target star PSF
    t_zodi = float
        Zodi transmission
    outdir_time_series: string
        Where to store the noiseless IFS detector images
    outdir_detector: string
        Where to store the noisy IFS detector images
    outdir_average: string
        Where to store the averages of the time series
    process_cubes: Boolean
        Whether to process the raw images from Krist, or skip this step if it was already done
    process_offaxis: Boolean
        Whether to process the offaxis images and add it to the images
    process_detector: Boolean
        Whether to add detector QE, IFS losses and noise to the IFS images
    process_noiseless: Boolean
        Whether to add detector QE, IFS losses but no detector noise to the IFS images
    take_averages: Boolean
        Whether to average all IFS detector images in the time series to create averages

    Returns
    -------
    signal: ndarray
        Array with the matched-filtered flux at each of the final cube's wavelengths;
        The background is already subtracted
    noise: ndarray
        Noise estimate using the pixel-to-pixel variance within the dark hole, multiplied
        by the number of effective pixels within the matched filter (sum of matched filter
        cube)


    '''

    times = {'Start': time()}

    ##########################################################################
    # Step 1: Convert all the cubes to photons/seconds
    ##########################################################################

    # load the filenames
    filelist = sorted(glob.glob(psf_time_series_folder + '/*'))

    # load first filelist to get its shape
    kristfile = Image(filename=filelist[0])
    fileshape = kristfile.data.shape
    adjust_krist_header(kristfile, lamc=lamc)

    Nlam = fileshape[0]
    lamlist = lamc * np.linspace(1. - BW / 2., 1. + BW / 2., Nlam) * u.nm

    # reference and target star cube conversions
    ref_star_cube = convert_krist_cube(
        fileshape,
        lamlist,
        ref_star_T,
        ref_star_Vmag,
        tel_pupil_area)
    target_star_cube = convert_krist_cube(
        fileshape,
        lamlist,
        target_star_T,
        target_star_Vmag,
        tel_pupil_area)

    ##########################################################################
    # Step 2: Process all the cubes and directly apply detector
    ##########################################################################
    ref_outlist = []
    target_outlist = []
    # Parallelized propagation
    if parallel:
        if process_cubes:
            tasks = multiprocessing.Queue()
            results = multiprocessing.Queue()
            ncpus = multiprocessing.cpu_count()
            consumers = [Consumer(tasks, results)
                         for i in range(ncpus)]
            for w in consumers:
                w.start()

            # you call the function here, with all its arguments in a list
            for i in range(len(filelist)):
                reffile = filelist[i]
                log.info('Processing file ' + reffile.split('/')[-1])
                cube = fits.open(reffile)[0]
                if i < n_ref_star_imgs:
                    cube.data *= ref_star_cube
                else:
                    cube.data *= target_star_cube
                # adjust headers for slightly different wavelength
                log.debug('Modifying cube header')
                adjust_krist_header(cube, lamc=lamc)
                par.saveDetector = False

                tasks.put(Task(i, polychromeIFS, (par, lamlist.value, cube)))

            for i in range(ncpus):
                tasks.put(None)

            for i in range(len(filelist)):
                index, result = results.get()
                reffile = filelist[index]
                if index < n_ref_star_imgs:
                    Image(data=result, header=par.hdr).write(outdir_time_series + '/' + \
                          reffile.split('/')[-1].split('.')[0] + '_refstar_IFS.fits', clobber=True)
                    ref_outlist.append(
                        outdir_time_series + '/' + reffile.split('/')[-1].split('.')[0] + '_refstar_IFS.fits')
                else:
                    Image(data=result, header=par.hdr).write(outdir_time_series + '/' + \
                          reffile.split('/')[-1].split('.')[0] + '_targetstar_IFS.fits', clobber=True)
                    target_outlist.append(
                        outdir_time_series + '/' + reffile.split('/')[-1].split('.')[0] + '_targetstar_IFS.fits')

        else:
            for i in range(len(filelist)):
                reffile = filelist[i]
                if i < n_ref_star_imgs:
                    ref_outlist.append(
                        outdir_time_series + '/' + reffile.split('/')[-1].split('.')[0] + '_refstar_IFS.fits')
                else:
                    target_outlist.append(
                        outdir_time_series + '/' + reffile.split('/')[-1].split('.')[0] + '_targetstar_IFS.fits')

    else:
        for i in range(len(filelist)):
            reffile = filelist[i]
            if process_cubes:
                log.info('Processing file ' + reffile.split('/')[-1])
                cube = fits.open(reffile)[0]
                if i < n_ref_star_imgs:
                    cube.data *= ref_star_cube
                else:
                    cube.data *= target_star_cube
                # adjust headers for slightly different wavelength
                log.debug('Modifying cube header')
                adjust_krist_header(cube, lamc=lamc)
                par.saveDetector = False

                detectorFrame = polychromeIFS(
                    par, lamlist.value, cube, QE=useQE)

                if i < n_ref_star_imgs:
                    Image(data=detectorFrame, header=par.hdr).write(outdir_time_series + '/' + \
                          reffile.split('/')[-1].split('.')[0] + '_refstar_IFS.fits', clobber=True)
                    ref_outlist.append(
                        outdir_time_series + '/' + reffile.split('/')[-1].split('.')[0] + '_refstar_IFS.fits')
                else:
                    Image(data=detectorFrame, header=par.hdr).write(outdir_time_series + '/' + \
                          reffile.split('/')[-1].split('.')[0] + '_targetstar_IFS.fits', clobber=True)
                    target_outlist.append(
                        outdir_time_series + '/' + reffile.split('/')[-1].split('.')[0] + '_targetstar_IFS.fits')
            else:

                if i < n_ref_star_imgs:
                    ref_outlist.append(
                        outdir_time_series + '/' + reffile.split('/')[-1].split('.')[0] + '_refstar_IFS.fits')
                else:
                    target_outlist.append(
                        outdir_time_series + '/' + reffile.split('/')[-1].split('.')[0] + '_targetstar_IFS.fits')

    times['Process cubes through IFS'] = time()

    ##########################################################################
    # Step 3: Process the off-axis PSF in the same way; also process a flipped version
    ##########################################################################

#     first recenter the offaxis psf cube
#     recenter_offaxis(offaxis_psf_filename,threshold=0.01,outname=par.exportDir+'/centered_offaxis.fits')
#
#     now shift the cube to put it at the desired lambda/D
#     input_sampling = kristfile.header['PIXSIZE']
#     input_wav = kristfile.header['LAM_C']*1000.
#     par.pixperlenslet = par.lenslet_sampling/(input_sampling * input_wav/par.lenslet_wav)
#     par.hdr['DX_OFFAX']=xshift*par.pixperlenslet
#     par.hdr['DY_OFFAX']=yshift*par.pixperlenslet
#
#     shifted_cube = shiftCube(cube,dx=sep*kristfile.header['PIXSIZE'],dy=0,order=1)

    if process_offaxis:
        offaxiscube = Image(offaxis_psf_filename)
        print('Processing file ' + offaxis_psf_filename)

        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube.data.shape[1] < fileshape[1]:
            diff = fileshape[1] - offaxiscube.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:, diff // 2:-diff //
                                   2, diff // 2:-diff // 2] += offaxiscube.data
            offaxiscube = Image(
                data=offaxiscube_recentered,
                header=offaxiscube.header)
        offaxis_star_cube = convert_krist_cube(
            offaxiscube.data.shape,
            lamlist,
            target_star_T,
            target_star_Vmag,
            tel_pupil_area)
        contrast = calc_contrast_Bijan(lamlist.value)

        contrast_cube = np.zeros(offaxiscube.data.shape)
        for i in range(offaxiscube.data.shape[0]):
            contrast_cube[i, :, :] += contrast[i]
        offaxiscube.data *= offaxis_star_cube * contrast_cube

        # adjust headers for slightly different wavelength
        log.debug('Modifying cube header')
        adjust_krist_header(offaxiscube, lamc=lamc)
        par.saveDetector = False
        Image(
            data=offaxiscube.data,
            header=offaxiscube.header).write(
            outdir_average +
            '/offaxiscube_processed.fits',
            clobber=True)
        detectorFrame = polychromeIFS(
            par, lamlist.value, offaxiscube, QE=useQE)
        Image(
            data=detectorFrame,
            header=par.hdr).write(
            outdir_average +
            '/offaxis.fits',
            clobber=True)

        offaxiscube_flipped = fits.open(offaxis_psf_filename)[0]
        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube_flipped.data.shape[1] < fileshape[1]:
            diff = fileshape[1] - offaxiscube_flipped.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:, diff // 2:-diff // 2,
                                   diff // 2:-diff // 2] += offaxiscube_flipped.data
            offaxiscube_flipped = Image(
                data=offaxiscube_recentered,
                header=offaxiscube.header)

        for i in range(offaxiscube_flipped.data.shape[0]):
            offaxiscube_flipped.data[i, :, :] = np.fliplr(
                offaxiscube_flipped.data[i, :, :])
        offaxiscube_flipped.data *= offaxis_star_cube * contrast_cube
        detectorFrame = polychromeIFS(
            par, lamlist.value, offaxiscube_flipped, QE=useQE)
        Image(
            data=detectorFrame,
            header=par.hdr).write(
            outdir_average +
            '/offaxis_flipped.fits',
            clobber=True)

    ##########################################################################
    # Add the zodi and exozodi, uniform for now as opposed to decreasing
    # as a function of separation from the star; over the full FOV instead of just the bowtie
    ##########################################################################

    local_zodi_mag = 23
    exozodi_mag = 22
    D = 2.37
    pixarea = kristfile.header['PIXSIZE'] * lamc * 1e-9 / D / 4.848e-6
    absmag = target_star_Vmag - 5 * np.log10(planet_dist_pc / 10.)
    zodicube = zodi_cube(target_star_cube,
                         area_per_pixel=pixarea,
                         absmag=absmag,
                         Vstarmag=target_star_Vmag,
                         zodi_surfmag=23, exozodi_surfmag=22,
                         distAU=planet_AU, t_zodi=t_zodi)

    zodicube = Image(data=zodicube, header=kristfile.header)
    detectorFrame = polychromeIFS(par, lamlist.value, zodicube, QE=useQE)

    times['Cube conversion'] = time()

    Image(
        data=detectorFrame,
        header=par.hdr).write(
        outdir_average +
        '/zodicube.fits',
        clobber=True)

    times['Process off-axis PSF through IFS'] = time()

    ##########################################################################
    # Step 4: Add the off-axis PSF before reading on the detector
    ##########################################################################

    if process_detector:
        # Apply detector for both reference star and target
        ref_det_outlist = averageDetectorReadout(
            par, ref_outlist, outdir_detector)
        offaxis_filename = os.path.abspath(outdir_average + '/offaxis.fits')
        zodi_filename = os.path.abspath(outdir_average + '/zodicube.fits')
        target_det_outlist = averageDetectorReadout(
            par,
            target_outlist,
            outdir_detector,
            offaxis=offaxis_filename,
            factor=pp_fact,
            zodi=zodi_filename)
    else:
        ref_det_outlist = []
        target_det_outlist = []
        suffix = 'detector'
        for reffile in ref_outlist:
            ref_det_outlist.append(
                outdir_detector + '/' + reffile.split('/')[-1].split('.')[0] + '_' + suffix + '.fits')
        for reffile in target_outlist:
            target_det_outlist.append(
                outdir_detector + '/' + reffile.split('/')[-1].split('.')[0] + '_' + suffix + '.fits')
    times['Construct IFS detector'] = time()

    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(('comment', '*' * 22 + ' Scene ' + '*' * 20), end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(('comment', ''), end=True)
    try:
        par.hdr.append(('TZODI', t_zodi, 'Zodi throughput'), end=True)
        par.hdr.append(
            ('ABSMAG',
             absmag,
             'Target star absolute magnitude'),
            end=True)
    except BaseException:
        pass

    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(
        ('comment',
         '*' *
         22 +
         ' Postprocessing ' +
         '*' *
         20),
        end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('PPFACT', pp_fact, 'Post-processing factor'), end=True)

    ##########################################################################
    # Take averages
    ##########################################################################

    if take_averages:
        log.info('Taking averages')
        ref_star_average = np.zeros(
            Image(filename=ref_det_outlist[0]).data.shape)
        target_star_average = np.zeros(
            Image(filename=target_det_outlist[0]).data.shape)
        for reffile in ref_det_outlist:
            ref_star_average += Image(filename=reffile).data
        #ref_star_average /= len(ref_det_outlist)
        for reffile in target_det_outlist:
            target_star_average += Image(filename=reffile).data
        Image(
            data=ref_star_average,
            header=par.hdr).write(
            outdir_average +
            '/average_ref_star_detector.fits',
            clobber=True)
        Image(
            data=target_star_average,
            header=par.hdr).write(
            outdir_average +
            '/average_target_star_detector.fits',
            clobber=True)
        par.hdr.append(
            ('NIMGS',
             len(target_det_outlist),
                'Number of time series steps'),
            end=True)
        par.hdr.append(
            ('TOTTIME',
             len(target_det_outlist) *
                par.timeframe,
                'Total integration time on source'),
            end=True)

    times['Taking averages'] = time()

    ##########################################################################
    # Process the cubes
    ##########################################################################
    img = Image(
        filename=os.path.abspath(
            outdir_average +
            '/average_ref_star_detector.fits'))
    ref_cube = reduceIFSMap(
        par,
        os.path.abspath(
            outdir_average +
            '/average_ref_star_detector.fits'))
    target_cube = reduceIFSMap(
        par,
        os.path.abspath(
            outdir_average +
            '/average_target_star_detector.fits'))
    offaxis_ideal = reduceIFSMap(
        par, os.path.abspath(
            outdir_average + '/offaxis.fits'))
    offaxis_ideal_flipped = reduceIFSMap(
        par, os.path.abspath(
            outdir_average + '/offaxis_flipped.fits'))

    ##########################################################################
    # Construct a bowtie mask
    ##########################################################################
    ydim, xdim = target_cube[0].shape
    mask, scratch = bowtie(target_cube[0], ydim //
                           2, xdim //
                           2, openingAngle=65, clocking=-
                           par.philens *
                           180. /
                           np.pi, IWApix=IWA *
                           lamc /
                           par.lenslet_wav /
                           par.lenslet_sampling, OWApix=OWA *
                           lamc /
                           par.lenslet_wav /
                           par.lenslet_sampling, export=None, twomasks=False)

    times['Process average cubes'] = time()
    ##########################################################################
    # Least-squares slice-by-slice subtraction
    ##########################################################################
    if subtract_ref_psf:
        #         ref_cube_stack = np.sum(ref_cube.data,axis=0)
        #         target_cube_stack = np.sum(target_cube.data,axis=0)
        #         ratio = np.sum(target_star_cube[:,0,0]) / np.sum(ref_star_cube[:,0,0])
        #         residual = target_cube.data - ratio*ref_cube.data
        #         residual[np.isnan(target_cube.data)] = np.NaN
        #         residual[(residual>1e10)*(residual<-1e10)] = np.NaN

        # NEED TO subtract the mean of the cube slice by slice before the lstsq
        # step

        # residual is target_cube-coefs*ref_cube
        coefs, residual = scale2imgs(
            target_cube, ref_cube, mask=mask, returndiff=True)
        par.hdr.append(
            ('comment', 'Subtracted scaled mean of reference star PSF'), end=True)
    else:
        residual = target_cube.data
    Image(
        data=residual,
        header=par.hdr).write(
        outdir_average +
        '/residual.fits',
        clobber=True)
    Image(data=np.sum(residual, axis=0), header=par.hdr).write(
        outdir_average + '/residual_stack.fits', clobber=True)

    times['Normalize and subtract reference PSF'] = time()

    ##########################################################################
    # Flatfielding, to account for small errors in wavelength calibration
    ##########################################################################

    flatfield = Image(par.exportDir + '/flatfield_red_optext.fits')
    residual[~np.isnan(residual)] /= flatfield.data[~np.isnan(residual)]
    residual[np.logical_or((residual > 1e10), (residual < -1e10))] = np.NaN
    par.hdr.append(('comment', 'Divided by lenslet flatfield'), end=True)
    Image(
        data=residual,
        header=par.hdr).write(
        outdir_average +
        '/residual_flatfielded.fits',
        clobber=True)
    Image(
        data=np.sum(
            residual,
            axis=0),
        header=par.hdr).write(
            outdir_average +
            '/residual_flatfielded_stack.fits',
        clobber=True)

    ##########################################################################
    # Construct a matched filter
    ##########################################################################

    #
#     # loop over all the slices in the cube:
#     matched_filter = np.zeros(residual.shape)
#     matched_filter_flipped = np.zeros(residual.shape)
#     signal = np.zeros(residual.shape[0]) # on source
#     off = np.zeros(residual.shape[0])   # off source
#     mf_npix = np.zeros(residual.shape[0]) # effective background area of matched filter
#     noise = np.zeros(residual.shape[0]) # noise
#     for slicenum in range(residual.shape[0]):
#         # ON
#         offaxis_ideal_norm = offaxis_ideal.data[slicenum]/np.nansum(offaxis_ideal.data[slicenum])
#         this_slice = offaxis_ideal_norm/np.nansum((offaxis_ideal_norm)**2)
#         # calculate correction factor since we are going to crop only the top the of the hat
#         # this is the inverse ratio of the contribution of the brightest pixels over the rest
#         aper_phot = np.nansum(this_slice)/np.nansum(this_slice[this_slice>1.0])
#         # Set all low-contributing pixels to 0.0
#         this_slice[this_slice<1.0] = 0.0
#         matched_filter[slicenum,:,:] = this_slice
#         # Multiply what is left by that aperture correction factor
#         matched_filter[slicenum,:,:]*=aper_phot
#         signal[slicenum] = np.nansum(matched_filter[slicenum,:,:]*residual[slicenum,:,:])
#         # OFF
#         offaxis_ideal_flipped_norm = offaxis_ideal_flipped.data[slicenum]/np.nansum(offaxis_ideal_flipped.data[slicenum])
#         this_slice = offaxis_ideal_flipped_norm/np.nansum((offaxis_ideal_flipped_norm)**2)
#         aper_phot = np.nansum(this_slice)/np.nansum(this_slice[this_slice>1.0])
#         this_slice[this_slice<1.0] = 0.0
#         matched_filter_flipped[slicenum,:,:] = this_slice
#         matched_filter_flipped[slicenum,:,:]*=aper_phot
#         off[slicenum] = np.nansum(matched_filter_flipped[slicenum,:,:]*residual[slicenum,:,:])
#     mf_npix = np.nansum(np.nansum(matched_filter,axis=2),axis=1)
#
#     ###################################################################################
#     # Step 9: Determine the pixel noise in the dark hole
#     ###################################################################################
#
#     # PSF is in the right mask
#     pixstd = [np.nanstd(residual[i,:,:]*maskright) for i in range(residual.shape[0])]
#     noiseright = np.sqrt(2*mf_npix)*pixstd # twice the num of pix since we subtract the off field
#     pixstd = [np.nanstd(residual[i,:,:]*maskleft) for i in range(residual.shape[0])]
# noiseleft = np.sqrt(2*mf_npix)*pixstd # twice the num of pix since we
# subtract the off field

    Image(
        data=matched_filter).write(
        outdir_average +
        '/matched_filter.fits',
        clobber=True)
    Image(
        data=matched_filter_flipped).write(
        outdir_average +
        '/matched_filter_flipped.fits',
        clobber=True)

    times['Computed signal and noise arrays'] = time()

    log.info(
        'Cube conversion: %.3f' %
        (times['Cube conversion'] - times['Start']))
    log.info('Process cubes through IFS: %.3f' %
             (times['Process cubes through IFS'] - times['Cube conversion']))
    log.info(
        'Process off-axis PSF through IFS: %.3f' %
        (times['Process off-axis PSF through IFS'] -
         times['Process cubes through IFS']))
    log.info(
        'Construct IFS detector: %.3f' %
        (times['Construct IFS detector'] -
         times['Process off-axis PSF through IFS']))
    log.info(
        'Taking averages: %.3f' %
        (times['Taking averages'] -
         times['Construct IFS detector']))
    log.info(
        'Process average cubes: %.3f' %
        (times['Process average cubes'] -
         times['Taking averages']))
    log.info(
        'Normalize and subtract reference PSF: %.3f' %
        (times['Normalize and subtract reference PSF'] -
         times['Process average cubes']))
    log.info('Computed signal and noise arrays: %.3f' %
             (times['Computed signal and noise arrays'] -
              times['Normalize and subtract reference PSF']))
    log.info(
        'Total time: %.3f' %
        (times['Computed signal and noise arrays'] - times['Start']))

    return signal - off, noiseright, noiseleft


def process_SPC_IFS2(par,
                     psf_time_series_folder,
                     offaxis_psf_filename,
                     xshift=0.0, yshift=0.0, order=3,
                     albedo_filename='Jupiter_1x_5AU_90deg.dat',
                     planet_radius=1.27,
                     planet_AU=3.6, planet_dist_pc=14.1,
                     albedo=0.28,
                     ref_star_T=9377 * u.K, ref_star_Vmag=2.37,
                     target_star_T=5887 * u.K, target_star_Vmag=5.03,
                     lamc=770., BW=0.18, n_ref_star_imgs=30,
                     tel_pupil_area=3.650265060424805 * u.m**2,
                     IWA=3, OWA=9,
                     forced_inttime_ref=10.0,
                     forced_tottime_ref=1000.0,
                     pp_fact=1.00,
                     RDI=True,
                     subtract_dark=False,
                     normalize_cubes=True,
                     t_zodi=0.09,
                     useQE=True,
                     mflib='/mflib.fits.gz',
                     outdir_time_series='OS5',
                     outdir_detector='OS5/OS5_detector',
                     outdir_average='OS5/OS5_average',
                     process_cubes=True,
                     process_offaxis_files=True,
                     process_detector=True,
                     take_averages=True,
                     take_ref_average=True,
                     nosource=True):
    '''
    Process SPC PSF cubes from J. Krist through the IFS

    Parameters
    ----------
    par: Params instance
        Contains all the parameters of the IFS
    psf_time_series_folder: string
        Where the files from Krist are located
    offaxis_psf_filename: string
        Where the off-axis PSF is located
    xshift: float
        X-shift of the reference star with respect to the target star(pixels)
    yshift: float
        Y-shift of the reference star with respect to the target star(pixels)
    order: int
        Order of all interpolations (3 recommended)
    albedo_filename: string
        Name of file containing the geometric albedo of the planet, located in Inputs/ folder
    planet_radius: float
        Radius of planet in Jupiter radii
    planet_AU: float
        Distance of the planet from its star in AU
    planet_dist_pc: float
        Distance of the star system in pc
    albedo: float
        Overrides the albedo. If None, then the original albedo curve is used. If not None,
        then the maximum of the albedo curve is normalized to the new albedo, while keeping
        overall albedo shape
    ref_star_T: `u.K` float
        Reference star temperature, float multiplied by astropy.units.K
    ref_star_Vmag: float
        Vmag of the reference star
    target_star_T: `u.K` float
        Target star temperature, float multiplied by astropy.units.K
    target_star_T_star_Vmag: float
        Vmag of the target star
    lamc: float
        Band's central wavelength in nm
    BW: float
        Bandwidth
    n_ref_star_imgs: int
        Number of reference star images in the list
    tel_pupil_area: `u.m**2` float
        Effective surface area of telescope, including all geometric losses. Float
        multiplied by astropy.units.m**2
    IWA: float
        Inner working angle defined at par.lenslet_wav
    OWA: float
        Outer working angle defined at par.lenslet_wav
    forced_inttime_ref: float
        Forced time of a single integration for the reference star. This should be smaller
        than the individual integration time for the target since the reference star
        is typically brighter
    forced_tottime_ref: float
        Forced time spent for a single input cube for the reference. Typically, this is the
        same as the time for a single input cube for the target
    pp_fact: float
        Post-processing factor - artificially multiplies the target star PSF. Note that
        this is NOT the same as Bijan's f_pp
    RDI: Boolean
        If True, run an RDI function to clean up the raw cubes.
    mflib: String
        Path towards an already-existing matched filter library. If empty string is given,
        the software will construct the library each time
    subtract_dark: Boolean
        If True, estimate the mean dark region (bottom of the detector) and subtract from
        map. Default: True. For different kinds of input maps that might cover the entire
        defector, this might be more tricky.
    normalize_cubes: Boolean
        If True, all cubes are normalized to contrast units. Default is True.
    t_zodi: float
        Zodi transmission (not operational for now)
    useQE: Boolean
        If True, the QE is applied when constructing the IFS maps. Note that in this case
        the QE file should represent both the actual detector QE, and the wavelength-dependent
        optical losses
    outdir_time_series: string
        Where to store the noiseless IFS detector images
    outdir_detector: string
        Where to store the noisy IFS detector images
    outdir_average: string
        Where to store the averages of the time series
    process_cubes: Boolean
        Whether to process the raw images from Krist, or skip this step if it was already done
    process_offaxis: Boolean
        Whether to process the offaxis images and add it to the images
    process_detector: Boolean
        Whether to add detector QE, IFS losses and noise to the IFS images
    process_noiseless: Boolean
        Whether to add detector QE, IFS losses but no detector noise to the IFS images
    take_averages: Boolean
        Whether to average all IFS detector images in the time series to create averages
    nosource: Boolean
        Whether or not to process everything without actually adding the planet

    Returns
    -------
    signal: ndarray
        Array with the matched-filtered flux at each of the final cube's wavelengths;
        The background is already subtracted
    noise: ndarray
        Noise estimate using the pixel-to-pixel variance within the dark hole, multiplied
        by the number of effective pixels within the matched filter (sum of matched filter
        cube)


    '''
    times = {'Start': time()}

    ##########################################################################
    # Convert all the cubes to photons/seconds and propagate them through the IFS
    ##########################################################################

    filelist = sorted(glob.glob(psf_time_series_folder + '/*.fits'))

    reffiles = filelist[:n_ref_star_imgs]
    targetfiles = filelist[n_ref_star_imgs:]

    ref_outlist = processReferenceCubes(
        par,
        xshift=xshift,
        yshift=yshift,
        order=order,
        outdir_time_series=outdir_time_series,
        ref_input_list=reffiles,
        process_cubes=process_cubes,
        ref_star_T=ref_star_T,
        ref_star_Vmag=ref_star_Vmag,
        lamc=lamc,
        BW=BW,
        tel_pupil_area=tel_pupil_area)

    target_outlist = processTargetCubes(
        par,
        targetfiles,
        outdir_time_series=outdir_time_series,
        process_cubes=process_cubes,
        target_star_T=target_star_T,
        target_star_Vmag=target_star_Vmag,
        lamc=lamc,
        BW=BW,
        tel_pupil_area=tel_pupil_area)

    times['Process cubes through IFS'] = time()

    ##########################################################################
    # Process the off-axis PSF in the same way
    ##########################################################################

    fileshape = fits.open(reffiles[0])[0].data.shape
    lamlist = lamc * np.linspace(1. - BW / 2.,
                                 1. + BW / 2., fileshape[0]) * u.nm
    if process_offaxis_files:

        # planetary off-axis source
        #         process_planet(par,offaxis_psf_filename,fileshape=fileshape,
        #                     lamlist=lamlist,
        #                     lamc=lamc,
        #                     outdir_average=outdir_average,
        #                     planet_radius = planet_radius,
        #                     planet_AU = planet_AU,planet_dist_pc=planet_dist_pc,
        #                     target_star_T=target_star_T, target_star_Vmag=target_star_Vmag,
        #                     tel_pupil_area=tel_pupil_area, order=order)

        # stellar off-axis source for contrast normalization - watch out for
        # the photon counting issues!
        process_offaxis(
            par,
            offaxis_psf_filename=offaxis_psf_filename,
            fileshape=fileshape,
            lamlist=lamlist,
            lamc=lamc,
            filename=par.codeRoot +
            '/Inputs/' +
            albedo_filename,
            albedo=albedo,
            planet_radius=planet_radius,
            planet_AU=planet_AU,
            planet_dist_pc=planet_dist_pc,
            inttime=1,
            Nreads=1,
            Nave=1,
            outdir_average=outdir_average,
            target_star_T=target_star_T,
            target_star_Vmag=target_star_Vmag,
            tel_pupil_area=tel_pupil_area)

    times['Cube conversion'] = time()

    ##########################################################################
    # Add the zodi and exozodi, uniform for now as opposed to decreasing
    # as a function of separation from the star; over the full FOV instead of just the bowtie
    ##########################################################################

#     local_zodi_mag = 23
#     exozodi_mag = 22
#     D = 2.37
#     pixarea = kristfile.header['PIXSIZE']*lamc*1e-9/D/4.848e-6
#     absmag = target_star_Vmag-5*np.log10(planet_dist_pc/10.)
#     zodicube = zodi_cube(target_star_cube,
#                         area_per_pixel=pixarea,
#                         absmag=absmag,
#                         Vstarmag=target_star_Vmag,
#                         zodi_surfmag=23,exozodi_surfmag=22,
#                         distAU=planet_AU,t_zodi=t_zodi)
#
#     zodicube = Image(data=zodicube,header=kristfile.header)
#     detectorFrame = polychromeIFS(par,lamlist.value,zodicube,QE=useQE)


#     Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/zodicube.fits',clobber=True)

    times['Process off-axis PSF through IFS'] = time()

    ##########################################################################
    # Add the off-axis PSF before reading on the detector
    ##########################################################################

    if process_detector:
        log.info("Taking average of reference star")
        ref_det_outlist = averageDetectorReadoutParallel(
            par,
            ref_outlist,
            outdir_detector,
            forced_inttime=forced_inttime_ref,
            forced_tottime=forced_tottime_ref)
        offaxis_filename = os.path.abspath(
            outdir_average + '/offaxis_planet.fits')
        if nosource:
            log.info("Taking average of target star without planet")
            target_nosource_outlist = averageDetectorReadout(
                par, target_outlist, outdir_detector, suffix='nosource_detector', factor=pp_fact)
        log.info("Taking average of target star with planet")
        target_det_outlist = averageDetectorReadoutParallel(
            par, target_outlist, outdir_detector, offaxis=offaxis_filename, factor=pp_fact)
    else:
        ref_det_outlist = []
        if nosource:
            target_nosource_outlist = []
        target_det_outlist = []
        suffix = 'detector'
        for reffile in ref_outlist:
            ref_det_outlist.append(
                outdir_detector + '/' + reffile.split('/')[-1].split('.')[0] + '_' + suffix + '.fits')
        for reffile in target_outlist:
            if nosource:
                target_nosource_outlist.append(
                    outdir_detector + '/' + reffile.split('/')[-1].split('.')[0] + '_nosource_detector.fits')
            target_det_outlist.append(
                outdir_detector + '/' + reffile.split('/')[-1].split('.')[0] + '_' + suffix + '.fits')
    times['Construct IFS detector'] = time()

    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(
        ('comment',
         '*' *
         22 +
         ' Postprocessing ' +
         '*' *
         20),
        end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('PPFACT', pp_fact, 'Post-processing factor'), end=True)

    ##########################################################################
    # Take averages
    ##########################################################################

    if take_averages:
        log.info('Taking averages')
        ref_star_average = np.zeros(
            Image(filename=ref_det_outlist[0]).data.shape)
        target_star_average = np.zeros(
            Image(filename=target_det_outlist[0]).data.shape)
        if nosource:
            target_star_nosource_average = np.zeros(
                Image(filename=target_nosource_outlist[0]).data.shape)
        for reffile in ref_det_outlist:
            ref_star_average += Image(filename=reffile).data
        ref_star_average /= len(ref_det_outlist) * \
            Image(filename=ref_det_outlist[0]).header['NREADS']
        log.info('Divide total image sum by %f times %f' %
                 (len(ref_det_outlist), Image(filename=reffile).header['NREADS']))
        if par.PCmode:
            ref_star_average *= np.exp(par.RN * par.threshold / par.EMGain)
            ref_star_average = -np.log(1. - ref_star_average)
        ref_star_average /= Image(
            filename=ref_det_outlist[0]).header['INTTIME']

        for reffile in target_det_outlist:
            target_star_average += Image(filename=reffile).data
        target_star_average /= len(target_det_outlist) * \
            Image(filename=reffile).header['NREADS']
        log.info(
            'Divide total image sum by %f times %f' %
            (len(target_det_outlist), Image(
                filename=reffile).header['NREADS']))
        if par.PCmode:
            target_star_average *= np.exp(par.RN * par.threshold / par.EMGain)
            target_star_average = -np.log(1. - target_star_average)
        target_star_average /= Image(filename=reffile).header['INTTIME']

        if nosource:
            for reffile in target_nosource_outlist:
                target_star_nosource_average += Image(filename=reffile).data
            target_star_nosource_average /= len(target_nosource_outlist) * Image(
                filename=reffile).header['NREADS']
            log.info(
                'Divide total image sum by %f times %f' %
                (len(target_nosource_outlist), Image(
                    filename=reffile).header['NREADS']))
            if par.PCmode:
                target_star_nosource_average *= np.exp(
                    par.RN * par.threshold / par.EMGain)
                target_star_nosource_average = - \
                    np.log(1. - target_star_nosource_average)
            target_star_nosource_average /= Image(
                filename=reffile).header['INTTIME']

        Image(
            data=ref_star_average,
            header=par.hdr).write(
            outdir_average +
            '/average_ref_star_detector.fits',
            clobber=True)
        Image(
            data=target_star_average,
            header=par.hdr).write(
            outdir_average +
            '/average_target_star_detector.fits',
            clobber=True)
        if nosource:
            Image(
                data=target_star_nosource_average,
                header=par.hdr).write(
                outdir_average +
                '/average_target_star_nosource_detector.fits',
                clobber=True)
    else:
        ref_star_average = Image(
            outdir_average +
            '/average_ref_star_detector.fits').data
        target_star_average = Image(
            outdir_average +
            '/average_target_star_detector.fits').data
        if nosource:
            target_star_nosource_average = Image(
                outdir_average + '/average_target_star_nosource_detector.fits').data

    if "NREADS" not in par.hdr:
        par.hdr.append(
            ('NREADS',
             par.Nreads,
             'Number of subframes co-added per image'),
            end=True)
        par.hdr.append(
            ('EXPTIME',
             par.timeframe,
             'Total exposure time for number of frames'),
            end=True)
    par.hdr.append(
        ('NIMGS',
         len(target_det_outlist),
         'Number of time series steps'),
        end=True)
    par.hdr.append(
        ('TOTTIME',
         len(target_det_outlist) *
         par.timeframe,
         'Total integration time on source'),
        end=True)

    ##########################################################################
    # Subtract average, supposed known
    ##########################################################################
    if subtract_dark:
        log.info("Mean of average reference map: %f" %
                 np.mean(ref_star_average[:100, :]))
        ref_star_average -= np.mean(ref_star_average[:100, :])
        log.info("Mean of average target map: %f" %
                 np.mean(target_star_average[:100, :]))
        target_star_average -= np.mean(target_star_average[:100, :])
        if nosource:
            target_star_nosource_average -= np.mean(
                target_star_nosource_average[:100, :])
        par.hdr.append(
            ('SUBDARK',
             subtract_dark,
             'Subtract dark frame?'),
            end=True)
        Image(
            data=ref_star_average,
            header=par.hdr).write(
            outdir_average +
            '/average_ref_star_detector_darksub.fits',
            clobber=True)
        Image(
            data=target_star_average,
            header=par.hdr).write(
            outdir_average +
            '/average_target_star_detector_darksub.fits',
            clobber=True)
        if nosource:
            Image(
                data=target_star_nosource_average,
                header=par.hdr).write(
                outdir_average +
                '/average_target_star_nosource_detector_darksub.fits',
                clobber=True)
    else:
        ref_star_average = Image(outdir_average +
                                 '/average_ref_star_detector_darksub.fits').data
        target_star_average = Image(
            outdir_average +
            '/average_target_star_detector_darksub.fits').data
        if nosource:
            target_star_nosource_average = Image(
                outdir_average + '/average_target_star_nosource_detector_darksub.fits').data

    times['Taking averages'] = time()

    ##########################################################################
    # Reduce the cubes
    ##########################################################################
    ref_reduced = reduceIFSMap(par, ref_star_average)
    target_reduced = reduceIFSMap(par, target_star_average)
    if nosource:
        target_nosource_reduced = reduceIFSMap(
            par, target_star_nosource_average)


#     flatfield = Image(par.exportDir+'/flatfield_red_optext.fits')
#     ref_reduced.data[~np.isnan(ref_reduced.data)] /= flatfield.data[~np.isnan(ref_reduced.data)]
#     target_reduced.data[~np.isnan(target_reduced.data)] /= flatfield.data[~np.isnan(target_reduced.data)]
#     target_nosource_reduced.data[~np.isnan(target_nosource_reduced.data)] /= flatfield.data[~np.isnan(target_nosource_reduced.data)]
#     par.hdr.append(('comment', 'Divided by lenslet flatfield'), end=True)
    if normalize_cubes:
        log.info("Normalizing cubes...")
#         offaxiscube_star_processed = Image(outdir_average+'/offaxiscube_star_processed.fits')
#         input_flux_density = scipy.interpolate.interp1d(lamlist,np.nansum(np.nansum(offaxiscube_star_processed.data,axis=2),axis=1))
        offaxis_star_cube = Image(
            outdir_average +
            "/offaxis_star_red_optext.fits")
        offaxis_ref_star_cube = Image(
            outdir_average + "/offaxis_ref_star_red_optext.fits")
        lam_midpts, waveList = calculateWaveList(par, method='optext')
        dlam = (waveList[-1] - waveList[0]) / (len(waveList) - 1)
#         ratio = input_flux_density(lam_midpts)/np.nansum(np.nansum(offaxis_detector_cube.data,axis=2),axis=1)/dlam
#         ratio = 1./np.nansum(np.nansum(offaxis_star_cube.data,axis=2),axis=1)
        ratio = 1. / \
            np.nanmax(np.nanmax(offaxis_star_cube.data, axis=2), axis=1)
        ratio_ref = 1. / \
            np.nanmax(np.nanmax(offaxis_ref_star_cube.data, axis=2), axis=1)
        ref_reduced.data *= ratio_ref[:, np.newaxis, np.newaxis]
        target_reduced.data *= ratio[:, np.newaxis, np.newaxis]
        if nosource:
            target_nosource_reduced.data *= ratio[:, np.newaxis, np.newaxis]
        par.hdr.append(
            ('comment', 'Normalized cubes to photons per sec per nm'), end=True)

    Image(
        data=ref_reduced.data,
        header=par.hdr).write(
        outdir_average +
        '/average_ref_star_detector_red_optext_flatfielded.fits')
    Image(
        data=target_reduced.data,
        header=par.hdr).write(
        outdir_average +
        '/average_target_star_detector_red_optext_flatfielded.fits')
    if nosource:
        Image(
            data=target_nosource_reduced.data,
            header=par.hdr).write(
            outdir_average +
            '/average_target_star_nosource_detector_red_optext_flatfielded.fits')

    times['Process average cubes'] = time()

    ##########################################################################
    # Do basic least squares RDI, slice by slice with trimmed mean subtraction
    ##########################################################################
    ydim, xdim = target_reduced.data[0].shape
    mask, scratch = bowtie(target_reduced.data[0], ydim //
                           2 -
                           1, xdim //
                           2, openingAngle=65, clocking=-
                           par.philens *
                           180. /
                           np.pi, IWApix=IWA *
                           lamc /
                           par.lenslet_wav /
                           par.lenslet_sampling, OWApix=OWA *
                           lamc /
                           par.lenslet_wav /
                           par.lenslet_sampling, export=None, twomasks=False)
    maskleft, maskright = bowtie(target_reduced.data[0], ydim //
                                 2 -
                                 1, xdim //
                                 2, openingAngle=65, clocking=-
                                 par.philens *
                                 180. /
                                 np.pi, IWApix=IWA *
                                 lamc /
                                 par.lenslet_wav /
                                 par.lenslet_sampling, OWApix=OWA *
                                 lamc /
                                 par.lenslet_wav /
                                 par.lenslet_sampling, export=None, twomasks=True)

    if RDI:
        log.info("Doing RDI")
        # do least square subtraction on image without the source (Neil's idea)
        if nosource:
            coefs_scratch, est_nosource = scale2imgs(target_nosource_reduced,
                                                     ref_reduced,
                                                     bowtie_mask=mask,
                                                     returndiff=False,
                                                     returnest=True)
        coefs, est = scale2imgs(target_reduced,
                                ref_reduced,
                                bowtie_mask=maskleft,
                                returndiff=False,
                                returnest=True)
        par.hdr.append(('comment', 'Applied RDI'), end=True)
        residual = target_reduced.data - est
        if nosource:
            residual_nosource = target_reduced.data - est_nosource

    else:
        #         for i in range(target_reduced.data.shape[0]):
        #             target_reduced.data[i] -= np.mean(target_reduced.data[i][mean_pixel_mask])#scipy.stats.trim_mean(target_reduced.data[i][target_reduced.data[i]>0.0],0.4)
        #         for i in range(target_nosource_reduced.data.shape[0]):
        # target_nosource_reduced.data[i] -=
        # np.mean(target_nosource_reduced.data[i][mean_pixel_mask])#scipy.stats.trim_mean(target_reduced.data[i][target_reduced.data[i]>0.0],0.4)
        residual = target_reduced.data  # - target_nosource_reduced.data
        if nosource:
            residual_nosource = target_nosource_reduced.data

    # mask if this is not already done
#     residual *=mask
#     residual_nosource *=mask
    Image(
        data=residual,
        header=par.hdr).write(
        outdir_average +
        "/lstsq_residual.fits")

    #residual[~np.isnan(residual)] /= flatfield.data[~np.isnan(residual)]
    # Image(data=residual*mask,header=par.hdr).write(outdir_average+"/lstsq_residual_flatfielded.fits")
    if nosource:
        Image(
            data=residual_nosource,
            header=par.hdr).write(
            outdir_average +
            "/lstsq_residual_nosource.fits")

    # matched filter attempt
#     offaxis_ideal = Image(outdir_average+'/offaxis_planet_red_optext.fits')
#     offaxis_ideal_flipped = Image(outdir_average+'/offaxis_flipped_planet_red_optext.fits')
#     matched_filter = mf(offaxis_ideal,mask,0.20)
#     Image(data=matched_filter,header=par.hdr).write(outdir_average+'/matched_filter.fits')
#     matched_filter_flipped = mf(offaxis_ideal_flipped,mask,0.20)
# signal = np.nansum(np.nansum(matched_filter*residual,axis=2),axis=1)# -
# np.nansum(np.nansum(matched_filter_flipped*residual,axis=2),axis=1)

    times['RDI'] = time()
    ##########################################################################
    # Convolve with matched filter
    ##########################################################################
    if mflib == '':
        # re-build the matched filter library
        construct_mflib(
            par,
            planet_cube=outdir_average +
            '/offaxis_planet_red_optext.fits',
            threshold=0.2,
            lamc=lamc,
            BW=BW,
            outdir=outdir_average,
            mask=mask,
            trim=30,
            outname='mflib.fits.gz',
            order=3)
        mflib = outdir_average + '/mflib.fits.gz'

    log.info("Convolving with matched filter")
    convolved = convolved_mf(residual, mflib)
    par.hdr.append(('comment', 'Convolved with matched filter'), end=True)
    Image(
        data=convolved,
        header=par.hdr).write(
        outdir_average +
        '/convolved.fits')
    if nosource:
        convolved_nosource = convolved_mf(residual_nosource, mflib)
        convolved_nosource_reduced = convolved_mf(
            target_nosource_reduced.data, mflib)
        Image(
            data=convolved_nosource,
            header=par.hdr).write(
            outdir_average +
            '/convolved_nosource.fits')
        Image(
            data=convolved_nosource_reduced,
            header=par.hdr).write(
            outdir_average +
            '/convolved_nosource_reduced.fits')
    convolved_target = convolved_mf(target_reduced.data, mflib)
    Image(data=convolved_target, header=par.hdr).write(
        outdir_average + '/convolved_target_reduced.fits')
    convolved_ref = convolved_mf(ref_reduced.data, mflib)
    Image(
        data=convolved_ref,
        header=par.hdr).write(
        outdir_average +
        '/convolved_ref_reduced.fits')

    times['Convolve'] = time()
    ##########################################################################
    # this computes the convolution with an offaxis source as bright as the star
    ##########################################################################
#     if normalize_cubes:
#         log.info("Normalizing contrast")
#         par.hdr.append(('comment', 'Normalized with offaxis PSF from star'), end=True)
#         starmf = convolved_mf(offaxis_detector_cube.data,mflib)
#         Image(data=starmf,header=par.hdr).write(outdir_average+'/starmf.fits')
#         max_starmf = np.amax(np.amax(starmf,axis=2),axis=1)
#         log.info("Max star matched filter:%f" % np.amax(max_starmf))
#     else:
#         max_starmf = np.ones(convolved.shape[0])

#     convolved /= max_starmf[:,np.newaxis,np.newaxis]
#     convolved_nosource /= max_starmf[:,np.newaxis,np.newaxis]
#     convolved_target /= max_starmf[:,np.newaxis,np.newaxis]

    outkey = fits.HDUList(fits.PrimaryHDU(convolved.astype(np.float)))
    outkey.writeto(outdir_average + '/convolved_normalized.fits', clobber=True)
    if nosource:
        outkey = fits.HDUList(
            fits.PrimaryHDU(
                convolved_nosource.astype(
                    np.float)))
        outkey.writeto(
            outdir_average +
            '/convolved_nosource_normalized.fits',
            clobber=True)
    outkey = fits.HDUList(fits.PrimaryHDU(convolved_target.astype(np.float)))
    outkey.writeto(
        outdir_average +
        '/convolved_no_rdi_normalized.fits',
        clobber=True)

    # for the noise, use the scene with no planet
    noise = np.sqrt(2) * [np.nanstd(convolved[i] * maskleft)
                          for i in range(convolved.shape[0])]
    if nosource:
        noise_no_source = np.sqrt(2) * [np.nanstd(convolved_nosource[i] * maskleft)
                                        for i in range(convolved_nosource.shape[0])]
    else:
        noise_no_source = noise
    noise_no_rdi = np.sqrt(2) * [np.nanstd(convolved_target[i] * maskleft)
                                 for i in range(convolved_target.shape[0])]

    # for the signal, just pick where the planet is
    procplanet = Image(outdir_average + '/offaxis_planet_red_optext.fits')
    procplanet.data *= ratio[:, np.newaxis, np.newaxis]
    Image(
        data=procplanet.data,
        header=par.hdr).write(
        outdir_average +
        '/offaxis_planet_red_optext_flatfielded.fits')
    procplanet_convolved = convolved_mf(procplanet.data, mflib)

    yp, xp = np.unravel_index(np.nanargmax(
        procplanet_convolved[procplanet_convolved.shape[0] // 2]), procplanet_convolved[procplanet_convolved.shape[0] // 2].shape)
    log.info("Coordinates of the planet in lenslets: %.2f, %.2f" % (xp, yp))

    signal = convolved[:, yp, xp]
    signal_planet = procplanet_convolved[:, yp, xp]
    signal_star = convolved_ref[:, yp, xp]
    signal_no_rdi = convolved_target[:, yp, xp]
    if nosource:
        signal_no_source = convolved_nosource_reduced[:, yp, xp]
    else:
        signal_no_source = signal_star
    outkey = fits.HDUList(
        fits.PrimaryHDU(
            procplanet_convolved.astype(
                np.float)))
    outkey.writeto(outdir_average + '/convolved_planet.fits', clobber=True)

    times['Computed signal and noise arrays'] = time()

    log.info(
        'Cube conversion: %.3f' %
        (times['Cube conversion'] - times['Start']))
    log.info('Process cubes through IFS: %.3f' %
             (times['Process cubes through IFS'] - times['Cube conversion']))
    log.info(
        'Process off-axis PSF through IFS: %.3f' %
        (times['Process off-axis PSF through IFS'] -
         times['Process cubes through IFS']))
    log.info(
        'Construct IFS detector: %.3f' %
        (times['Construct IFS detector'] -
         times['Process off-axis PSF through IFS']))
    log.info(
        'Taking averages: %.3f' %
        (times['Taking averages'] -
         times['Construct IFS detector']))
    log.info(
        'Process average cubes: %.3f' %
        (times['Process average cubes'] -
         times['Taking averages']))
    log.info('RDI: %.3f' % (times['RDI'] - times['Process average cubes']))
    log.info('Convolve: %.3f' % (times['Convolve'] - times['RDI']))
    log.info('Computed signal and noise arrays: %.3f' %
             (times['Computed signal and noise arrays'] - times['Convolve']))
    log.info(
        'Total time: %.3f' %
        (times['Computed signal and noise arrays'] - times['Start']))

    return signal, noise, noise_no_source, noise_no_rdi, signal_planet, signal_star, signal_no_rdi, signal_no_source


def SNR_spectrum(lam_midpts, signal, noise,
                 filename='Jupiter_1x_5AU_90deg.dat',
                 planet_radius=1.27,
                 planet_AU=3.6,
                 albedo=0.28,
                 lam_contrast=None, plot=True,
                 outname='SNR.png', outfolder='',
                 title='Planet+star',
                 edges=1,
                 FWHM=2,
                 FWHMdata=2.,
                 ymargin=0.05,  # in percent
                 ):
    '''
    Plot the outputs of process_SPC_IFS
    '''
    if lam_contrast is not None:
        lams = lam_contrast
    else:
        lams = lam_midpts
    real_vals = calc_contrast(
        lams,
        distance=planet_AU,
        radius=planet_radius,
        filename=filename,
        albedo=albedo)
    smooth = ndimage.filters.gaussian_filter1d(
        real_vals, FWHM / 2.35, order=0, mode='nearest')
    smoothfunc = interp1d(lams, smooth)
    smoothdata = ndimage.filters.gaussian_filter1d(
        signal, FWHMdata / 2.35, order=0, mode='nearest')
    smoothdatafunc = interp1d(lam_midpts, smoothdata)

#     chisq = np.sum((signal[edges:-edges] - smoothfunc(lam_midpts[edges:-edges]))**2/(noise[edges:-edges])**2)

    if plot:
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(lams, real_vals, label='Original spectrum')
        ax.fill_between(
            lams,
            0.9 * real_vals,
            1.1 * real_vals,
            alpha=0.3,
            facecolor='Gray')
        ax.errorbar(
            lam_midpts,
            signal,
            yerr=noise,
            label='Recovered spectrum',
            fmt='o')
        ax.plot(
            lams,
            smooth,
            '-',
            label='Original spectrum convolved w/ R=50 spectrograph')
#         if ratio is not None:
#             ax.plot(newlam,smoothdatafunc(newlam)*ratio*np.mean(real_vals)/np.mean(signal[edges:-edges]),'-',label='Gaussian-smoothed data w/ FWHM=%.0f bins' % FWHMdata)
#         if ratio is None:
#             ax.plot(newlam,smoothdatafunc(newlam)*np.mean(real_vals)/np.mean(signal[edges:-edges]),'-',label='Gaussian-smoothed data w/ FWHM=%.0f bins' % FWHMdata)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Contrast')
        ax.set_ylim([(1.0 - ymargin) * min(real_vals[edges:-edges]),
                     (1.0 + ymargin) * max(real_vals[edges:-edges])])
        #ax.set_title(title+', chisq='+str(chisq/len(signal[edges:-edges])))
        plt.legend()
        fig.savefig(outfolder + outname)
    return smoothfunc(lam_midpts) / signal


def mf(cube, mask, threshold):
    '''
    Matched filter function

    Parameters
    ----------
    cube: 3D ndarray
        An IFS cube representing the offaxis PSF, from which to compute the matched filter
    mask: 2D ndarray
        This is typically the coronagraphic mask
    threshold: float
        Fraction of max below which we crop the normalized PSF

    Returns
    -------
    matched_filter: 3D ndarray
        Matched filter with the same dimensions as input cube

    '''
    matched_filter = np.zeros(cube.data.shape)

    for slicenum in range(cube.data.shape[0]):
        cube_norm = cube.data[slicenum] / np.nansum(cube.data[slicenum])
        msk = mask * (cube_norm > np.nanmax(cube_norm) * threshold)

        # zero out all pixels outside of the thresholded area
        cube_norm[~msk] = 0.0

        # normalize
        cube_norm /= np.nansum(cube_norm)

        # this is now the final matched filter coefficients
        matched_filter[slicenum, :, :] = cube_norm / \
            np.nansum(cube_norm**2) * np.amax(cube_norm)
    return matched_filter


def recenter_offaxis(offaxis_file, threshold, outname='centered_offaxis.fits'):
    '''
    Example: recenter_offaxis('/Users/mrizzo/IFS/OS5/offaxis/spc_offaxis_psf.fits',0.01,par.exportDir+'/centered_offaxis.fits')

    Parameters
    ----------
    offaxis_file: string
        Path to off-axis file from John Krist
    threshold: float
        Threshold under which we don't care about the values
    outname: string
        The path to the centered off axis cube

    Returns
    -------
    outkey: HDUList
        HDU list of the recentered cube
    '''
    offaxis = Image(offaxis_file)
#     log.info('Previous sum: %.3f'% np.sum(offaxis.data))
    offsetpx = offaxis.header['OFFSET'] / offaxis.header['PIXSIZE']
    centered_offaxis = Image(data=shiftCube(offaxis.data, dx=-offsetpx, dy=0))
    oldsum = np.nansum(offaxis.data)
    centered_offaxis.data[centered_offaxis.data <
                          threshold * np.amax(centered_offaxis.data)] = 0.0
    centered_offaxis.data *= oldsum / np.nansum(centered_offaxis.data)
#     log.info('New sum: %.3f'% np.nansum(centered_offaxis.data))
    outkey = fits.HDUList(
        fits.PrimaryHDU(
            centered_offaxis.data,
            offaxis.header))
    outkey.writeto(outname, clobber=True)
    return outkey


# def construct_mflib(par,offaxis_psf_filename,threshold,lamc,BW,outdir,
#                     mask=None,IWA=None,OWA=None,trim=30,outname = 'mflib.fits.gz',order=3):
#     '''
#     Construct a library of matched filters for all points within the bowtie mask
#     For now, this uses the reduced, ideal offaxis psf already in cube space.
#     We could also build a function that offsets the original file, before IFS transformation.
#
#     This particular function saves memory and time by only recording the relevant pixels
#
#     Parameters
#     ----------
#     par: Parameters instance
#         Crispy parameter instance
#     offaxis_psf_filename: string
#         Path to the off-axis cube from J. Krist
#     threshold: float
#         Value below which we zero out the matched filter pixels
#     lamc: float
#         Central wavelength in nm
#     BW: float
#         Spectral bandwidth
#     outdir: string
#         Export directory
#     mask: 2D ndarray
#         2D bowtie mask. If left to None, then the bowtie mask is calculated and IWA and OWA need to be specified
#     IWA: float
#         Inner working angle (in terms of lam/D
#     OWA: float
#         Outer working angle (in terms of lam/D
#     trim: integer
#         How much to trim the cubes on each side to save memory space (nominally 30)
#     outname: string
#         Name of the matched filter library file. Highly recommend to compress it to save space
#     order: int
#         Order at which we do the spline transformation to move offaxis PSF around.
#
#
#     '''
#
#     # Recenter the offaxis PSF file so that it is nominally in the center of the IFS
#     recenter_offaxis(offaxis_psf_filename,0.01,par.exportDir+'/centered_offaxis.fits')
#     centered_offaxis_file = Image(par.exportDir+'/centered_offaxis.fits')
#
#     # adjust the file header for correct IFS propagation
#     adjust_krist_header(centered_offaxis_file,lamc=lamc)
#     par.saveDetector=False
#     Nlam = centered_offaxis_file.data.shape[0]
#     lamlist = lamc*np.linspace(1.-BW/2.,1.+BW/2.,Nlam)
#
#     # propagate the offaxis PSF
#     detectorFrame = polychromeIFS(par,lamlist,centered_offaxis_file,QE=True)
#     Image(data = detectorFrame,header=par.hdr).write(outdir+"/offaxis_centered_detector.fits",clobber=True)
#
#     # reduce the offaxis PSF
#     offaxis_reduced = reduceIFSMap(par,outdir+"/offaxis_centered_detector.fits")
#     offaxis_reduced.write(outdir+"/offaxis_centered_detector_red_optext.fits")
#
#     psf = Image(outdir+"/offaxis_centered_detector_red_optext.fits")
#
#     # calculate mask if unspecified
#     if mask is None:
#         ydim,xdim = psf.data[0].shape
#         mask,junk = bowtie(psf.data[0],ydim//2,xdim//2,openingAngle=65,
#                     clocking=-par.philens*180./np.pi,IWApix=IWA*lamc/par.lenslet_wav/par.lenslet_sampling,OWApix=OWA*lamc/par.lenslet_wav/par.lenslet_sampling,
#                     export=None,twomasks=False)
#
#     x = np.arange(mask.shape[1])
#     y = np.arange(mask.shape[0])
#     x,y = np.meshgrid(x,y)
#
#     # only care about the coordinates of pixels within the mask
#     # pay attention to bookkeeping in the end
#     xlist= x[mask]
#     ylist= y[mask]
#
#     # trim the data to save space
#     psftrim = psf.data[:,trim:-trim,trim:-trim]
#     masktrim = mask[trim:-trim,trim:-trim]
#     mflib = np.zeros(list(xlist.shape)+list(psftrim.shape))
#
#     # Now loop on all the valid pixel coordinates
#     ic = mask.shape[0]//2 # i or x axis is horizontal
#     jc = mask.shape[1]//2 # j or y axis is vertical
#     for ii in range(len(xlist)):
#         i = xlist[ii]
#         j = ylist[ii]
#         # move the offaxis pixel to center it on that pixel
#         decentered = Image(data=ndimage.interpolation.shift(psftrim,
#                          [0.0,j-jc,i-ic],order=order))
#         # calculate the matched filter for that image with the desired threshold
#         mflib[ii] = mf(decentered,masktrim,threshold)
#
#     # save the matched filter library but also the corresponding coordinates, necessary for unpacking
#     outkey = fits.HDUList(fits.PrimaryHDU(mflib))
#     outkey.append(fits.PrimaryHDU(mask.astype(np.int)))
#     outkey.append(fits.PrimaryHDU(xlist.astype(np.int)))
#     outkey.append(fits.PrimaryHDU(ylist.astype(np.int)))
#     outkey.writeto(outdir+'/'+outname,clobber=True)


def construct_mflib(par, planet_cube, threshold, lamc, BW, outdir, mask,
                    trim=30, outname='mflib.fits.gz', order=3):
    '''
    Construct a library of matched filters for all points within the bowtie mask
    For now, this uses the reduced, ideal offaxis psf already in cube space.
    We could also build a function that offsets the original file, before IFS transformation.

    This particular function saves memory and time by only recording the relevant pixels

    Parameters
    ----------
    par: Parameters instance
        Crispy parameter instance
    planet_cube: string
        Path to the off-axis planet IFS cube
    threshold: float
        Value below which we zero out the matched filter pixels
    lamc: float
        Central wavelength in nm
    BW: float
        Spectral bandwidth
    outdir: string
        Export directory
    mask: 2D ndarray
        2D bowtie mask. If left to None, then the bowtie mask is calculated and IWA and OWA need to be specified
    IWA: float
        Inner working angle (in terms of lam/D
    OWA: float
        Outer working angle (in terms of lam/D
    trim: integer
        How much to trim the cubes on each side to save memory space (nominally 30)
    outname: string
        Name of the matched filter library file. Highly recommend to compress it to save space
    order: int
        Order at which we do the spline transformation to move offaxis PSF around.


    '''

    # load PSF
    psf = Image(planet_cube)

    # coordinates
    jc, ic = np.unravel_index(np.nanargmax(
        psf.data[psf.data.shape[0] // 2]), psf.data[psf.data.shape[0] // 2].shape)
    log.info("Coordinates of the planet in lenslets: %.2f, %.2f" % (ic, jc))

    x = np.arange(mask.shape[1])
    y = np.arange(mask.shape[0])
    x, y = np.meshgrid(x, y)

    # only care about the coordinates of pixels within the mask
    # pay attention to bookkeeping in the end
    xlist = x[mask]
    ylist = y[mask]

    # trim the data to save space
    psftrim = psf.data[:, trim:-trim, trim:-trim]
    masktrim = mask[trim:-trim, trim:-trim]
    mflib = np.zeros(list(xlist.shape) + list(psftrim.shape))

    # Now loop on all the valid pixel coordinates
    for ii in range(len(xlist)):
        i = xlist[ii]
        j = ylist[ii]
        # move the offaxis pixel to center it on that pixel
        decentered = Image(data=ndimage.interpolation.shift(
            psftrim, [0.0, j - jc, i - ic], order=order))
        # calculate the matched filter for that image with the desired
        # threshold
        mflib[ii] = mf(decentered, masktrim, threshold)

    # save the matched filter library but also the corresponding coordinates,
    # necessary for unpacking
    outkey = fits.HDUList(fits.PrimaryHDU(mflib))
    outkey.append(fits.PrimaryHDU(mask.astype(np.int)))
    outkey.append(fits.PrimaryHDU(xlist.astype(np.int)))
    outkey.append(fits.PrimaryHDU(ylist.astype(np.int)))
    outkey.writeto(outdir + '/' + outname, clobber=True)


def convolved_mf(incube, mflibname, trim=30,):
    '''
    Generates a pseudo-convolution of the image by the matched filter library

    Parameters
    ----------
    incube: 3D ndarray
        Cube to convolve. Make sure all NaNs have been converted to zeros
    mflibname: string
        Path to the Matched Filter library, that was pre-computed
    trim: integer
        How much to trim the cubes on each side to save memory space (nominally 30)


    Returns
    -------
    convolvedmap: 3D ndarray
        Input cube multiplied in each non-zero pixel by the matched filter corresponding to that pixel

    '''

    # load the matched filter library and all of its extensions
    mflibHDUlist = fits.open(mflibname)
    mflib = mflibHDUlist[0].data
    mask = mflibHDUlist[1].data
    xlist = mflibHDUlist[2].data
    ylist = mflibHDUlist[3].data

    convolvedmap = np.zeros(incube.shape)
    for i in range(len(xlist)):
        # this now allows us to match an entry of the library to the cube
        # coordinates
        ix = xlist[i]
        iy = ylist[i]
        # for each pixel, we do a 2D sum
        convolvedmap[:, iy, ix] = np.nansum(np.nansum(
            incube[:, trim:-trim, trim:-trim] * mflib[i], axis=2), axis=1)

    return convolvedmap


def processReferenceCubes(par, xshift=0.0, yshift=0.0, order=3,
                          outdir_time_series='OS5',
                          ref_input_list=[],
                          process_cubes=True,
                          ref_star_T=9377 * u.K, ref_star_Vmag=2.37,
                          lamc=660., BW=0.18,
                          tel_pupil_area=3.650265060424805 * u.m**2,
                          ):
    '''
    Processes raw cubes from John Krist into IFS flux cubes just before the IFS.
    Doesn't take into account the optical losses, but does take into account the detector QE.

    Parameters
    ----------
    par: Parameters instance
        Crispy parameter instance
    xshift: float
        The amount to shift the input in X (in final IFS cube pixels)
    yshift: float
        The amount to shift the input in Y (in final IFS cube pixels)
    order: integer
        The order of the spline transform used for shifting the cubes
    outdir_time_series: string
        Path to which we will export the IFS fluxes at the detector
    ref_input_list: string
        List of filenames with the reference cubes
    Nref: integer
        Out of the files in John Krist's folder, how many correspond to observations of the reference star
    ref_only: Boolean
        Whether to only process the reference files or also target files
    ref_star_T: 'u.K'
        Temperature of the reference star in u.K
    ref_star_Vmag: float
        V Magnitude of the reference star
    lamc: float
        Central wavelength of the band, in nm
    BW: float
        Bandwidth
    tel_pupil_area: 'u.m**2'
        Collecting area of the telescope, minus obscurations, in u.m**2

    Returns
    -------
    ref_outlist: list of strings
        List of the filenames corresponding the reference star
    target_outlist: list of strings
        List of the filenames corresponding the target star
    fileshape: tuple
        Size of the original John Krist cube


    '''

    ##########################################################################
    # Load, shift and propagate all of the IFS images for the reference star
    ##########################################################################

    # load the filenames

    # load first filelist to get its shape
    kristfile = Image(filename=ref_input_list[0])
    fileshape = kristfile.data.shape
    adjust_krist_header(kristfile, lamc=lamc)

    Nlam = fileshape[0]
    lamlist = lamc * np.linspace(1. - BW / 2., 1. + BW / 2., Nlam) * u.nm

    # reference and target star cube conversions
    ref_star_cube = convert_krist_cube(
        fileshape,
        lamlist,
        ref_star_T,
        ref_star_Vmag,
        tel_pupil_area)

    # compute the amount to be shifted in the cubes from J. Krist
    input_sampling = kristfile.header['PIXSIZE']
    input_wav = kristfile.header['LAM_C'] * 1000.
    par.pixperlenslet = par.lenslet_sampling / \
        (input_sampling * input_wav / par.lenslet_wav)
    log.info('X,Y Shift in px in original cubes: %.2f, %.2f' %
             (xshift * par.pixperlenslet, yshift * par.pixperlenslet))

    ##########################################################################
    # simulate the IFS flux at the detector plane (no losses other than QE)
    ##########################################################################
    ref_outlist = []
    for i in range(len(ref_input_list)):
        reffile = ref_input_list[i]
        if process_cubes:
            log.info('Processing file ' + reffile.split('/')[-1])
            cube = fits.open(reffile)[0]
            cube.data *= ref_star_cube

            # adjust headers for slightly different wavelength
            adjust_krist_header(cube, lamc=lamc)
            par.saveDetector = False

            # shift the cube
            log.info("Shifting input cube")
            cube.data = ndimage.interpolation.shift(
                cube.data, [
                    0.0, yshift * par.pixperlenslet, xshift * par.pixperlenslet], order=order)
            detectorFrame = polychromeIFS(par, lamlist.value, cube, QE=True)

            par.hdr.append(
                ('XSHIFT',
                 xshift * par.pixperlenslet,
                 'X Shift in px in original cubes'),
                end=True)
            par.hdr.append(
                ('YSHIFT',
                 yshift * par.pixperlenslet,
                 'Y Shift in px in original cubes'),
                end=True)

            Image(data=detectorFrame, header=par.hdr).write(outdir_time_series + '/' + \
                  reffile.split('/')[-1].split('.')[0] + '_refstar_IFS.fits', clobber=True)
            ref_outlist.append(outdir_time_series + '/' + reffile.split('/')
                               [-1].split('.')[0] + '_refstar_IFS.fits')

        else:
            ref_outlist.append(outdir_time_series + '/' + reffile.split('/')
                               [-1].split('.')[0] + '_refstar_IFS.fits')

    return ref_outlist


def processTargetCubes(par, target_file_list,
                       outdir_time_series='OS5',
                       process_cubes=True,
                       target_star_T=5887 * u.K, target_star_Vmag=5.03,
                       lamc=660., BW=0.18,
                       tel_pupil_area=3.650265060424805 * u.m**2,
                       ):
    '''
    Processes raw cubes from John Krist into IFS flux cubes just before the IFS.
    Doesn't take into account the optical losses, but does take into account the detector QE.

    Parameters
    ----------
    par: Parameters instance
        Crispy parameter instance
    target_file_list: string
        List of files
    outdir_time_series: string
        Path to which we will export the IFS fluxes at the detector
    target_star_T: 'u.K'
        Temperature of the target star in u.K
    target_star_Vmag: float
        V Magnitude of the target star
    lamc: float
        Central wavelength of the band, in nm
    BW: float
        Bandwidth
    tel_pupil_area: 'u.m**2'
        Collecting area of the telescope, minus obscurations, in u.m**2

    Returns
    -------
    target_outlist: list of strings
        List of the filenames corresponding the target star


    '''

    ##########################################################################
    # Load, shift and propagate all of the IFS images for the reference star
    ##########################################################################

    # load first filelist to get its shape
    kristfile = Image(filename=target_file_list[0])
    fileshape = kristfile.data.shape
    adjust_krist_header(kristfile, lamc=lamc)

    Nlam = fileshape[0]
    lamlist = lamc * np.linspace(1. - BW / 2., 1. + BW / 2., Nlam) * u.nm

    # reference and target star cube conversions
    target_star_cube = convert_krist_cube(
        fileshape,
        lamlist,
        target_star_T,
        target_star_Vmag,
        tel_pupil_area)

    ##########################################################################
    # simulate the IFS flux at the detector plane (no losses other than QE)
    ##########################################################################
    target_outlist = []
    for i in range(len(target_file_list)):
        reffile = target_file_list[i]
        if process_cubes:
            log.info('Processing file ' + reffile.split('/')[-1])
            cube = fits.open(reffile)[0]
            cube.data *= target_star_cube

            # adjust headers for slightly different wavelength
            adjust_krist_header(cube, lamc=lamc)
            par.saveDetector = False

            detectorFrame = polychromeIFS(par, lamlist.value, cube, QE=True)

            Image(data=detectorFrame, header=par.hdr).write(outdir_time_series + '/' + \
                  reffile.split('/')[-1].split('.')[0] + '_targetstar_IFS.fits', clobber=True)
            target_outlist.append(outdir_time_series + '/' + reffile.split('/')
                                  [-1].split('.')[0] + '_targetstar_IFS.fits')

        else:
            target_outlist.append(outdir_time_series + '/' + reffile.split('/')
                                  [-1].split('.')[0] + '_targetstar_IFS.fits')

    return target_outlist


# def process_offaxis(par,offaxis_psf_filename,fileshape,lamlist,lamc,outdir_average,Nave=1,inttime=1,Nreads=1,
#                     target_star_T=5887*u.K, target_star_Vmag=5.03,tel_pupil_area=3.650265060424805*u.m**2):
#
#     '''
#     Processes an off-axis PSF cube from John Krist. This is used to normalized the cubes to contrast units,
#     so the single image needs to be read by the detector model and averaged the same number of times that
#     the target image is being averaged.
#
#     Parameters
#     ----------
#     par: Parameters instance
#         Crispy parameter instance
#     offaxis_psf_filename: string
#         Path to the off-axis cube from J. Krist
#     fileshape: tuple
#         Size of the original John Krist cube
#     lamlist: list
#         List of wavelengths corresponding to each slice
#     lamc: float
#         Central wavelength
#     outdir_average: string
#         Path to which we will export the off-axis IFS flux maps at the detector
#     Nave: integer
#         Number of files to average. This needs to be the same number as the number of times the target is averaged (nominally 100)
#     target_star_T: 'u.K'
#         Temperature of the target star in u.K
#     target_star_Vmag: float
#         V Magnitude of the target star
#     tel_pupil_area: 'u.m**2'
#         Collecting area of the telescope, minus obscurations, in u.m**2
#
#     Returns
#     -------
#     offaxis_reduced: 3D ndarray
#         Reduced cube
#
#
#     '''
#     offaxiscube = Image(offaxis_psf_filename)
#     print('Processing file '+offaxis_psf_filename)
#
#
#     # Need to re-center the off-axis psf if it is not the right size
#     if offaxiscube.data.shape[1] < fileshape[1]:
#         diff = fileshape[1]-offaxiscube.data.shape[1]
#         offaxiscube_recentered = np.zeros(fileshape)
#         offaxiscube_recentered[:,diff//2:-diff//2,diff//2:-diff//2] += offaxiscube.data
#         offaxiscube = Image(data=offaxiscube_recentered,header = offaxiscube.header)
#     offaxis_star_cube = convert_krist_cube(offaxiscube.data.shape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)
#     offaxiscube.data*=offaxis_star_cube
#
#     # adjust headers for different wavelength
#     adjust_krist_header(offaxiscube,lamc=lamc)
#     par.saveDetector=False
#     Image(data=offaxiscube.data).write(outdir_average+'/offaxiscube_star_processed.fits',clobber=True)
#     detectorFrame = polychromeIFS(par,lamlist.value,offaxiscube,QE=True)
#     det = Image(data = detectorFrame,header=par.hdr)
#     det.write(outdir_average+'/offaxis_star.fits',clobber=True)
#
#     det_out = np.zeros(detectorFrame.shape)
#     # temporary use no noise
#     old_noise_param = par.nonoise
#     par.nonoise=True
#     for i in range(Nave*Nreads):
#         det_out+=readDetector(par,det,inttime=inttime)
#
#     Image(data = det_out,header=par.hdr).write(outdir_average+'/offaxis_detector.fits',clobber=True)
#
#     # revert back to original noise parameter
#     par.nonoise=old_noise_param
#
#     offaxis_reduced = reduceIFSMap(par,outdir_average+'/offaxis_detector.fits')
#     Image(data=offaxis_reduced.data).write(outdir_average+"/offaxis_detector_red_optext.fits")
#
#     return offaxis_reduced

def process_offaxis(par, offaxis_psf_filename, fileshape, lamlist, lamc, outdir_average, Nave=1, inttime=1, Nreads=1,
                    filename='Jupiter_1x_5AU_90deg.dat',
                    planet_radius=1.27,  # in R_jup
                    planet_AU=3.6, planet_dist_pc=14.1,
                    albedo=0.28,
                    target_star_T=5887 * u.K, target_star_Vmag=5.03,
                    ref_star_T=9377 * u.K, ref_star_Vmag=2.37,
                    tel_pupil_area=3.650265060424805 * u.m**2, order=3,
                    useQE=True, polychromeOut=True):
    '''
    Processes an off-axis PSF cube from John Krist. This is used to to construct an ideal cube of the planet

    Parameters
    ----------
    par: Parameters instance
        Crispy parameter instance
    offaxis_psf_filename: string
        Path to the off-axis cube from J. Krist
    fileshape: tuple
        Size of the original John Krist cube
    lamlist: list
        List of wavelengths corresponding to each slice
    lamc: float
        Central wavelength
    outdir_average: string
        Path to which we will export the off-axis IFS flux maps at the detector
    Nave: integer
        Number of files to average. This needs to be the same number as the number of times the target is averaged (nominally 100)
    target_star_T: 'u.K'
        Temperature of the target star in u.K
    target_star_Vmag: float
        V Magnitude of the target star
    tel_pupil_area: 'u.m**2'
        Collecting area of the telescope, minus obscurations, in u.m**2
    polychromeOut: Boolean
        Whether or not to export a polychrome cube weighted by the PSF

    Returns
    -------
    offaxis_reduced: 3D ndarray
        Reduced cube


    '''
    kristfile = fits.open(offaxis_psf_filename)[0]
    adjust_krist_header(kristfile, lamc=lamc)
    log.info('Recentering off-axis cube')
    offaxiscube = recenter_offaxis(
        offaxis_psf_filename,
        0.01,
        outname=outdir_average +
        '/centered_offaxis.fits')[0]

    # compute the amount to be shifted in the cubes from J. Krist
    input_sampling = kristfile.header['PIXSIZE']
    input_wav = kristfile.header['LAM_C'] * 1000.
    par.pixperlenslet = par.lenslet_sampling / \
        (input_sampling * input_wav / par.lenslet_wav)
    log.info(
        'The number of input pixels per lenslet is %f' %
        par.pixperlenslet)

    # Need to re-center the off-axis psf if it is not the right size
    if offaxiscube.data.shape[1] < fileshape[1]:
        diff = fileshape[1] - offaxiscube.data.shape[1]
        offaxiscube_recentered = np.zeros(fileshape)
        offaxiscube_recentered[:, diff // 2:-diff // 2,
                               diff // 2:-diff // 2] += offaxiscube.data
        offaxiscube = Image(
            data=offaxiscube_recentered,
            header=offaxiscube.header)
    Image(
        data=offaxiscube.data).write(
        outdir_average +
        '/offaxiscube.fits',
        clobber=True)

    # shift the planet to the correct position
    planet_WA = planet_AU / planet_dist_pc / (lamc * 1e-9 / 2.37 / 4.848e-6)
    log.info(
        'Constructing off-axis cube at planet separation: %.2f lam/D (%.2f arcsec, %.2f lenslets)' %
        (planet_WA,
         planet_AU /
         planet_dist_pc,
         planet_WA *
         lamc /
         par.lenslet_wav /
         par.lenslet_sampling))

    # ensure correct normalization after shifting, for extra safety
    oldsum = np.nansum(offaxiscube.data)
    offaxiscube.data = ndimage.interpolation.shift(
        offaxiscube.data,
        [
            0.0,
            0.0,
            planet_WA *
            par.pixperlenslet *
            lamc /
            par.lenslet_wav /
            par.lenslet_sampling],
        order=order)
    offaxiscube.data *= oldsum / np.nansum(offaxiscube.data)
    Image(
        data=offaxiscube.data).write(
        outdir_average +
        '/offaxiscube_shifted.fits',
        clobber=True)

    ##########################################################################
    # Propagate the target star
    ##########################################################################

    # this takes care of the photometry
    offaxis_star_cube = convert_krist_cube(
        offaxiscube.data.shape,
        lamlist,
        target_star_T,
        target_star_Vmag,
        tel_pupil_area)

    offtarget = Image(
        data=offaxiscube.data *
        offaxis_star_cube,
        header=offaxiscube.header)
    # adjust headers for different wavelength
    adjust_krist_header(offtarget, lamc=lamc)

#     log.info('{:}'.format(offtarget.data[1,1,1]))
    out = fits.HDUList(fits.PrimaryHDU(None, offtarget.header))
    out.append(fits.PrimaryHDU((offtarget.data.value).astype(np.float32)))

    out.writeto(
        outdir_average +
        '/offaxiscube_star_processed.fits',
        clobber=True)
    detectorFrame = polychromeIFS(par, lamlist.value, offtarget, QE=useQE)
    det = Image(data=detectorFrame, header=par.hdr)
    det.write(outdir_average + '/offaxis_star.fits', clobber=True)

    det_out = np.zeros(detectorFrame.shape)
    # temporary use no noise
    old_noise_param = par.nonoise
    par.nonoise = True
    for i in range(Nave * Nreads):
        det_out += readDetector(par, det, inttime=inttime)

    Image(
        data=det_out,
        header=par.hdr).write(
        outdir_average +
        '/offaxis_star_detector.fits',
        clobber=True)

    # revert back to original noise parameter
    par.nonoise = old_noise_param

    offaxis_reduced = reduceIFSMap(
        par, outdir_average + '/offaxis_star_detector.fits')
    offaxis_reduced.write(outdir_average + "/offaxis_star_red_optext.fits")

    ##########################################################################
    # Propagate the ref star
    ##########################################################################

    # this takes care of the photometry
    offaxis_ref_star_cube = convert_krist_cube(
        offaxiscube.data.shape,
        lamlist,
        ref_star_T,
        ref_star_Vmag,
        tel_pupil_area)

    offref = Image(
        data=offaxiscube.data *
        offaxis_ref_star_cube,
        header=offaxiscube.header)
    # adjust headers for different wavelength
    adjust_krist_header(offref, lamc=lamc)

    out = fits.HDUList(fits.PrimaryHDU(None, offref.header))
    out.append(fits.PrimaryHDU((offref.data.value).astype(np.float32)))
    out.writeto(
        outdir_average +
        '/offaxiscube_ref_star_processed.fits',
        clobber=True)
    detectorFrame = polychromeIFS(par, lamlist.value, offref, QE=useQE)
    det = Image(data=detectorFrame, header=par.hdr)
    det.write(outdir_average + '/offaxis_ref_star.fits', clobber=True)

    det_out = np.zeros(detectorFrame.shape)
    # temporary use no noise
    old_noise_param = par.nonoise
    par.nonoise = True
    for i in range(Nave * Nreads):
        det_out += readDetector(par, det, inttime=inttime)

    Image(
        data=det_out,
        header=par.hdr).write(
        outdir_average +
        '/offaxis_ref_star_detector.fits',
        clobber=True)

    # revert back to original noise parameter
    par.nonoise = old_noise_param

    offaxis_reduced = reduceIFSMap(
        par, outdir_average + '/offaxis_ref_star_detector.fits')
    offaxis_reduced.write(outdir_average + "/offaxis_ref_star_red_optext.fits")

    ##########################################################################
    # Propagate the planet
    ##########################################################################
    contrast = calc_contrast(
        lamlist.value,
        distance=planet_AU,
        radius=planet_radius,
        filename=filename,
        albedo=albedo)
    offaxiscube.data = offaxiscube.data * \
        offaxis_star_cube.value * contrast[:, np.newaxis, np.newaxis]

    # adjust headers for slightly different wavelength
    par.saveDetector = False
    out = fits.HDUList(fits.PrimaryHDU(None, offaxiscube.header))
    out.append(fits.PrimaryHDU((offaxiscube.data.value).astype(np.float32)))
    out.writeto(
        outdir_average +
        '/offaxiscube_planet_processed.fits',
        clobber=True)

    detectorFrame = polychromeIFS(par, lamlist.value, offaxiscube, QE=useQE)
    par.hdr.append(
        ('XSHIFT',
         planet_WA *
         par.pixperlenslet *
         lamc /
         par.lenslet_wav /
         par.lenslet_sampling,
         'X Shift in px in offaxis cubes'),
        end=True)
    det = Image(data=detectorFrame, header=par.hdr)
    det.write(outdir_average + '/offaxis_planet.fits', clobber=True)

    det_out = np.zeros(detectorFrame.shape)
    # temporarily use no noise
    old_noise_param = par.nonoise
    par.nonoise = True
    for i in range(Nave * Nreads):
        det_out += readDetector(par, det, inttime=inttime)

    Image(
        data=det_out,
        header=par.hdr).write(
        outdir_average +
        '/offaxis_planet_detector.fits',
        clobber=True)
    # revert back to original noise parameter
    par.nonoise = old_noise_param

    # reduce the IFS map
    reduced = reduceIFSMap(
        par,
        outdir_average +
        '/offaxis_planet_detector.fits')
    reduced.write(
        outdir_average +
        '/offaxis_planet_red_optext.fits',
        clobber=True)
    PSF_cube = fits.open(par.exportDir + '/detectorFramePoly.fits')[1].data
    out = fits.HDUList(fits.PrimaryHDU(None, offaxiscube.header))
    out.append(fits.PrimaryHDU(PSF_cube))
    out.writeto(outdir_average + '/PSF_cube.fits', clobber=True)

    ##########################################################################
    # Construct a polychrome cube with weights corresponding to the off-axis PSF for
    # maximum-likelihood extraction
    ##########################################################################
    if polychromeOut:
        # interpolate original list of wavelengths
        interpol = interp1d(lamlist.value, range(len(lamlist)))
        lam_midpts, _ = calculateWaveList(par, method='lstsq')
        # find closest index
        indices = np.round(interpol(lam_midpts)).astype(int)
        # now offaxiscube[indices] corresponds to the slices with the
        # coefficients for the polychrome
        newcube = Image(
            data=offaxiscube.data[indices],
            header=offaxiscube.header)
        # construct polychrome (make sure par.savePoly=True)
        detector = polychromeIFS(par, lam_midpts, newcube, QE=useQE)
        PSF_polychrome = fits.open(
            par.exportDir +
            '/detectorFramePoly.fits')[1].data
        out = fits.HDUList(fits.PrimaryHDU(None, offaxiscube.header))
        out.append(fits.PrimaryHDU(PSF_polychrome))
        out.writeto(outdir_average + '/PSF_polychrome.fits', clobber=True)

    return reduced


def RDI_noise(
        par,
        xshift,
        yshift,
        order=3,
        rootname="mflib",
        outdir_time_series='OS5',
        outdir_detector='OS5/OS5_detector',
        outdir_average='OS5/OS5_average',
        process_cubes=True,
        process_offaxis_files=True,
        process_detector=True,
        take_averages=True,
        countershift=True,
        normalize_cubes=True,
        forced_inttime_ref=50,
        offaxis_reduced="OS5/OS5_average/offaxis.fits",
        psf_time_series_folder='/Users/mrizzo/IFS/OS5/with_lowfc',
        Nref=30,
        ref_only=True,
        offaxis_psf_filename='/Users/mrizzo/IFS/OS5/offaxis/spc_offaxis_psf.fits',
        ref_star_T=9377 * u.K,
        ref_star_Vmag=2.37,
        target_star_T=5887 * u.K,
        target_star_Vmag=5.03,
        IWA=3,
        OWA=9,
        lamc=660.,
        BW=0.18,
        tel_pupil_area=3.650265060424805 * u.m**2,
        mflib='/mflib.fits.gz'):
    '''
    Determines the SNR map of a simulation using RDI.
    Here we assume the target PSF time series is already in hand, and already in outdir_time_series.

    Parameters
    ----------
    par: Parameters instance
        Crispy parameter instance
    xshift: float
        The amount to shift the input in X (in final IFS cube pixels)
    yshift: float
        The amount to shift the input in Y (in final IFS cube pixels)
    order: integer
        The order of the spline transform used for shifting the cubes
    rootname: string
        The root name of the final SNR cube maps
    outdir_time_series: string
        Path to which we will export the IFS fluxes at the detector
    outdir_detector: string
        Path to the folder where we will put the detector maps
    outdir_average: string
        Path to the folder where we will put the final results and other byproducts of the function
    process_cubes: Boolean
        Whether or not to process the IFS cubes, or assume that they are already processed
    countershift: Boolean
        Whether to countershift the reference star or not
    normalize_contrast: Boolean
        Whether to normalize the SNR map to contrast units using off-axis PSF
    offaxis_reduced: string
        Path to the offaxis PSF that will be used for contrast normalization
    psf_time_series_folder: string
        Path to the folder where all the original files are located
    Nref: integer
        Out of the files in John Krist's folder, how many correspond to observations of the reference star
    ref_only: Boolean
        Whether to only process the reference files or also target files
    ref_star_T: 'u.K'
        Temperature of the reference star in u.K
    target_star_T: 'u.K'
        Temperature of the target star in u.K
    ref_star_Vmag: float
        V Magnitude of the reference star
    target_star_Vmag: float
        V Magnitude of the target star
    lamc: float
        Central wavelength of the band, in nm
    BW: float
        Bandwidth
    tel_pupil_area: 'u.m**2'
        Collecting area of the telescope, minus obscurations, in u.m**2
    mflib: string
        Name of the matched filter library file



    Returns
    -------
    pixstd: array
        List of slice-by-slice standard deviation of the RDI cube convolved with the matched filter
    '''

    # load the filenames
    filelist = sorted(glob.glob(psf_time_series_folder + '/*.fits'))
    fileshape = fits.getdata(filelist[0]).shape
    lamlist = lamc * np.linspace(1. - BW / 2.,
                                 1. + BW / 2., fileshape[0]) * u.nm
    reffiles = filelist[:Nref]
    targetfiles = filelist[Nref:]

    c = np.cos(par.philens)
    s = np.sin(par.philens)

    ref_outlist = processReferenceCubes(
        par,
        xshift * c - yshift * s,
        yshift * c + xshift * s,
        order=order,
        outdir_time_series=outdir_time_series,
        ref_input_list=reffiles,
        process_cubes=process_cubes,
        ref_star_T=ref_star_T,
        ref_star_Vmag=ref_star_Vmag,
        lamc=lamc,
        BW=BW,
        tel_pupil_area=tel_pupil_area)
    if ref_only:
        target_outlist = processTargetCubes(
            par,
            targetfiles,
            outdir_time_series=outdir_time_series,
            process_cubes=False,
            target_star_T=target_star_T,
            target_star_Vmag=target_star_Vmag,
            lamc=lamc,
            BW=BW,
            tel_pupil_area=tel_pupil_area)
    else:
        target_outlist = processTargetCubes(
            par,
            targetfiles,
            outdir_time_series=outdir_time_series,
            process_cubes=process_cubes,
            target_star_T=target_star_T,
            target_star_Vmag=target_star_Vmag,
            lamc=lamc,
            BW=BW,
            tel_pupil_area=tel_pupil_area)

    if process_offaxis_files:
        process_offaxis(
            par,
            offaxis_psf_filename=offaxis_psf_filename,
            fileshape=fileshape,
            lamlist=lamlist,
            lamc=lamc,
            inttime=1,
            Nreads=1,
            Nave=1,
            outdir_average=outdir_average,
            target_star_T=target_star_T,
            target_star_Vmag=target_star_Vmag,
            tel_pupil_area=tel_pupil_area)

    ##########################################################################
    # Add the off-axis PSF before reading on the detector
    ##########################################################################

    if process_detector:
        log.info("Taking average of reference star")
        ref_det_outlist = averageDetectorReadout(
            par, ref_outlist, outdir_detector, forced_inttime=forced_inttime_ref)
        #zodi_filename = os.path.abspath(outdir_average+'/zodicube.fits')
        log.info("Taking average of target star without planet")
        target_nosource_outlist = averageDetectorReadout(
            par, target_outlist, outdir_detector, suffix='nosource_detector')
        log.info("Taking average of target star with planet")
        target_det_outlist = averageDetectorReadout(
            par,
            target_outlist,
            outdir_detector,
            offaxis=outdir_average +
            '/offaxis_planet.fits')
    else:
        ref_det_outlist = []
        target_nosource_outlist = []
        target_det_outlist = []
        suffix = 'detector'
        for reffile in ref_outlist:
            ref_det_outlist.append(
                outdir_detector + '/' + reffile.split('/')[-1].split('.')[0] + '_' + suffix + '.fits')
        for reffile in target_outlist:
            target_nosource_outlist.append(
                outdir_detector + '/' + reffile.split('/')[-1].split('.')[0] + '_nosource_detector.fits')
            target_det_outlist.append(
                outdir_detector + '/' + reffile.split('/')[-1].split('.')[0] + '_' + suffix + '.fits')

    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(
        ('comment',
         '*' *
         22 +
         ' Postprocessing ' +
         '*' *
         20),
        end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(('comment', ''), end=True)

    ##########################################################################
    # Take averages
    ##########################################################################

    if take_averages:
        log.info('Taking averages')
        ref_star_average = np.zeros(
            Image(filename=ref_det_outlist[0]).data.shape)
        target_star_average = np.zeros(
            Image(filename=target_det_outlist[0]).data.shape)
        target_star_nosource_average = np.zeros(
            Image(filename=target_nosource_outlist[0]).data.shape)
        for reffile in ref_det_outlist:
            ref_star_average += Image(filename=reffile).data
        ref_star_average /= par.timeframe * len(ref_det_outlist)
        for reffile in target_det_outlist:
            target_star_average += Image(filename=reffile).data
        target_star_average /= par.timeframe * len(target_det_outlist)
        for reffile in target_nosource_outlist:
            target_star_nosource_average += Image(filename=reffile).data
        target_star_nosource_average /= par.timeframe * \
            len(target_nosource_outlist)
        ref = Image(data=ref_star_average)
        ref.write(
            outdir_average +
            "/ref_average_detector_" +
            rootname +
            ".fits",
            clobber=True)
        target = Image(data=target_star_average, header=par.hdr)
        target.write(
            outdir_average +
            '/target_average_detector.fits',
            clobber=True)
        target_nosource = Image(
            data=target_star_nosource_average,
            header=par.hdr)
        target.write(
            outdir_average +
            '/target_star_nosource_average.fits',
            clobber=True)
    else:
        ref_star_average = Image(
            outdir_average +
            '/average_ref_star_detector.fits').data
        target_star_average = Image(
            outdir_average +
            '/average_target_star_detector.fits').data
        target_star_nosource_average = Image(
            outdir_average + '/average_target_star_nosource_detector.fits').data

    if "NREADS" not in par.hdr:
        par.hdr.append(
            ('NREADS',
             par.Nreads,
             'Number of subframes co-added per image'),
            end=True)
        par.hdr.append(
            ('EXPTIME',
             par.timeframe,
             'Total exposure time for number of frames'),
            end=True)
    par.hdr.append(
        ('NIMGS',
         len(target_det_outlist),
         'Number of time series steps'),
        end=True)
    par.hdr.append(
        ('TOTTIME',
         len(target_det_outlist) *
         par.timeframe,
         'Total integration time on source'),
        end=True)

    ##########################################################################
    # Reduce IFS images
    ##########################################################################

    ref_reduced = reduceIFSMap(
        par,
        outdir_average +
        "/ref_average_detector_" +
        rootname +
        ".fits")
    ref_reduced.write(
        outdir_average +
        "/ref_average_detector_" +
        rootname +
        "_red_optext.fits",
        clobber=True)
    target_reduced = reduceIFSMap(
        par, outdir_average + '/target_average_detector.fits')
    target_reduced.write(
        outdir_average +
        "/target_average_detector_red_optext.fits",
        clobber=True)
    target_nosource_reduced = reduceIFSMap(
        par, outdir_average + '/target_star_nosource_average.fits')
    target_nosource_reduced.write(
        outdir_average +
        "/target_star_nosource_average_red_optext.fits",
        clobber=True)

    if normalize_cubes:
        log.info("Normalizing cubes...")
        offaxis_star_cube = Image(
            outdir_average +
            "/offaxis_star_red_optext.fits")
        lam_midpts, waveList = calculateWaveList(par, method='optext')
        dlam = (waveList[-1] - waveList[0]) / (len(waveList) - 1)
#         ratio = 1./np.nansum(np.nansum(offaxis_star_cube.data,axis=2),axis=1)
        ratio = 1. / \
            np.nanmax(np.nanmax(offaxis_star_cube.data, axis=2), axis=1)
        ref_reduced.data *= ratio[:, np.newaxis, np.newaxis]
        target_reduced.data *= ratio[:, np.newaxis, np.newaxis]
        target_nosource_reduced.data *= ratio[:, np.newaxis, np.newaxis]
        par.hdr.append(
            ('comment', 'Normalized cubes to photons per sec per nm'), end=True)

    ref_reduced.write(
        outdir_average +
        "/ref_average_detector_red_optext_normalized.fits",
        clobber=True)
    target_reduced.write(
        outdir_average +
        "/target_average_detector_red_optext_normalized.fits",
        clobber=True)
    target_nosource_reduced.write(
        outdir_average +
        "/target_star_nosource_average_red_optext_normalized.fits",
        clobber=True)
    # ref_reduced is now the IFS cube from the shifted reference star
    # target_reduced is the IFS cube from the target star

    ##########################################################################
    # Counter-shift the reference cube
    ##########################################################################
    if countershift:
        ref_reduced.data[np.isnan(ref_reduced.data)] = 0.0
        ref_reduced.data = ndimage.interpolation.shift(
            ref_reduced.data, [0.0, -yshift, -xshift], order=order)
        ref_reduced.write(
            outdir_average +
            "/ref_average_detector_countershifted_" +
            rootname +
            "_red_optext.fits",
            clobber=True)

    ##########################################################################
    # Do basic least squares RDI, slice by slice; no mean subtraction for now
    ##########################################################################
    ydim, xdim = target_reduced.data[0].shape
    mask, scratch = bowtie(target_reduced.data[0], ydim //
                           2 -
                           1, xdim //
                           2, openingAngle=60, clocking=-
                           par.philens *
                           180. /
                           np.pi, IWApix=IWA *
                           lamc /
                           par.lenslet_wav /
                           par.lenslet_sampling, OWApix=OWA *
                           lamc /
                           par.lenslet_wav /
                           par.lenslet_sampling, export=outdir_average +
                           '/bowtie', twomasks=False)
    maskleft, maskright = bowtie(target_reduced.data[0], ydim //
                                 2 -
                                 1, xdim //
                                 2, openingAngle=60, clocking=-
                                 par.philens *
                                 180. /
                                 np.pi, IWApix=IWA *
                                 lamc /
                                 par.lenslet_wav /
                                 par.lenslet_sampling, OWApix=OWA *
                                 lamc /
                                 par.lenslet_wav /
                                 par.lenslet_sampling, export=outdir_average +
                                 '/bowtie', twomasks=True)

    ##########################################################################
    # Convolve with matched filter the files before post-processing
    ##########################################################################
    if mflib == '':
        # re-build the matched filter library
        construct_mflib(
            par,
            planet_cube=outdir_average +
            '/offaxis_planet_red_optext.fits',
            threshold=0.2,
            lamc=lamc,
            BW=BW,
            outdir=outdir_average,
            mask=mask,
            trim=30,
            outname='mflib.fits.gz',
            order=3)
        mflib = outdir_average + '/mflib.fits.gz'

    convolvedwoRDItarget = convolved_mf(target_reduced.data, mflib)
    Image(data=convolvedwoRDItarget, header=par.hdr).write(
        outdir_average + '/convolved_without_RDI_target.fits')
    convolvedwoRDIreference = convolved_mf(ref_reduced.data, mflib)
    Image(data=convolvedwoRDIreference, header=par.hdr).write(
        outdir_average + '/convolved_without_RDI_reference.fits')
    convolvedwoRDItarget_nosource = convolved_mf(
        target_nosource_reduced.data, mflib)
    Image(data=convolvedwoRDItarget_nosource, header=par.hdr).write(
        outdir_average + '/convolved_without_RDI_target_nosource.fits')

    ##########################################################################
    # RDI
    ##########################################################################

    coefs, residual = scale2imgs(
        target_reduced, ref_reduced, maskleft, returndiff=True)
    coefs, residual_nosource = scale2imgs(
        target_reduced, ref_reduced, maskleft, returndiff=True)
    par.hdr.append(('comment', 'Applied RDI'), end=True)

    Image(
        data=residual,
        header=par.hdr).write(
        outdir_average +
        "/lstsq_residual_" +
        rootname +
        ".fits")
    Image(
        data=residual_nosource,
        header=par.hdr).write(
        outdir_average +
        "/lstsq_residual_nosource_" +
        rootname +
        ".fits")

    log.info("Convolving with matched filter")
    convolvedwRDI = convolved_mf(residual, mflib)
    Image(
        data=convolvedwRDI,
        header=par.hdr).write(
        outdir_average +
        '/convolved_with_RDI.fits')
    convolvedwRDI_nosource = convolved_mf(residual_nosource, mflib)
    Image(data=convolvedwRDI_nosource, header=par.hdr).write(
        outdir_average + '/convolved_with_RDI_nosource.fits')

    ##########################################################################
    # Outputs
    ##########################################################################

    outkey = fits.HDUList(fits.PrimaryHDU(convolvedwRDI.astype(np.float)))
    outkey.append(fits.PrimaryHDU(convolvedwRDI_nosource.astype(np.float)))
    outkey.append(fits.PrimaryHDU(convolvedwoRDItarget.astype(np.float)))
    outkey.writeto(
        outdir_average +
        '/mflib' +
        rootname +
        '.fits',
        clobber=True)
    pixstd = np.array([np.nanstd(convolvedwoRDItarget[i] * maskleft)
                       for i in range(convolvedwoRDItarget.shape[0])])
    pixstd /= np.array([np.nanstd(convolvedwRDI[i] * maskleft)
                        for i in range(convolvedwRDI.shape[0])])
    return pixstd


### Functions from Neil Zimmerman
def vectorize_image_series(image_series, data_mask_nan):
    data_ind = np.where( ~np.isnan(data_mask_nan) )
    nan_ind = np.where( np.isnan(data_mask_nan) )
    N_pix = len(data_ind[0])
    N_mask = len(nan_ind[0])
    vec_series = image_series[:,data_ind[0],data_ind[1]]
    
    return vec_series, data_ind, nan_ind, N_pix, N_mask
    
def reconstruct_image_series(vec_series, data_ind, nan_ind, height, width):
    N_pix = len(data_ind[0])
    N_mask = len(nan_ind[0])
    reconst_cube = np.zeros((vec_series.shape[0], height, width))
    for i in range(N_pix):
        reconst_cube[:,data_ind[0][i],data_ind[1][i]] = vec_series[:,i]
    for i in range(N_mask):
        reconst_cube[:,nan_ind[0][i],nan_ind[1][i]] = np.nan
        
    return reconst_cube    
    
def get_correlation_matrix(data_vec_series):
    Nt = data_vec_series.shape[0]
    corr_matx = np.zeros((Nt, Nt))
    for ti in range(Nt):
        for tj in range(ti+1):
            corrcoef_matx = np.corrcoef(data_vec_series[ti], data_vec_series[tj])
            corr_matx[ti, tj] = corrcoef_matx[0][1]
            corr_matx[tj, ti] = corr_matx[ti, tj]

    return corr_matx