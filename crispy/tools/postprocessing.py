import numpy as np
import astropy.units as u
import astropy.constants as c
from inputScene import convert_krist_cube,calc_contrast,calc_contrast_Bijan
import glob
from IFS import propagateIFS,reduceIFSMap
from initLogger import getLogger
log = getLogger('crispy')
from image import Image
try:
    from astropy.io import fits
except:
    import pyfits as fits
from time import time
import os
from detector import averageDetectorReadout,noiselessDetector,readDetector

import multiprocessing
from par_utils import Task, Consumer

def process_SPC_IFS(par,
                    psf_time_series_folder,
                    offaxis_psf_filename,
                    planet_radius = 1.27*c.R_jup,
                    mean_contrast=1e-8,
                    ref_star_T=9377*u.K, ref_star_Vmag=2.37,
                    target_star_T=5887*u.K, target_star_Vmag=5.03,
                    lamc=770.,BW=0.18,n_ref_star_imgs=30,
                    tel_pupil_area=3.650265060424805*u.m**2,
                    outdir_time_series = 'OS5',
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
    
    times = {'Start':time()}

    ###################################################################################
    # Step 1: Convert all the cubes to photons/seconds
    ###################################################################################

    # load the filenames
    filelist = glob.glob(psf_time_series_folder+'/*')
    filelist.sort()
    
    # load first filelist to get its shape
    fileshape = Image(filename=filelist[0]).data.shape
    
    Nlam = fileshape[0]
    lamlist = lamc*np.linspace(1.-BW/2.,1.+BW/2.,Nlam)*u.nm

    # reference and target star cube conversions
    ref_star_cube = convert_krist_cube(fileshape,lamlist,ref_star_T,ref_star_Vmag,tel_pupil_area)
    target_star_cube = convert_krist_cube(fileshape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)

    times['Cube conversion'] = time()

    ###################################################################################
    # Step 2: Process all the cubes and directly apply detector
    ###################################################################################
    ref_outlist = []
    target_outlist = []
    # Parallelized propagation
    if parallel:
        if process_cubes:
            tasks = multiprocessing.Queue()
            results = multiprocessing.Queue()
            ncpus = multiprocessing.cpu_count()
            consumers = [ Consumer(tasks, results)
                          for i in range(ncpus) ]
            for w in consumers:
                w.start()

            # you call the function here, with all its arguments in a list
            for i in range(len(filelist)):
                reffile = filelist[i]
                log.info('Processing file '+reffile.split('/')[-1])
                cube = fits.open(reffile)[0]
                if i<n_ref_star_imgs:
                    cube.data*=ref_star_cube
                else:
                    cube.data*=target_star_cube
                # adjust headers for slightly different wavelength
                log.debug('Modifying cube header')
                if cube.header['LAM_C']==0.8:
                    if lamc==770.:
                        cube.header['LAM_C']=0.77
                        # by NOT changing the pixelsize, we implicitly assume that the PSF is the same at 770 then at 800
                    else:
                        cube.header['LAM_C']=lamc/1000.
                        cube.header['PIXSIZE']*=lamc/0.77
                else:
                    cube.header['LAM_C']=lamc/1000.
                    cube.header['PIXSIZE']*=lamc/0.77
                par.saveDetector=False  

                tasks.put(Task(i, propagateIFS, (par, lamlist.value/1000.,cube)))

            for i in range(ncpus):
                tasks.put(None)

            for i in range(len(filelist)):
                index, result = results.get()
                reffile = filelist[index]
                if index<n_ref_star_imgs:
                    Image(data = result,header=par.hdr).write(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits',clobber=True)
                    ref_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits')
                else:
                    Image(data = result,header=par.hdr).write(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits',clobber=True)
                    target_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits')
                

        else:
            for i in range(len(filelist)):
                reffile = filelist[i]
                if i<n_ref_star_imgs:
                    ref_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits')
                else:
                    target_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits')
            

    else:
        for i in range(len(filelist)):
            reffile = filelist[i]
            if process_cubes:
                log.info('Processing file '+reffile.split('/')[-1])
                cube = fits.open(reffile)[0]
                if i<n_ref_star_imgs:
                    cube.data*=ref_star_cube
                else:
                    cube.data*=target_star_cube
                # adjust headers for slightly different wavelength
                log.debug('Modifying cube header')
                if cube.header['LAM_C']==0.8:
                    if lamc==770.:
                        cube.header['LAM_C']=0.77
                        # by NOT changing the pixelsize, we implicitly assume that the PSF is the same at 770 then at 800
                    else:
                        cube.header['LAM_C']=lamc/1000.
                        cube.header['PIXSIZE']*=lamc/0.77
                else:
                    cube.header['LAM_C']=lamc/1000.
                    cube.header['PIXSIZE']*=lamc/0.77
                par.saveDetector=False  

                detectorFrame = propagateIFS(par,lamlist.value/1000.,cube)

                if i<n_ref_star_imgs:
                    Image(data = detectorFrame,header=par.hdr).write(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits',clobber=True)
                    ref_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits')
                else:
                    Image(data = detectorFrame,header=par.hdr).write(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits',clobber=True)
                    target_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits')
            else:

                if i<n_ref_star_imgs:
                    ref_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits')
                else:
                    target_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits')
            
    times['Process cubes through IFS'] = time()

    ###################################################################################
    # Step 3: Process the off-axis PSF in the same way; also process a flipped version
    ###################################################################################
    if process_offaxis:
        offaxiscube = Image(offaxis_psf_filename)
        print('Processing file '+offaxis_psf_filename)

        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube.data.shape[1] < fileshape[1]:
            diff = fileshape[1]-offaxiscube.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:,diff//2:-diff//2,diff//2:-diff//2] += offaxiscube.data
            offaxiscube = Image(data=offaxiscube_recentered,header = offaxiscube.header)
        offaxis_star_cube = convert_krist_cube(offaxiscube.data.shape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)
        contrast = calc_contrast_Bijan(lamlist.value,mean_contrast=mean_contrast)

        contrast_cube = np.zeros(offaxiscube.data.shape)
        for i in range(offaxiscube.data.shape[0]):
            contrast_cube[i,:,:] += contrast[i]*offaxiscube.data.shape[0]
        offaxiscube.data*=offaxis_star_cube*contrast_cube


        # adjust headers for slightly different wavelength
        log.debug('Modifying cube header')
        if offaxiscube.header['LAM_C']==0.8:
            if lamc==770.:
                offaxiscube.header['LAM_C']=0.77
                # by NOT changing the pixelsize, we implicitly assume that the PSF is the same at 770 then at 800
            else:
                offaxiscube.header['LAM_C']=lamc/1000.
                offaxiscube.header['PIXSIZE']*=lamc/770.
        else:
            offaxiscube.header['LAM_C']=lamc/1000.
            offaxiscube.header['PIXSIZE']*=lamc/770.
        par.saveDetector=False
        offaxiscube.write(outdir_average+'/offaxiscube_processed.fits',clobber=True)
        detectorFrame = propagateIFS(par,lamlist.value/1000.,offaxiscube)
        Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/offaxis.fits',clobber=True)

        offaxiscube_flipped = fits.open(offaxis_psf_filename)[0]
        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube_flipped.data.shape[1] < fileshape[1]:
            diff = fileshape[1]-offaxiscube_flipped.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:,diff//2:-diff//2,diff//2:-diff//2] += offaxiscube_flipped.data
            offaxiscube_flipped = Image(data=offaxiscube_recentered,header = offaxiscube.header)

        for i in range(offaxiscube_flipped.data.shape[0]):
            offaxiscube_flipped.data[i,:,:] = np.fliplr(offaxiscube_flipped.data[i,:,:])
        offaxiscube_flipped.data*=offaxis_star_cube*contrast_cube
        detectorFrame = propagateIFS(par,lamlist.value/1000.,offaxiscube_flipped)
        Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/offaxis_flipped.fits',clobber=True)

    times['Process off-axis PSF through IFS'] = time()

    ###################################################################################
    # Step 4: Add the off-axis PSF before reading on the detector
    ###################################################################################

    if process_detector:
        # Apply detector for both reference star and target
        ref_det_outlist = averageDetectorReadout(par,ref_outlist,outdir_detector)   
        offaxis_filename = os.path.abspath(outdir_average+'/offaxis.fits')
        target_det_outlist = averageDetectorReadout(par,target_outlist,outdir_detector,offaxis = offaxis_filename)
    elif process_noiseless:
        ref_det_outlist = noiselessDetector(par,ref_outlist,outdir_detector)   
        offaxis_filename = os.path.abspath(outdir_average+'/offaxis.fits')
        target_det_outlist = noiselessDetector(par,target_outlist,outdir_detector,offaxis = offaxis_filename)
    else:
        ref_det_outlist = []
        target_det_outlist = []
        suffix='detector'
        for reffile in ref_outlist:
            ref_det_outlist.append(outdir_detector+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
        for reffile in target_outlist:
            target_det_outlist.append(outdir_detector+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
    times['Construct IFS detector'] = time()


    ###################################################################################
    # Step 5: Take averages
    ###################################################################################
    
    if take_averages:
        log.info('Taking averages')
        ref_star_average = np.zeros(Image(filename=ref_det_outlist[0]).data.shape)
        target_star_average = np.zeros(Image(filename=target_det_outlist[0]).data.shape)
        for reffile in ref_det_outlist:
            ref_star_average += Image(filename=reffile).data
        #ref_star_average /= len(ref_det_outlist)
        for reffile in target_det_outlist:
            target_star_average += Image(filename=reffile).data
        #target_star_average /= len(target_det_outlist)
        #Image(data=ref_star_average,header=par.hdr,extraheader=Image(ref_det_outlist[0]).header).write(outdir_average+'/average_ref_star_detector.fits',clobber=True)
        #Image(data=target_star_average,header=par.hdr,extraheader=Image(ref_det_outlist[0]).header).write(outdir_average+'/average_target_star_detector.fits',clobber=True)
        Image(data=ref_star_average,header=par.hdr).write(outdir_average+'/average_ref_star_detector.fits',clobber=True)
        Image(data=target_star_average,header=par.hdr).write(outdir_average+'/average_target_star_detector.fits',clobber=True)

    times['Taking averages'] = time()

    ###################################################################################
    # Step 6: Process the cubes
    ###################################################################################
    print(os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
    img = Image(filename=os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
    print(img.data.shape)
    ref_cube = reduceIFSMap(par,os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
    target_cube = reduceIFSMap(par,os.path.abspath(outdir_average+'/average_target_star_detector.fits'))
    offaxis_ideal = reduceIFSMap(par,os.path.abspath(outdir_average+'/offaxis.fits'))
    offaxis_ideal_flipped = reduceIFSMap(par,os.path.abspath(outdir_average+'/offaxis_flipped.fits'))

    times['Process average cubes'] = time()
    ###################################################################################
    # Step 7: Naive PSF subtraction; match the flux using the Vmag difference
    ###################################################################################
    ref_cube_noave = ref_cube.data -np.nanmean(ref_cube.data)
    target_cube_noave = target_cube.data - np.nanmean(target_cube.data)
    residual = target_cube_noave - 10**(0.4*(ref_star_Vmag-target_star_Vmag))*ref_cube_noave
    Image(data=residual).write(outdir_average+'/residual.fits',clobber=True)

    times['Normalize and subtract reference PSF'] = time()

    ###################################################################################
    # Step 8: Construct an off-axis PSF matched filter and propagate it through the IFS
    ###################################################################################

    # First off, normalize the residual cube by a flatfield
    flatfield = Image(par.exportDir+'/flatfield_red_optext.fits')
    residual /= flatfield.data
    residual[np.isnan(flatfield.data)]=np.nan
    Image(data=residual).write(outdir_average+'/residual_flatfielded.fits',clobber=True)
    
    # loop over all the slices in the cube:
    matched_filter = np.zeros(residual.shape)
    matched_filter_flipped = np.zeros(residual.shape)
    signal = np.zeros(residual.shape[0]) # on source
    off = np.zeros(residual.shape[0])   # off source
    mf_npix = np.zeros(residual.shape[0]) # effective background area of matched filter
    noise = np.zeros(residual.shape[0]) # noise
    for slicenum in range(residual.shape[0]):
        # ON
        offaxis_ideal_norm = offaxis_ideal.data[slicenum]/np.nansum(offaxis_ideal.data[slicenum])
        matched_filter[slicenum,:,:] = offaxis_ideal_norm/np.nansum((offaxis_ideal_norm)**2)
        signal[slicenum] = np.nansum(matched_filter[slicenum,:,:]*residual[slicenum,:,:])
        # OFF
        offaxis_ideal_flipped_norm = offaxis_ideal_flipped.data[slicenum]/np.nansum(offaxis_ideal_flipped.data[slicenum])
        matched_filter_flipped[slicenum,:,:] = offaxis_ideal_flipped_norm/np.nansum((offaxis_ideal_flipped_norm)**2)
        off[slicenum] = np.nansum(matched_filter_flipped[slicenum,:,:]*residual[slicenum,:,:])
    mf_npix = np.nansum(np.nansum(matched_filter,axis=2),axis=1)
    
    ###################################################################################
    # Step 9: Determine the pixel noise in the dark hole
    ###################################################################################
    from tools.imgtools import bowtie
    ydim,xdim = residual[0].shape
    maskleft,maskright = bowtie(residual[0],ydim//2,xdim//2,openingAngle=65,
                clocking=-par.philens*180./np.pi,IWApix=6*0.77/0.6,OWApix=18*0.77/0.6,
                export='bowtie',twomasks=True)
    # PSF is in the right mask
    pixstd = [np.nanstd(residual[i,:,:]*maskright) for i in range(residual.shape[0])]
    noise = np.sqrt(2*mf_npix)*pixstd # twice the num of pix since we subtract the off field
    
    Image(data=matched_filter).write(outdir_average+'/matched_filter.fits',clobber=True)
    Image(data=matched_filter_flipped).write(outdir_average+'/matched_filter_flipped.fits',clobber=True)
    
    times['Computed signal and noise arrays'] = time()
    
    log.info('Cube conversion: %.3f' % (times['Cube conversion']-times['Start']))
    log.info('Process cubes through IFS: %.3f' % (times['Process cubes through IFS']-times['Cube conversion']))
    log.info('Process off-axis PSF through IFS: %.3f' % (times['Process off-axis PSF through IFS']-times['Process cubes through IFS']))
    log.info('Construct IFS detector: %.3f' % (times['Construct IFS detector']-times['Process off-axis PSF through IFS']))
    log.info('Taking averages: %.3f' % (times['Taking averages']-times['Construct IFS detector']))  
    log.info('Process average cubes: %.3f' % (times['Process average cubes']-times['Taking averages']))
    log.info('Normalize and subtract reference PSF: %.3f' % (times['Normalize and subtract reference PSF']-times['Process average cubes']))
    log.info('Computed signal and noise arrays: %.3f' % (times['Computed signal and noise arrays']-times['Normalize and subtract reference PSF']))
    log.info('Total time: %.3f' % (times['Computed signal and noise arrays']-times['Start']))

    return signal-off,noise
    
def SPC_IFS_systematics(par,psf_time_series_folder,offaxis_psf_filename,
                    planet_radius = 1*c.R_jup,mean_contrast=1e-8,
                    ref_star_T=9377*u.K, ref_star_Vmag=2.37, target_star_T=5887*u.K, target_star_Vmag=5.03,
                    lamc=770.,BW=0.18,Nlam=45,n_ref_star_imgs=30,tel_pupil_area=3.650265060424805*u.m**2,
                    outdir_time_series = 'OS5',outdir_detector='OS5/OS5_detector',outdir_average='OS5/OS5_average',
                    process_cubes=True,process_offaxis=True,process_detector=True,take_averages=True):
    '''
    Process SPC PSF cubes from J. Krist through the IFS


    '''
    
    times = {'Start':time()}

    ###################################################################################
    # Step 1: Convert all the cubes to photons/seconds
    ###################################################################################

    lamlist = lamc*np.linspace(1.-BW/2.,1.+BW/2.,Nlam)*u.nm

    # load the filenames
    filelist = glob.glob(psf_time_series_folder+'/*')
    filelist.sort()
    
    # load first filelist to get its shape
    fileshape = Image(filename=filelist[0]).data.shape

    # reference and target star cube conversions
    ref_star_cube = convert_krist_cube(fileshape,lamlist,ref_star_T,ref_star_Vmag,tel_pupil_area)
    target_star_cube = convert_krist_cube(fileshape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)

    times['Cube conversion'] = time()

    ###################################################################################
    # Step 2: Process all the cubes and directly apply detector
    ###################################################################################
    ref_outlist = []
    target_outlist = []
    # construct the list of files
    for i in range(len(filelist)):
        reffile = filelist[i]
        if i<n_ref_star_imgs:
            ref_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits')
        else:
            target_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits')
            
    times['Process cubes through IFS'] = time()

    ###################################################################################
    # Step 3: Process the off-axis PSF in the same way; also process a flipped version
    ###################################################################################
    if process_offaxis:
        offaxiscube = Image(offaxis_psf_filename)
        log.info('Processing file '+offaxis_psf_filename)

        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube.data.shape[1] < fileshape[1]:
            diff = fileshape[1]-offaxiscube.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:,diff//2:-diff//2,diff//2:-diff//2] += offaxiscube.data
            offaxiscube = Image(data=offaxiscube_recentered,header = offaxiscube.header)
        offaxis_star_cube = convert_krist_cube(offaxiscube.data.shape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)
        contrast = calc_contrast(lamlist.value,mean_contrast=mean_contrast)

        contrast_cube = np.zeros(offaxiscube.data.shape)
        for i in range(offaxiscube.data.shape[0]):
            contrast_cube[i,:,:] += contrast[i]
        offaxiscube.data*=offaxis_star_cube*contrast_cube
        offaxiscube.write(outdir_average+'/offaxiscubePhPerSec.fits',clobber=True)

        
        # adjust headers for slightly different wavelength
        log.info('Modifying cube header')
        if offaxiscube.header['LAM_C']==0.8:
            if lamc==770.:
                offaxiscube.header['LAM_C']=0.77
                # by NOT changing the pixelsize, we implicitly assume that the PSF is the same at 770 then at 800
            else:
                offaxiscube.header['LAM_C']=lamc/1000.
                offaxiscube.header['PIXSIZE']*=lamc/770.
        else:
            offaxiscube.header['LAM_C']=lamc/1000.
            offaxiscube.header['PIXSIZE']*=lamc/770.
        par.saveDetector=False

        detectorFrame = propagateIFS(par,lamlist.value/1000.,offaxiscube)
        Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/offaxis.fits',clobber=True)

        offaxiscube_flipped = fits.open(offaxis_psf_filename)[0]
        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube_flipped.data.shape[1] < fileshape[1]:
            diff = fileshape[1]-offaxiscube_flipped.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:,diff//2:-diff//2,diff//2:-diff//2] += offaxiscube_flipped.data
            offaxiscube_flipped = Image(data=offaxiscube_recentered,header = offaxiscube.header)

        for i in range(offaxiscube_flipped.data.shape[0]):
            offaxiscube_flipped.data[i,:,:] = np.fliplr(offaxiscube_flipped.data[i,:,:])
        offaxiscube_flipped.data*=offaxis_star_cube*contrast_cube
        detectorFrame = propagateIFS(par,lamlist.value/1000.,offaxiscube_flipped)
        Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/offaxis_flipped.fits',clobber=True)

    times['Process off-axis PSF through IFS'] = time()

    ###################################################################################
    # Step 4: Only process the off-axis target, no noise added
    ###################################################################################

#   if process_detector:
        # Apply detector for both reference star and target
#       ref_det_outlist = averageDetectorReadout(par,ref_outlist,outdir_detector)   
#       offaxis_filename = os.path.abspath(outdir_average+'/offaxis.fits')
#       target_det_outlist = averageDetectorReadout(par,target_outlist,outdir_detector,offaxis = offaxis_filename)
    im_offaxis = Image(outdir_average+'/offaxis.fits')
    im_offaxis.data *= par.QE*par.losses *par.timeframe/par.Nreads
    im_offaxis.write(outdir_average+'/offaxis_at_detector.fits')

#   else:
#       ref_det_outlist = []
#       target_det_outlist = []
#       suffix='detector'
#       for reffile in ref_outlist:
#           ref_det_outlist.append(outdir_detector+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
#       for reffile in target_outlist:
#           target_det_outlist.append(outdir_detector+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
    times['Construct IFS detector'] = time()


    ###################################################################################
    # Step 5: Take averages
    ###################################################################################
    # no need to take averages since there is no noise
#   if take_averages:
#       log.info('Taking averages')
#       ref_star_average = np.zeros(Image(filename=ref_det_outlist[0]).data.shape)
#       target_star_average = np.zeros(Image(filename=target_det_outlist[0]).data.shape)
#       for reffile in ref_det_outlist:
#           ref_star_average += Image(filename=reffile).data
#       ref_star_average /= len(ref_det_outlist)
#       for reffile in target_det_outlist:
#           target_star_average += Image(filename=reffile).data
#       target_star_average /= len(target_det_outlist)
#       #Image(data=ref_star_average,header=par.hdr,extraheader=Image(ref_det_outlist[0]).header).write(outdir_average+'/average_ref_star_detector.fits',clobber=True)
#       #Image(data=target_star_average,header=par.hdr,extraheader=Image(ref_det_outlist[0]).header).write(outdir_average+'/average_target_star_detector.fits',clobber=True)
#       Image(data=ref_star_average,header=par.hdr).write(outdir_average+'/average_ref_star_detector.fits',clobber=True)
#       Image(data=target_star_average,header=par.hdr).write(outdir_average+'/average_target_star_detector.fits',clobber=True)
# 
    times['Taking averages'] = time()

    ###################################################################################
    # Step 6: Process the cubes
    ###################################################################################
#   print(os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
#   img = Image(filename=os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
#   print(img.data.shape)
#   ref_cube = reduceIFSMap(par,os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
#   target_cube = reduceIFSMap(par,os.path.abspath(outdir_average+'/average_target_star_detector.fits'))
    offaxis_ideal = reduceIFSMap(par,os.path.abspath(outdir_average+'/offaxis.fits'))
    offaxis_ideal_flipped = reduceIFSMap(par,os.path.abspath(outdir_average+'/offaxis_flipped.fits'))

    residualImg = reduceIFSMap(par,outdir_average+'/offaxis_at_detector.fits')
    
    times['Process average cubes'] = time()
    ###################################################################################
    # Step 7: Naive PSF subtraction; match the flux using the Vmag difference
    ###################################################################################
#   ref_cube_noave = ref_cube.data -np.nanmean(ref_cube.data)
#   target_cube_noave = target_cube.data - np.nanmean(target_cube.data)
#   residual = target_cube_noave - 10**(0.4*(ref_star_Vmag-target_star_Vmag))*ref_cube_noave
#   Image(data=residual).write(outdir_average+'/residual.fits',clobber=True)

    times['Normalize and subtract reference PSF'] = time()

    ###################################################################################
    # Step 8: Construct an off-axis PSF matched filter and propagate it through the IFS
    ###################################################################################

    # First off, normalize the residual cube by a flatfield
    flatfield = Image(par.exportDir+'/flatfield_red_optext.fits')
    residual = residualImg.data
    residual /= flatfield.data
    residual[np.isnan(flatfield.data)]=np.nan
    Image(data=residual).write(outdir_average+'/residual_flatfielded.fits',clobber=True)
    
    # loop over all the slices in the cube:
    matched_filter = np.zeros(residual.shape)
    matched_filter_flipped = np.zeros(residual.shape)
    signal = np.zeros(residual.shape[0]) # on source
    off = np.zeros(residual.shape[0])   # off source
    mf_npix = np.zeros(residual.shape[0]) # effective background area of matched filter
    noise = np.zeros(residual.shape[0]) # noise
    for slicenum in range(residual.shape[0]):
        # ON
        offaxis_ideal_norm = offaxis_ideal.data[slicenum]/np.nansum(offaxis_ideal.data[slicenum])
        matched_filter[slicenum,:,:] = offaxis_ideal_norm/np.nansum((offaxis_ideal_norm)**2)
        signal[slicenum] = np.nansum(matched_filter[slicenum,:,:]*residual[slicenum,:,:])
        # OFF
        offaxis_ideal_flipped_norm = offaxis_ideal_flipped.data[slicenum]/np.nansum(offaxis_ideal_flipped.data[slicenum])
        matched_filter_flipped[slicenum,:,:] = offaxis_ideal_flipped_norm/np.nansum((offaxis_ideal_flipped_norm)**2)
        off[slicenum] = np.nansum(matched_filter_flipped[slicenum,:,:]*residual[slicenum,:,:])
    mf_npix = np.nansum(np.nansum(matched_filter,axis=2),axis=1)
    
    ###################################################################################
    # Step 9: Determine the pixel noise in the dark hole
    ###################################################################################
    from tools.imgtools import bowtie
    ydim,xdim = residual[0].shape
    maskleft,maskright = bowtie(residual[0],ydim//2,xdim//2,openingAngle=65,
                clocking=-par.philens*180./np.pi,IWApix=6*0.77/0.6,OWApix=18*0.77/0.6,
                export='bowtie',twomasks=True)
    # PSF is in the right mask
    pixstd = [np.nanstd(residual[i,:,:]*maskright) for i in range(residual.shape[0])]
    noise = np.sqrt(2*mf_npix)*pixstd # twice the num of pix since we subtract the off field
    
    Image(data=matched_filter).write(outdir_average+'/matched_filter.fits',clobber=True)
    Image(data=matched_filter_flipped).write(outdir_average+'/matched_filter_flipped.fits',clobber=True)
    
    times['Computed signal and noise arrays'] = time()
    
    log.info('Cube conversion: %.3f' % (times['Cube conversion']-times['Start']))
    log.info('Process cubes through IFS: %.3f' % (times['Process cubes through IFS']-times['Cube conversion']))
    log.info('Process off-axis PSF through IFS: %.3f' % (times['Process off-axis PSF through IFS']-times['Process cubes through IFS']))
    log.info('Construct IFS detector: %.3f' % (times['Construct IFS detector']-times['Process off-axis PSF through IFS']))
    log.info('Taking averages: %.3f' % (times['Taking averages']-times['Construct IFS detector']))  
    log.info('Process average cubes: %.3f' % (times['Process average cubes']-times['Taking averages']))
    log.info('Normalize and subtract reference PSF: %.3f' % (times['Normalize and subtract reference PSF']-times['Process average cubes']))
    log.info('Computed signal and noise arrays: %.3f' % (times['Computed signal and noise arrays']-times['Normalize and subtract reference PSF']))
    log.info('Total time: %.3f' % (times['Computed signal and noise arrays']-times['Start']))

    return signal,noise,off

def SPC_process_offaxis_only(par,offaxis_psf_filename,
                    Naverage = 100, fileshape=(45,315,315),planet_radius = 1*c.R_jup,mean_contrast=1e-8,
                    ref_star_T=9377*u.K, ref_star_Vmag=2.37, target_star_T=5887*u.K, target_star_Vmag=5.03,
                    lamc=770.,BW=0.18,Nlam=45,n_ref_star_imgs=30,tel_pupil_area=3.650265060424805*u.m**2,
                    outdir_time_series = 'OS5',outdir_detector='OS5/OS5_detector',outdir_average='OS5/OS5_average',
                    process_offaxis=True,process_detector=True,take_averages=True):
    '''
    Process SPC PSF cubes from J. Krist through the IFS


    '''
    
    times = {'Start':time()}

    ###################################################################################
    # Step 1: Convert all the cubes to photons/seconds
    ###################################################################################

    lamlist = lamc*np.linspace(1.-BW/2.,1.+BW/2.,Nlam)*u.nm
    
    target_star_cube = convert_krist_cube(fileshape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)

    times['Cube conversion'] = time()

    times['Process cubes through IFS'] = time()

    ###################################################################################
    # Step 3: Process the off-axis PSF in the same way; also process a flipped version
    ###################################################################################
    if process_offaxis:
        offaxiscube = Image(offaxis_psf_filename)
        log.info('Processing file '+offaxis_psf_filename)

        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube.data.shape[1] < fileshape[1]:
            diff = fileshape[1]-offaxiscube.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:,diff//2:-diff//2,diff//2:-diff//2] += offaxiscube.data
            offaxiscube = Image(data=offaxiscube_recentered,header = offaxiscube.header)
        offaxis_star_cube = convert_krist_cube(offaxiscube.data.shape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)
        contrast = calc_contrast(lamlist.value,mean_contrast=mean_contrast)

        contrast_cube = np.zeros(offaxiscube.data.shape)
        for i in range(offaxiscube.data.shape[0]):
            contrast_cube[i,:,:] += contrast[i]
        offaxiscube.data*=offaxis_star_cube*contrast_cube

        
        # adjust headers for slightly different wavelength
        log.info('Modifying cube header')
        if offaxiscube.header['LAM_C']==0.8:
            if lamc==770.:
                offaxiscube.header['LAM_C']=0.77
                # by NOT changing the pixelsize, we implicitly assume that the PSF is the same at 770 then at 800
            else:
                offaxiscube.header['LAM_C']=lamc/1000.
                offaxiscube.header['PIXSIZE']*=lamc/770.
        else:
            offaxiscube.header['LAM_C']=lamc/1000.
            offaxiscube.header['PIXSIZE']*=lamc/770.
        par.saveDetector=False

        detectorFrame = propagateIFS(par,lamlist.value/1000.,offaxiscube)
        offaxiscube.write(outdir_average+'/offaxiscubePhPerSec.fits',clobber=True)
        Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/offaxis.fits',clobber=True)

        offaxiscube_flipped = fits.open(offaxis_psf_filename)[0]
        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube_flipped.data.shape[1] < fileshape[1]:
            diff = fileshape[1]-offaxiscube_flipped.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:,diff//2:-diff//2,diff//2:-diff//2] += offaxiscube_flipped.data
            offaxiscube_flipped = Image(data=offaxiscube_recentered,header = offaxiscube.header)

        for i in range(offaxiscube_flipped.data.shape[0]):
            offaxiscube_flipped.data[i,:,:] = np.fliplr(offaxiscube_flipped.data[i,:,:])
        offaxiscube_flipped.data*=offaxis_star_cube*contrast_cube
        detectorFrame = propagateIFS(par,lamlist.value/1000.,offaxiscube_flipped)
        Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/offaxis_flipped.fits',clobber=True)

    times['Process off-axis PSF through IFS'] = time()

    ###################################################################################
    # Step 4: Only process the off-axis target, no noise added
    ###################################################################################

    if process_detector:
        # Apply detector 
        img = Image(filename=outdir_average+'/offaxis.fits')
        inttime = par.timeframe/par.Nreads
        img.data*=par.QE*par.losses

        frame = np.zeros(img.data.shape)
        varframe = np.zeros(img.data.shape)
        # averaging reads
        for i in range(par.Nreads*Naverage):
            newread = readDetector(par,img,inttime=inttime)
            frame += newread
            varframe += newread**2
        frame /= par.Nreads*Naverage
        varframe /= par.Nreads*Naverage
        varframe -= frame**2
        Image(data=frame,header=par.hdr).write(outdir_average+'/offaxis_only_detectorized.fits')

#       target_det_outlist = averageDetectorReadout(par,target_outlist,outdir_detector,offaxis = outdir_average+'/offaxis.fits')
#     im_offaxis = Image(outdir_average+'/offaxis.fits')
#     im_offaxis.data *= par.QE*par.losses *par.timeframe/par.Nreads
#     im_offaxis.write(outdir_average+'/offaxis_at_detector.fits')

#   else:
#       ref_det_outlist = []
#       target_det_outlist = []
#       suffix='detector'
#       for reffile in ref_outlist:
#           ref_det_outlist.append(outdir_detector+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
#       for reffile in target_outlist:
#           target_det_outlist.append(outdir_detector+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
    times['Construct IFS detector'] = time()


    ###################################################################################
    # Step 5: Take averages
    ###################################################################################
    # no need to take averages since there is no noise
#   if take_averages:
#       log.info('Taking averages')
#       ref_star_average = np.zeros(Image(filename=ref_det_outlist[0]).data.shape)
#       target_star_average = np.zeros(Image(filename=target_det_outlist[0]).data.shape)
#       for reffile in ref_det_outlist:
#           ref_star_average += Image(filename=reffile).data
#       ref_star_average /= len(ref_det_outlist)
#       for reffile in target_det_outlist:
#           target_star_average += Image(filename=reffile).data
#       target_star_average /= len(target_det_outlist)
#       #Image(data=ref_star_average,header=par.hdr,extraheader=Image(ref_det_outlist[0]).header).write(outdir_average+'/average_ref_star_detector.fits',clobber=True)
#       #Image(data=target_star_average,header=par.hdr,extraheader=Image(ref_det_outlist[0]).header).write(outdir_average+'/average_target_star_detector.fits',clobber=True)
#       Image(data=ref_star_average,header=par.hdr).write(outdir_average+'/average_ref_star_detector.fits',clobber=True)
#       Image(data=target_star_average,header=par.hdr).write(outdir_average+'/average_target_star_detector.fits',clobber=True)
# 
    times['Taking averages'] = time()

    ###################################################################################
    # Step 6: Process the cubes
    ###################################################################################
#   print(os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
#   img = Image(filename=os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
#   print(img.data.shape)
#   ref_cube = reduceIFSMap(par,os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
#   target_cube = reduceIFSMap(par,os.path.abspath(outdir_average+'/average_target_star_detector.fits'))
    offaxis_ideal = reduceIFSMap(par,os.path.abspath(outdir_average+'/offaxis.fits'))
    offaxis_ideal_flipped = reduceIFSMap(par,os.path.abspath(outdir_average+'/offaxis_flipped.fits'))

    residualImg = reduceIFSMap(par,outdir_average+'/offaxis_only_detectorized.fits')
    
    times['Process average cubes'] = time()
    ###################################################################################
    # Step 7: Naive PSF subtraction; match the flux using the Vmag difference
    ###################################################################################
#   ref_cube_noave = ref_cube.data -np.nanmean(ref_cube.data)
#   target_cube_noave = target_cube.data - np.nanmean(target_cube.data)
#   residual = target_cube_noave - 10**(0.4*(ref_star_Vmag-target_star_Vmag))*ref_cube_noave
#   Image(data=residual).write(outdir_average+'/residual.fits',clobber=True)

    times['Normalize and subtract reference PSF'] = time()

    ###################################################################################
    # Step 8: Construct an off-axis PSF matched filter and propagate it through the IFS
    ###################################################################################

    # First off, normalize the residual cube by a flatfield
    flatfield = Image(par.exportDir+'/flatfield_red_optext.fits')
    residual = residualImg.data
    residual /= flatfield.data
    residual[np.isnan(flatfield.data)]=np.nan
    Image(data=residual,header=par.hdr).write(outdir_average+'/residual_flatfielded.fits',clobber=True)
    
    # loop over all the slices in the cube:
    matched_filter = np.zeros(residual.shape)
    matched_filter_flipped = np.zeros(residual.shape)
    signal = np.zeros(residual.shape[0]) # on source
    off = np.zeros(residual.shape[0])   # off source
    mf_npix = np.zeros(residual.shape[0]) # effective background area of matched filter
    noise = np.zeros(residual.shape[0]) # noise
    for slicenum in range(residual.shape[0]):
        # ON
        offaxis_ideal_norm = offaxis_ideal.data[slicenum]/np.nansum(offaxis_ideal.data[slicenum])
        matched_filter[slicenum,:,:] = offaxis_ideal_norm/np.nansum((offaxis_ideal_norm)**2)
        signal[slicenum] = np.nansum(matched_filter[slicenum,:,:]*residual[slicenum,:,:])
        # OFF
        offaxis_ideal_flipped_norm = offaxis_ideal_flipped.data[slicenum]/np.nansum(offaxis_ideal_flipped.data[slicenum])
        matched_filter_flipped[slicenum,:,:] = offaxis_ideal_flipped_norm/np.nansum((offaxis_ideal_flipped_norm)**2)
        off[slicenum] = np.nansum(matched_filter_flipped[slicenum,:,:]*residual[slicenum,:,:])
    mf_npix = np.nansum(np.nansum(matched_filter,axis=2),axis=1)
    
    ###################################################################################
    # Step 9: Determine the pixel noise in the dark hole
    ###################################################################################
    from tools.imgtools import bowtie
    ydim,xdim = residual[0].shape
    maskleft,maskright = bowtie(residual[0],ydim//2,xdim//2,openingAngle=65,
                clocking=-par.philens*180./np.pi,IWApix=6*0.77/0.6,OWApix=18*0.77/0.6,
                export='bowtie',twomasks=True)
    # PSF is in the right mask
    pixstd = [np.nanstd(residual[i,:,:]*maskright) for i in range(residual.shape[0])]
    noise = np.sqrt(2*mf_npix)*pixstd # twice the num of pix since we subtract the off field
    
    Image(data=matched_filter).write(outdir_average+'/matched_filter.fits',clobber=True)
    Image(data=matched_filter_flipped).write(outdir_average+'/matched_filter_flipped.fits',clobber=True)
    
    times['Computed signal and noise arrays'] = time()
    
    log.info('Cube conversion: %.3f' % (times['Cube conversion']-times['Start']))
    log.info('Process cubes through IFS: %.3f' % (times['Process cubes through IFS']-times['Cube conversion']))
    log.info('Process off-axis PSF through IFS: %.3f' % (times['Process off-axis PSF through IFS']-times['Process cubes through IFS']))
    log.info('Construct IFS detector: %.3f' % (times['Construct IFS detector']-times['Process off-axis PSF through IFS']))
    log.info('Taking averages: %.3f' % (times['Taking averages']-times['Construct IFS detector']))  
    log.info('Process average cubes: %.3f' % (times['Process average cubes']-times['Taking averages']))
    log.info('Normalize and subtract reference PSF: %.3f' % (times['Normalize and subtract reference PSF']-times['Process average cubes']))
    log.info('Computed signal and noise arrays: %.3f' % (times['Computed signal and noise arrays']-times['Normalize and subtract reference PSF']))
    log.info('Total time: %.3f' % (times['Computed signal and noise arrays']-times['Start']))

    return signal,noise,off


import seaborn as sns
from tools.inputScene import calc_contrast
from tools.reduction import calculateWaveList
from scipy import ndimage

def plot_SNR_spectrum(wavelist,mean_contrast):
    lam_midpts,junk = calculateWaveList(par)
    # wavelist = np.arange(min(lam_midpts),max(lam_midpts),3)
    #wavelist = 770*np.linspace(1.-0.18/2.,1.+0.18/2.,45)
    sns.set_style("whitegrid")
    fig,ax = plt.subplots(figsize=(12,6))
    real_vals=calc_contrast(wavelist,mean_contrast=mean_contrast)
    ax.plot(wavelist,calc_contrast(wavelist,mean_contrast=mean_contrast),label='Original spectrum')
    ax.errorbar(lam_midpts,signal*np.mean(spectrum)/np.mean(signal),yerr=noise*np.mean(spectrum)/np.mean(signal),label='Recovered spectrum',fmt='o')
#     ax.plot(lam_midpts,(signal-off)*np.mean(spectrum)/np.mean(signal[1:-1]),'o',label='Recovered spectrum')
    
    FWHM = 4.
    smooth = ndimage.filters.gaussian_filter1d(real_vals,FWHM/2.35,order=0,mode='constant')
    ax.plot(wavelist,smooth,'-',label='Gaussian-smoothed original spectrum w/ FWHM=%.0f bins' % FWHM)

    #ax.errorbar(lam_midpts,(signal),yerr=noise,label='Recovered spectrum',fmt='o')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Contrast')
    ax.set_title('Planet, no star, no noise')
    plt.legend()
    chisq = np.sum((signal*np.mean(spectrum)/np.mean(signal[1:-1]) - calc_contrast(lam_midpts,mean_contrast=mean_contrast))**2/(noise*np.mean(spectrum)/np.mean(signal[1:-1]))**2)
    print (chisq/len(signal))


