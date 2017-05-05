import numpy as np
import astropy.units as u
import astropy.constants as c
from crispy.tools.inputScene import convert_krist_cube,calc_contrast,calc_contrast_Bijan,zodi_cube,adjust_krist_header
import glob
from crispy.IFS import reduceIFSMap,polychromeIFS
from initLogger import getLogger
log = getLogger('crispy')
from crispy.tools.image import Image
try:
    from astropy.io import fits
except:
    import pyfits as fits
from time import time
import os
from crispy.tools.detector import averageDetectorReadout,readDetector
import matplotlib.pyplot as plt

import multiprocessing
from crispy.tools.par_utils import Task, Consumer
import seaborn as sns
from crispy.tools.inputScene import calc_contrast
from crispy.tools.reduction import calculateWaveList
from crispy.tools.imgtools import bowtie,scale2imgs
from crispy.tools.rotate import shiftCube
from scipy import ndimage
from scipy.interpolate import interp1d
from crispy.params import Params



def process_SPC_IFS(par,
                    psf_time_series_folder,
                    offaxis_psf_filename,
                    planet_radius = 1.27*c.R_jup,
                    planet_AU = 3.6,planet_dist_pc=14.1,
                    ref_star_T=9377*u.K, ref_star_Vmag=2.37,
                    target_star_T=5887*u.K, target_star_Vmag=5.03,
                    lamc=770.,BW=0.18,n_ref_star_imgs=30,
                    tel_pupil_area=3.650265060424805*u.m**2,
                    IWA = 3,OWA = 9,
                    xshift=0.0,yshift=0.0, ## need to tie this to planet distance
                    pp_fact = 0.05,
                    t_zodi = 0.09,
                    useQE=True,
                    subtract_ref_psf=True,
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
    
    times = {'Start':time()}

    ###################################################################################
    # Step 1: Convert all the cubes to photons/seconds
    ###################################################################################

    # load the filenames
    filelist = glob.glob(psf_time_series_folder+'/*')
    filelist.sort()
    
    # load first filelist to get its shape
    kristfile = Image(filename=filelist[0])
    fileshape = kristfile.data.shape
    adjust_krist_header(kristfile,lamc=lamc)
    
    Nlam = fileshape[0]
    lamlist = lamc*np.linspace(1.-BW/2.,1.+BW/2.,Nlam)*u.nm

    # reference and target star cube conversions
    ref_star_cube = convert_krist_cube(fileshape,lamlist,ref_star_T,ref_star_Vmag,tel_pupil_area)
    target_star_cube = convert_krist_cube(fileshape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)

    
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
                adjust_krist_header(cube,lamc=lamc)
                par.saveDetector=False  

                tasks.put(Task(i, polychromeIFS, (par, lamlist.value,cube)))

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
                adjust_krist_header(cube,lamc=lamc)
                par.saveDetector=False  

                detectorFrame = polychromeIFS(par,lamlist.value,cube,QE=useQE)

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
        print('Processing file '+offaxis_psf_filename)

        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube.data.shape[1] < fileshape[1]:
            diff = fileshape[1]-offaxiscube.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:,diff//2:-diff//2,diff//2:-diff//2] += offaxiscube.data
            offaxiscube = Image(data=offaxiscube_recentered,header = offaxiscube.header)
        offaxis_star_cube = convert_krist_cube(offaxiscube.data.shape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)
        contrast = calc_contrast_Bijan(lamlist.value)

        contrast_cube = np.zeros(offaxiscube.data.shape)
        for i in range(offaxiscube.data.shape[0]):
            contrast_cube[i,:,:] += contrast[i]
        offaxiscube.data*=offaxis_star_cube*contrast_cube


        # adjust headers for slightly different wavelength
        log.debug('Modifying cube header')
        adjust_krist_header(offaxiscube,lamc=lamc)
        par.saveDetector=False
        Image(data=offaxiscube.data,header=offaxiscube.header).write(outdir_average+'/offaxiscube_processed.fits',clobber=True)
        detectorFrame = polychromeIFS(par,lamlist.value,offaxiscube,QE=useQE)
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
        detectorFrame = polychromeIFS(par,lamlist.value,offaxiscube_flipped,QE=useQE)
        Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/offaxis_flipped.fits',clobber=True)

    ###################################################################################
    # Add the zodi and exozodi, uniform for now as opposed to decreasing
    # as a function of separation from the star; over the full FOV instead of just the bowtie
    ###################################################################################

    local_zodi_mag = 23
    exozodi_mag = 22
    D = 2.37
    pixarea = kristfile.header['PIXSIZE']*lamc*1e-9/D/4.848e-6
    absmag = target_star_Vmag-5*np.log10(planet_dist_pc/10.)
    zodicube = zodi_cube(target_star_cube,
                        area_per_pixel=pixarea,
                        absmag=absmag,
                        Vstarmag=target_star_Vmag,
                        zodi_surfmag=23,exozodi_surfmag=22,
                        distAU=planet_AU,t_zodi=t_zodi)
    
    zodicube = Image(data=zodicube,header=kristfile.header)
    detectorFrame = polychromeIFS(par,lamlist.value,zodicube,QE=useQE)

    times['Cube conversion'] = time()

    Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/zodicube.fits',clobber=True)

    times['Process off-axis PSF through IFS'] = time()

    

    ###################################################################################
    # Step 4: Add the off-axis PSF before reading on the detector
    ###################################################################################

    if process_detector:
        # Apply detector for both reference star and target
        ref_det_outlist = averageDetectorReadout(par,ref_outlist,outdir_detector)   
        offaxis_filename = os.path.abspath(outdir_average+'/offaxis.fits')
        zodi_filename = os.path.abspath(outdir_average+'/zodicube.fits')
        target_det_outlist = averageDetectorReadout(par,target_outlist,outdir_detector,offaxis = offaxis_filename,factor = pp_fact,zodi=zodi_filename)
    else:
        ref_det_outlist = []
        target_det_outlist = []
        suffix='detector'
        for reffile in ref_outlist:
            ref_det_outlist.append(outdir_detector+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
        for reffile in target_outlist:
            target_det_outlist.append(outdir_detector+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
    times['Construct IFS detector'] = time()

    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*'*60), end=True)
    par.hdr.append(('comment', '*'*22 + ' Scene ' + '*'*20), end=True)
    par.hdr.append(('comment', '*'*60), end=True)    
    par.hdr.append(('comment', ''), end=True)
    try:
        par.hdr.append(('TZODI',t_zodi,'Zodi throughput'),end=True)
        par.hdr.append(('ABSMAG',absmag,'Target star absolute magnitude'),end=True)
    except:
        pass
    
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*'*60), end=True)
    par.hdr.append(('comment', '*'*22 + ' Postprocessing ' + '*'*20), end=True)
    par.hdr.append(('comment', '*'*60), end=True)    
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('PPFACT',pp_fact,'Post-processing factor'),end=True)

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
        Image(data=ref_star_average,header=par.hdr).write(outdir_average+'/average_ref_star_detector.fits',clobber=True)
        Image(data=target_star_average,header=par.hdr).write(outdir_average+'/average_target_star_detector.fits',clobber=True)
        par.hdr.append(('NIMGS',len(target_det_outlist),'Number of time series steps'),end=True)
        par.hdr.append(('TOTTIME',len(target_det_outlist)*par.timeframe,'Total integration time on source'),end=True)

    times['Taking averages'] = time()

    ###################################################################################
    # Step 6: Process the cubes
    ###################################################################################
    img = Image(filename=os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
    ref_cube = reduceIFSMap(par,os.path.abspath(outdir_average+'/average_ref_star_detector.fits'))
    target_cube = reduceIFSMap(par,os.path.abspath(outdir_average+'/average_target_star_detector.fits'))
    offaxis_ideal = reduceIFSMap(par,os.path.abspath(outdir_average+'/offaxis.fits'))
    offaxis_ideal_flipped = reduceIFSMap(par,os.path.abspath(outdir_average+'/offaxis_flipped.fits'))

    ###################################################################################
    # Construct a bowtie mask
    ###################################################################################
    ydim,xdim = target_cube[0].shape
    mask,scratch = bowtie(target_cube[0],ydim//2,xdim//2,openingAngle=65,
                clocking=-par.philens*180./np.pi,IWApix=IWA*lamc/par.lenslet_wav/par.lenslet_sampling,OWApix=OWA*lamc/par.lenslet_wav/par.lenslet_sampling,
                export=None,twomasks=False)    

    times['Process average cubes'] = time()
    ###################################################################################
    # Step 7: Least-squares slice-by-slice subtraction
    ###################################################################################
    if subtract_ref_psf:
#         ref_cube_stack = np.sum(ref_cube.data,axis=0)
#         target_cube_stack = np.sum(target_cube.data,axis=0)
#         ratio = np.sum(target_star_cube[:,0,0]) / np.sum(ref_star_cube[:,0,0])
#         residual = target_cube.data - ratio*ref_cube.data
#         residual[np.isnan(target_cube.data)] = np.NaN
#         residual[(residual>1e10)*(residual<-1e10)] = np.NaN

        # NEED TO subtract the mean of the cube slice by slice before the lstsq step
        
        # residual is target_cube-coefs*ref_cube
        coefs,residual = scale2imgs(target_cube,ref_cube,mask=mask,returndiff=True)
        par.hdr.append(('comment', 'Subtracted scaled mean of reference star PSF'), end=True)
    else:
        residual = target_cube.data
    Image(data=residual,header = par.hdr).write(outdir_average+'/residual.fits', clobber=True)
    Image(data=np.sum(residual,axis=0),header = par.hdr).write(outdir_average+'/residual_stack.fits', clobber=True)

    times['Normalize and subtract reference PSF'] = time()

    ###################################################################################
    # Flatfielding, to account for small errors in wavelength calibration
    ###################################################################################

    flatfield = Image(par.exportDir+'/flatfield_red_optext.fits')
    residual[~np.isnan(residual)] /= flatfield.data[~np.isnan(residual)]
    residual[np.logical_or((residual>1e10),(residual<-1e10))] = np.NaN
    par.hdr.append(('comment', 'Divided by lenslet flatfield'), end=True)
    Image(data=residual,header=par.hdr).write(outdir_average+'/residual_flatfielded.fits',clobber=True)
    Image(data=np.sum(residual,axis=0),header=par.hdr).write(outdir_average+'/residual_flatfielded_stack.fits',clobber=True)
    
    ###################################################################################
    # Construct a matched filter
    ###################################################################################
    
    
    
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
#     noiseleft = np.sqrt(2*mf_npix)*pixstd # twice the num of pix since we subtract the off field
    
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

    return signal-off,noiseright,noiseleft
    

def SNR_spectrum(lam_midpts,signal, noise, 
                lam_contrast=None, plot=True,outname = 'SNR.png', outfolder = '',title='Planet+star',edges=1,FWHM=2):
    #lam_midpts,junk = calculateWaveList(par)
    # wavelist = np.arange(min(lam_midpts),max(lam_midpts),3)
    #wavelist = 770*np.linspace(1.-0.18/2.,1.+0.18/2.,45)
    if lam_contrast is not None:
        lams=lam_contrast
    else:
        lams=lam_midpts
    real_vals=calc_contrast_Bijan(lams)
    smooth = ndimage.filters.gaussian_filter1d(real_vals,FWHM/2.35,order=0,mode='constant')
    smoothfunc=interp1d(lams,smooth)

    chisq = np.sum((signal[edges:-edges]*np.mean(real_vals)/np.mean(signal) - smoothfunc(lam_midpts[edges:-edges]))**2/(noise[edges:-edges]*np.mean(real_vals)/np.mean(signal))**2)

    if plot:
        sns.set_style("whitegrid")
        fig,ax = plt.subplots(figsize=(12,6))
        ax.plot(lams,real_vals,label='Original spectrum')
        ax.errorbar(lam_midpts,signal*np.mean(real_vals)/np.mean(signal),yerr=noise*np.mean(real_vals)/np.mean(signal),label='Recovered spectrum',fmt='o')    
        ax.plot(lams,smooth,'-',label='Gaussian-smoothed original spectrum w/ FWHM=%.0f bins' % FWHM)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Contrast')
        ax.set_title(title+', chisq='+str(chisq/len(signal[edges:-edges])))
        plt.legend()
        fig.savefig(outfolder+outname)
    return (chisq/len(signal[edges:-edges]))


def mf(cube,mask,threshold):
    '''
    Matched filter function
    
    Parameters
    ----------
    cube: 3D ndarray
        An IFS cube from which to compute the matched filter
    mask: 2D ndarray
        This is typically the coronagraphic mask
    threshold: float
        Value below which we crop the matched filter
    
    Returns
    -------
    matched_filter: 3D ndarray
        Matched filter with the same dimensions as input cube
    
    '''
    matched_filter = np.zeros(cube.data.shape)
    
    for slicenum in range(cube.data.shape[0]):
        nanmask = np.isnan(cube.data[slicenum])
        cube_norm = cube.data[slicenum]/np.nansum(cube.data[slicenum])
        this_slice = cube_norm/np.nansum((cube_norm)**2)
        # calculate correction factor since we are going to crop only the top the of the hat
        # this is the inverse ratio of the contribution of the brightest pixels over the rest
        msk = mask*(this_slice>threshold)
        aper_phot = np.nansum(this_slice)/np.nansum(this_slice[msk])
        # Set all low-contributing pixels and pixels outside of mask to 0.0
        this_slice[~msk] = 0.0
        this_slice[nanmask] = np.NaN  # put NaNs where they belong
        matched_filter[slicenum,:,:] = this_slice
        # Multiply what is left by that aperture correction factor
        matched_filter[slicenum,:,:]*=aper_phot
    return matched_filter


def recenter_offaxis(offaxis_file,threshold,outname='centered_offaxis.fits'):

    '''
    Example: recenter_offaxis('/Users/mrizzo/IFS/OS5/offaxis/spc_offaxis_psf.fits',0.01,par.exportDir+'/centered_offaxis.fits')

    '''
    offaxis = Image(offaxis_file)
    offsetpx = offaxis.header['OFFSET']/offaxis.header['PIXSIZE']
    centered_offaxis = Image(data=shiftCube(offaxis.data,dx=-offsetpx,dy=0))
    maxi = np.nanmax(np.nanmax(centered_offaxis.data,axis=2),axis=1)
    total = np.nansum(np.nansum(centered_offaxis.data,axis=2),axis=1)
    centered_offaxis.data/=maxi[:,np.newaxis,np.newaxis]

    centered_offaxis.data[centered_offaxis.data<threshold] = 0.0
    newsum = np.nansum(np.nansum(centered_offaxis.data,axis=2),axis=1)
    centered_offaxis.data *= maxi[:,np.newaxis,np.newaxis]#total[:,np.newaxis,np.newaxis]/newsum[:,np.newaxis,np.newaxis]
    outkey = fits.HDUList(fits.PrimaryHDU(centered_offaxis.data,offaxis.header))
    outkey.writeto(outname,clobber=True)
    return outkey

def construct_mflib_old(par,psf,IWA,OWA,lamc,threshold):
    '''
    Construct a library of matched filters for all points within the bowtie mask
    For now, this uses the already-reduced, ideal offaxis psf already in cube space.
    We could also build a function that offsets the original, before transformation.
    '''
    mflib = np.zeros(list(psf.data[0].shape)+list(psf.data.shape))
    ydim,xdim = psf.data[0].shape
    mask,junk = bowtie(psf.data[0],ydim//2,xdim//2,openingAngle=65,
                clocking=-par.philens*180./np.pi,IWApix=IWA*lamc/par.lenslet_wav/par.lenslet_sampling,OWApix=OWA*lamc/par.lenslet_wav/par.lenslet_sampling,
                export=None,twomasks=False)

    ic = mflib.shape[0]//2 # i or x axis is horizontal
    jc = mflib.shape[1]//2 # j or y axis is vertical
    for i in range(mflib.shape[0]):
        for j in range(mflib.shape[1]):
            if ~np.isnan(mask[i,j]):
                decentered = Image(data=shiftCube(psf.data,dx=i-ic,dy=j-jc))
                mflib[i,j] = mf(decentered,mask,threshold)


def construct_mflib(par,psf,threshold,mask=None,IWA=None,OWA=None,lamc=None,trim=30,outname = None):
    '''
    Construct a library of matched filters for all points within the bowtie mask
    For now, this uses the already-reduced, ideal offaxis psf already in cube space.
    We could also build a function that offsets the original, before transformation.
    
    This particular function saves memory and time by only recording the relevant pixels
    '''
    if mask is None:
        ydim,xdim = psf.data[0].shape
        mask,junk = bowtie(psf.data[0],ydim//2,xdim//2,openingAngle=65,
                    clocking=-par.philens*180./np.pi,IWApix=IWA*lamc/par.lenslet_wav/par.lenslet_sampling,OWApix=OWA*lamc/par.lenslet_wav/par.lenslet_sampling,
                    export=None,twomasks=False)
    x = np.arange(mask.shape[1])
    y = np.arange(mask.shape[0])
    x,y = np.meshgrid(x,y)
    xlist= x[mask]
    ylist= y[mask]
    psftrim = psf.data[:,trim:-trim,trim:-trim]
    masktrim = mask[trim:-trim,trim:-trim]
    mflib = np.zeros(list(xlist.shape)+list(psftrim.shape))
    
    ic = mask.shape[0]//2 # i or x axis is horizontal
    jc = mask.shape[1]//2 # j or y axis is vertical
    for ii in range(len(xlist)):
        i = xlist[ii]
        j = ylist[ii]
        decentered = Image(data=shiftCube(psftrim,dx=i-ic,dy=j-jc))
        mflib[ii] = mf(decentered,masktrim,threshold)
        
    outkey = fits.HDUList(fits.PrimaryHDU(mflib))
    outkey.append(fits.PrimaryHDU(mask.astype(np.int)))
    outkey.append(fits.PrimaryHDU(xlist.astype(np.int)))
    outkey.append(fits.PrimaryHDU(ylist.astype(np.int)))
    if outname is None:
        outkey.writeto(par.exportDir+'/mflib.fits.gz',clobber=True)
    else:
        outkey.writeto(outname,clobber=True)
        


def convolved_mf(incube, mflibname,trim=30,):
    
    '''
    Generates a pseudo-convolution of the image by the matched filter library
    '''
    mflibHDUlist = fits.open(mflibname)
    mflib = mflibHDUlist[0].data
    mask = mflibHDUlist[1].data
    xlist = mflibHDUlist[2].data
    ylist = mflibHDUlist[3].data
    
    convolvedmap = np.zeros(incube.shape)
    for i in range(len(xlist)):
        ix = xlist[i]
        iy = ylist[i]
        convolvedmap[:,iy,ix] = np.nansum(np.nansum(incube[:,trim:-trim,trim:-trim] * mflib[i],axis=2),axis=1)
    
    return convolvedmap


def RDI_noise(par,xshift,yshift,order=3,
                rootname = "mflib",
                outdir_time_series = 'OS5',
                outdir_detector='OS5/OS5_detector',
                outdir_average='OS5/OS5_average',
                process_cubes=True,
                countershift=True,
                normalize_contrast=True,
                psf_time_series_folder='/Users/mrizzo/IFS/OS5/with_lowfc',Nref=30,
                offaxis_psf_filename='/Users/mrizzo/IFS/OS5/offaxis/spc_offaxis_psf.fits',
                ref_star_T=9377*u.K, ref_star_Vmag=2.37,
                target_star_T=5887*u.K, target_star_Vmag=5.03,
                nonoise=True,IWA=3,OWA=9,lamc=770.,BW = 0.18,
                tel_pupil_area=3.650265060424805*u.m**2,
                mflib='/mflib.fits.gz'):
    
    '''
    
    Here we assume the target PSF time series is already in hand, and already in outdir_time_series.
    '''
    
    ###################################################################################
    # Load, shift and propagate all of the IFS images for the reference star
    ###################################################################################
    
    # load the filenames
    filelist = glob.glob(psf_time_series_folder+'/*')
    filelist.sort()
    
    
    # load first filelist to get its shape
    kristfile = Image(filename=filelist[0])
    fileshape = kristfile.data.shape
    adjust_krist_header(kristfile,lamc=lamc)
    
    Nlam = fileshape[0]
    lamlist = lamc*np.linspace(1.-BW/2.,1.+BW/2.,Nlam)*u.nm

    # reference and target star cube conversions
    ref_star_cube = convert_krist_cube(fileshape,lamlist,ref_star_T,ref_star_Vmag,tel_pupil_area)
    target_star_cube = convert_krist_cube(fileshape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)

    input_sampling = kristfile.header['PIXSIZE']
    input_wav = kristfile.header['LAM_C']*1000.
    par.pixperlenslet = par.lenslet_sampling/(input_sampling * input_wav/par.lenslet_wav)
    par.hdr['XSHIFT']=xshift*par.pixperlenslet
    par.hdr['YSHIFT']=yshift*par.pixperlenslet
    
    try:
        target = Image(par.exportDir+"/target_average_detector_red_optext.fits")
        # limit only to the reference files
        filelist = filelist[:Nref]
    except:
        # also compute the target files
        filelist = filelist

    ###################################################################################
    # simulate the IFS flux at the detector plane (no losses)
    ###################################################################################
    ref_outlist = []
    target_outlist = []
    for i in range(len(filelist)):
        reffile = filelist[i]
        if process_cubes:
            log.info('Processing file '+reffile.split('/')[-1])
            cube = fits.open(reffile)[0]
            cube.data*=ref_star_cube
            if i<Nref:
                cube.data*=ref_star_cube
            else:
                cube.data*=target_star_cube
        
            # adjust headers for slightly different wavelength
            log.debug('Modifying cube header')
            adjust_krist_header(cube,lamc=lamc)
            par.saveDetector=False  
            
            # shift the cube
#             cube.data = shiftCube(cube.data,dx=xshift*par.pixperlenslet,dy=yshift*par.pixperlenslet,order=1)
            if i<Nref:
            cube.data = ndimage.interpolation.shift(cube.data,
                             [0.0,yshift*par.pixperlenslet,xshift*par.pixperlenslet],order=order)
            detectorFrame = polychromeIFS(par,lamlist.value,cube,QE=True)

            if i<Nref:
                Image(data = detectorFrame,header=par.hdr).write(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits',clobber=True)
                ref_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits')
            else:
                Image(data = detectorFrame,header=par.hdr).write(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits',clobber=True)
                target_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits')

        else:
            ref_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits')
        else:

            if i<Nref:
                ref_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits')
            else:
                target_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits')
    
    if normalize_contrast:
        offaxiscube = Image(offaxis_psf_filename)
        print('Processing file '+offaxis_psf_filename)

        # Need to re-center the off-axis psf if it is not the right size
        if offaxiscube.data.shape[1] < fileshape[1]:
            diff = fileshape[1]-offaxiscube.data.shape[1]
            offaxiscube_recentered = np.zeros(fileshape)
            offaxiscube_recentered[:,diff//2:-diff//2,diff//2:-diff//2] += offaxiscube.data
            offaxiscube = Image(data=offaxiscube_recentered,header = offaxiscube.header)
        offaxis_star_cube = convert_krist_cube(offaxiscube.data.shape,lamlist,target_star_T,target_star_Vmag,tel_pupil_area)

        # adjust headers for slightly different wavelength
        log.debug('Modifying cube header')
        adjust_krist_header(offaxiscube,lamc=lamc)
        par.saveDetector=False
        Image(data=offaxiscube.data,header=offaxiscube.header).write(outdir_average+'/offaxiscube_processed.fits',clobber=True)
        detectorFrame = polychromeIFS(par,lamlist.value,offaxiscube,QE=True)
        Image(data = detectorFrame,header=par.hdr).write(outdir_average+'/offaxis.fits',clobber=True)
        offaxis_reduced = reduceIFSMap(par,outdir_average+"/offaxis.fits")


    ###################################################################################
    # simulate the detector readout (including optical losses)
    ###################################################################################
    par.nonoise=nonoise
    ref_det_outlist = averageDetectorReadout(par,ref_outlist,outdir_detector)  
    if len(target_outlist)>0:
        target_det_outlist = averageDetectorReadout(par,target_outlist,outdir_detector)  
        
    
    ###################################################################################
    # Average all detector images and reduce IFS map
    ###################################################################################
    ave_ref = np.zeros(fits.open(ref_det_outlist[0])[1].data.shape)
    for targetfile in ref_det_outlist:
        ave_ref += fits.open(targetfile)[1].data
    ref = Image(data = ave_ref)
    ref.write(outdir_average+"/ref_average_detector_"+rootname+".fits",clobber=True)
    ref_reduced = reduceIFSMap(par,outdir_average+"/ref_average_detector_"+rootname+".fits")
    
    if len(target_outlist)>0:
        target_star_average = np.zeros(Image(filename=target_det_outlist[0]).data.shape)
        for reffile in target_det_outlist:
            target_star_average += Image(filename=reffile).data
        target = Image(data=target_star_average,header=par.hdr)
        # write it out for the future iterations of this program
        target.write(outdir_average+'/average_target_star_detector.fits',clobber=True)

    
    # ref_reduced is now the IFS cube from the shifted reference star
    # target is the IFS cube from the target star

    ###################################################################################
    # Counter-shift the reference cube
    ###################################################################################
    if countershift:
        c = np.cos(par.philens)
        s = np.sin(par.philens)
        ref_reduced.data[np.isnan(ref_reduced.data)] = 0.0
        ref_reduced.data = ndimage.interpolation.shift(ref_reduced.data,
                                    [0.0,-yshift*c+xshift*s,-xshift*c-yshift*s],order=order)
        ref_reduced.write(par.exportDir+"/ref_average_detector_countershifted_"+rootname+"_red_optext.fits",clobber=True)
    
    ###################################################################################
    # Do basic least squares RDI, slice by slice; no mean subtraction for now
    ###################################################################################
    
    ydim,xdim = target.data[0].shape
    mask,scratch = bowtie(target.data[0],ydim//2,xdim//2,openingAngle=65,
            clocking=-par.philens*180./np.pi,
            IWApix=IWA*lamc/par.lenslet_wav/par.lenslet_sampling,
            OWApix=OWA*lamc/par.lenslet_wav/par.lenslet_sampling,
            export=None,twomasks=False)    
    coefs,residual = scale2imgs(target,ref_reduced,mask=mask,returndiff = True)
    Image(data=residual).write(par.exportDir+"/lstsq_residual_"+rootname+".fits")
    
    ###################################################################################
    # Convolve with matched filter
    ###################################################################################
    if mflib is None:
        recenter_offaxis(offaxis_psf_filename,0.01,par.exportDir+'/centered_offaxis.fits')
        centered_offaxis_file = Image(par.exportDir+'/centered_offaxis.fits')
        adjust_krist_header(centered_offaxis_file,lamc=lamc)
        par.saveDetector=False  
        Nlam = centered_offaxis_file.data.shape[0]
        lamlist = lamc*np.linspace(1.-BW/2.,1.+BW/2.,Nlam)

        # propagate the offaxis PSF
        detectorFrame = polychromeIFS(par,lamlist,centered_offaxis_file,QE=True)
        Image(data = detectorFrame,header=par.hdr).write(par.exportDir+"/offaxis_detector.fits",clobber=True)

        # reduce the offaxis PSF
        offaxis_reduced = reduceIFSMap(par,par.exportDir+"/offaxis_detector.fits")

        # construct matched filter
        psf = Image(par.exportDir+"/offaxis_detector_red_optext.fits")
        construct_mflib(par,psf,threshold=0.5,mask=mask)
    
    # this will return an error if the library is not found
    convolved = convolved_mf(residual,par.exportDir+mflib)
    
    # this computes the convolution with an offaxis source as bright as the star
    if normalize_contrast:
        starmf = convolved_mf(offaxis,par.exportDir+mflib)
        max_starmf = np.amax(np.amax(starmf,axis=2),axis=1)
    else:
        max_starmf = np.ones(starmf.shape[0])
    
    convolved /= max_starmf[:,np.newaxis,np.newaxis]
    
    outkey = fits.HDUList(fits.PrimaryHDU(convolved.astype(np.float)))
    outkey.writeto(par.exportDir+'/mflib'+rootname+'.fits',clobber=True)
    pixstd = [np.nanstd(convolved[i]) for i in range(residual.shape[0])]
    return pixstd