import numpy as np
import astropy.units as u
import astropy.constants as c
from tools.inputScene import convert_krist_cube,calc_contrast
import glob
from IFS import propagateIFS,reduceIFSMap
from tools.initLogger import getLogger
log = getLogger('crispy')
from tools.image import Image
try:
    from astropy.io import fits
except:
    import pyfits as fits
from time import time
import os
from tools.detector import averageDetectorReadout


def process_SPC_IFS(par,psf_time_series_folder,offaxis_psf_filename,
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
	# this should be parallelized
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
		offaxiscube = fits.open(offaxis_psf_filename)[0]
		print('Processing file '+offaxis_psf_filename)

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
		log.debug('Modifying cube header')
		if offaxiscube.header['LAM_C']==0.8:
			if lamc==770.:
				offaxiscube.header['LAM_C']=0.77
				# by NOT changing the pixelsize, we implicitly assume that the PSF is the same at 770 then at 800
			else:
				offaxiscube.header['LAM_C']=lamc/1000.
				offaxiscube.header['PIXSIZE']*=lamc/0.77
		else:
			offaxiscube.header['LAM_C']=lamc/1000.
			offaxiscube.header['PIXSIZE']*=lamc/0.77
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
	# Step 4: Add the off-axis PSF before reading on the detector
	###################################################################################

	if process_detector:
		# Apply detector for both reference star and target
		ref_det_outlist = averageDetectorReadout(par,ref_outlist,outdir_detector)	
		contrast=calc_contrast(lamlist.value,mean_contrast=mean_contrast)
		offaxis_filename = os.path.abspath(outdir_average+'/offaxis.fits')
		target_det_outlist = averageDetectorReadout(par,target_outlist,outdir_detector,offaxis = offaxis_filename)
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
		ref_star_average /= len(ref_det_outlist)
		for reffile in target_det_outlist:
			target_star_average += Image(filename=reffile).data
		target_star_average /= len(target_det_outlist)
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
	
	# loop over all the slices in the cube:
	matched_filter = np.zeros(residual.shape)
	matched_filter_flipped = np.zeros(residual.shape)
	signal = np.zeros(residual.shape[0])
	noise = np.zeros(residual.shape[0])
	for slicenum in range(residual.shape[0]):
		
		offaxis_ideal_norm = offaxis_ideal.data[slicenum]/np.nansum(offaxis_ideal.data[slicenum])
		matched_filter[slicenum,:,:] = offaxis_ideal_norm/np.nansum((offaxis_ideal_norm)**2)
		signal[slicenum] = np.nansum(matched_filter[slicenum,:,:]*residual[slicenum,:,:])
		offaxis_ideal_flipped_norm = offaxis_ideal_flipped.data[slicenum]/np.nansum(offaxis_ideal_flipped.data[slicenum])
		matched_filter_flipped[slicenum,:,:] = offaxis_ideal_flipped_norm/np.nansum((offaxis_ideal_flipped_norm)**2)
		noise[slicenum] = np.nansum(matched_filter_flipped[slicenum,:,:]*residual[slicenum,:,:])
	
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

	return signal,noise
	


