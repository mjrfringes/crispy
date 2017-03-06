import numpy as np
import astropy.units as u
from tools.inputScene import convert_krist_cube
import glob
from IFS import propagateIFS
import logging
from tools.image import Image
from astropy.io import fits

log = logging.getLogger('main')


def process_SPC_IFS(par,psf_time_series_folder,offaxis_psf_filename,contrast,
					ref_star_T=9377*u.K, ref_star_Vmag=2.37, target_star_T=5887*u.K, target_star_Vmag=5.03,
					lamc=770.,BW=0.18,Nlam=45,n_ref_star_imgs=30,tel_pupil_area=3.650265060424805*u.m**2,
					outdir_time_series = 'OS5',outdir_detector='OS5/OS5_detector',outdir_average='OS5/OS5_average'):
	'''
	Process SPC PSF cubes from J. Krist through the IFS


	'''
	
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

	###################################################################################
	# Step 2: Process all the cubes and directly apply detector
	###################################################################################
	ref_outlist = []
	target_outlist = []
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
		detectorFrame = propagateIFS(par,lamlist.value/1000.,cube)

		if i<n_ref_star_imgs:
			Image(data = detectorFrame,header=par.hdr).write(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits',clobber=True)
			ref_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_refstar_IFS.fits')
		else:
			Image(data = detectorFrame,header=par.hdr).write(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits',clobber=True)
			target_outlist.append(outdir_time_series+'/'+reffile.split('/')[-1].split('.')[0]+'_targetstar_IFS.fits')
			
	###################################################################################
	# Step 3: Process the off-axis PSF in the same way
	###################################################################################
	offaxiscube = fits.open(offaxis_psf_filename)[0]
	print('Processing file '+offaxis_psf_filename)
	offaxiscube.data*=target_star_cube
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
	Image(data = detectorFrame,header=par.hdr).write(outdir_time_series+'/offaxis.fits',clobber=True)

	###################################################################################
	# Step 4: Add the off-axis PSF before reading on the detector
	###################################################################################

	# Apply detector for both reference star and target
	ref_det_outlist = averageDetectorReadout(par,ref_outlist,outdir_detector)	
	target_det_outlist = averageDetectorReadout(par,target_outlist,outdir_detector,offaxis = outdir_time_series+'/offaxis.fits',contrast=contrast)

	###################################################################################
	# Step 5: Take averages
	###################################################################################
	ref_star_average = np.zeros(Image(filename=ref_det_outlist[0]).data.shape)
	target_star_average = np.zeros(Image(filename=target_det_outlist[0]).data.shape)
	for reffile in ref_det_outlist:
		ref_star_average += Image(filename=reffile).data
	ref_star_average /= len(ref_det_outlist)
	for reffile in target_det_outlist:
		target_star_average += Image(filename=reffile).data
	target_star_average /= len(target_det_outlist)
	Image(data=ref_star_average,header=par.hdr,extraheader=ref_det_outlist[0].header).write(outdir_average+'/average_ref_star_detector.fits',clobber=True)
	Image(data=target_star_average,header=par.hdr,extraheader=target_det_outlist[0].header).write(outdir_average+'/average_target_star_detector.fits',clobber=True)

	###################################################################################
	# Step 6: Process the cubes
	###################################################################################
	ref_cube = reduceIFSMap(par,outdir_average+'/average_ref_star_detector.fits')
	target_cube = reduceIFSMap(par,outdir_average+'/average_target_star_detector.fits')

	###################################################################################
	# Step 7: Naive PSF subtraction; match the flux using the Vmag difference
	###################################################################################
	ref_cube_noave = ref_cube.data -np.nanmean(ref_cube.data)
	target_cube_noave = target_cube.data - np.nanmean(target_cube.data)
	residual = target_cube_noave - 10**(0.4*(ref_star_Vmag-target_star_Vmag))*ref_cube_noave
	Image(data=residual).write(outdir_average+'/residual.fits',clobber=True)

	
	###################################################################################
	# Step 8: Construct an off-axis PSF matched filter and propagate it through the IFS
	###################################################################################
	#offaxiscube_norm = offaxiscube.data[0]/np.nansum(offaxiscube.data[0])
	#matched_filter = offaxiscube_norm/np.nansum((offaxis_ideal_norm)**2)
	#Image(data=matched_filter).write('OS5/OS5_average/matched_filter.fits',clobber=True)




