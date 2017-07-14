import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.analytic_functions as af
try:
    from astropy.io import fits
except:
    import pyfits as fits
from scipy.interpolate import interp1d

def adjust_krist_header(cube,lamc,pixsize=None):
    '''
    Force a different central wavelength, assuming that everything is wavelength-independent
    
    Parameters
    ----------
    
    cube: 3D float array
        Input cube from J. Krist
    lamc: float
        Central wavelength to override in nm
    pixsize: float
        Pixel scale at central wavelength to override (in lambda/D)
    
    '''
    oldlam = cube.header['LAM_C']
    cube.header['LAM_C'] = lamc/1000.
    cube.header['LAM_MIN'] *= lamc/1000./oldlam
    cube.header['LAM_MAX'] *= lamc/1000./oldlam
    
    if pixsize is not None:
        cube.header['PIXSIZE'] = pixsize


def convert_krist_cube(cubeshape,lamlist,star_T,star_Vmag,tel_area):
    '''
    This function calculates the number of photons per second
    entering the WFIRST obscured aperture, given the star Vmag and its temperature
    for each slice of the input cube
    This only works with John Krist's cubes, 
    '''
    
    
    # We need to determine the coefficient of proportionality between a blackbody source and the
    # actualy flux received (depends on size of the star, distance, etc)
    # define Vband
    lambda_cent = 550*u.nanometer

    # this is the flux density per steradian (specific intensity) you would expect from Vband
    flux_bb_F550 = af.blackbody_lambda(lambda_cent, star_T).to(u.Watt/u.m**2/u.um/u.sr)

    # this is the actual flux density received in Vband
    Vband_zero_pt = (3953*u.Jansky).to(u.Watt/u.m**2/u.Hertz)
    Vband_zero_pt *= (c.c/lambda_cent**2)
    flux_star_Vband = Vband_zero_pt*10**(-0.4*star_Vmag)

    # the ratio is coefficient we seek; this will multiply a blackbody function to yield flux densities
    # at all wavelengths
    ratio_star = (flux_star_Vband/flux_bb_F550)

    # this is the ratio which we want to multiply phot_Uma_Vband for the other bands
    #print("Ratio of blackbodies is %f" % ratio_Uma)

    # Now convert each slice to photons per second per square meter
    dlam = lamlist[1]-lamlist[0]
    newcube=np.zeros(cubeshape)
    for i in range(len(lamlist)):
        E_ph = (c.h*c.c/lamlist[i]).to(u.J) # photon energy at middle frequency
        BBlam = af.blackbody_lambda(lamlist[i], star_T).to(u.Watt/u.m**2/u.nm/u.sr)
        flux = (BBlam*ratio_star).to(u.W/u.m**2/u.nm) # this is Watts per m2 per nm
        photon_flux = flux/E_ph # This is in Photons per second per m2 per nm
        newcube[i,:,:] += photon_flux.to(1./u.s/u.m**2/u.nm) # add value to entire cube since we will multiply Krist's cube
    
    # multiply by the number of wavelengths since this is the way J. Krist normalizes his cubes
    newcube *= tel_area*len(lamlist)
    # note that this is still per unit nanometer of bandwidth, so it still needs to be multipled by dlam in the IFS.
    return newcube


def zodi_cube(krist_cube,area_per_pixel,absmag,Vstarmag,zodi_surfmag,exozodi_surfmag,distAU,t_zodi):
    cube = krist_cube.copy()
    cube /= cube.shape[0]#*cube.shape[1]*cube.shape[2] # now we consider the cube as a real spectral datacube instead of a multiplication cube
    # each pixel now represents the number of photons per pixel per slice if it was uniform across the field
    Msun = 4.83
    zodicube = cube* t_zodi*area_per_pixel*10**(-0.4*(zodi_surfmag-Vstarmag)) # where area_per_pixel has to be in square arcsec
    exozodicube = cube* t_zodi*area_per_pixel*10**(-0.4*(exozodi_surfmag+absmag-Msun-Vstarmag))/distAU**2 # where area_per_pixel has to be in square arcsec
    return zodicube+exozodicube
    


# def calc_contrast(wavelist,star_T=6000*u.K,planet_type='Jupiter',abundance=1,distance = 5,phase=90,mean_contrast = 1e-8,
#                     folder= '/Users/mrizzo/Science/Haystacks/Cahoy_Spectra/albedo_spectra/'):
#     '''
#     Calculates the contrast curve for a list of wavelengths
#     '''
#     # load corresponding file
#     
#     filename = folder+planet_type+'_'+str(abundance)+'x_'+str(distance)+'AU_'+str(phase)+'deg.dat'
#     spectrum = np.loadtxt(filename)
#     spec_func = interp1d(spectrum[:,0]*1000.,spectrum[:,1])
#     vals = spec_func(wavelist)
#     vals /= np.mean(vals)
#     vals *= mean_contrast
#     return vals 

def calc_contrast_Bijan(wavelist,
    # default values are for 47 Uma c
    albedo = 0.28, # in the continuum; use albedo=0 to use native albedo files from Cahoy et al
    radius = 1.27, # in Jupiter radius
    dist = 3.6, # in AU - note that it is different from the distance keyword below because there simply isn't a Cahoy spectrum for 3.6AU
    # this is just to load some spectrum
    planet_type='Jupiter',
    abundance=1,
    distance = 5,
    phase=90, 
    folder= '/Users/mrizzo/Science/Haystacks/Cahoy_Spectra/albedo_spectra/'):
    
    if folder is None:
        vals = np.ones(len(wavelist))
    else:
        filename = folder+planet_type+'_'+str(abundance)+'x_'+str(distance)+'AU_'+str(phase)+'deg.dat'
        spectrum = np.loadtxt(filename)
        spec_func = interp1d(spectrum[:,0]*1000.,spectrum[:,1])
        vals = spec_func(wavelist)
        if albedo!=0:
            vals /= np.amax(vals)
    if albedo!=0:
        vals*=albedo
    vals *= (radius*c.R_jup.to(u.m)/(dist*u.AU).to(u.m))**2
    return vals
            
def calc_contrast(wavelist,distance,radius,filename,albedo=None):

    spectrum = np.loadtxt(filename)
    spec_func = interp1d(spectrum[:,0]*1000.,spectrum[:,1])
    vals = spec_func(wavelist)
    if albedo!=None:
        vals /= np.amax(vals)
        vals *= albedo
    
    vals *= (radius*c.R_jup.to(u.m)/(distance*u.AU).to(u.m))**2
    return vals