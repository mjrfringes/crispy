import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.analytic_functions as af
from astropy.io import fits





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
    flux_bb_F550 = af.blackbody_nu(lambda_cent, star_T).to(u.Watt/u.m**2/u.Hertz/u.sr)

    # this is the actual flux density received in Vband
    Vband_zero_pt = (3953*u.Jansky).to(u.Watt/u.m**2/u.Hertz)
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
        high = c.c/(lamlist[i]-dlam/2.)
        low = c.c/(lamlist[i]+dlam/2.)
        mid = 0.5*(low+high) # middle frequency
        dnu = high-low
        E_ph = (c.h*mid).to(u.J) # photon energy at middle frequency
        BBlam = af.blackbody_nu(lamlist[i], star_T).to(u.Watt/u.m**2/u.Hertz/u.sr)
        flux = (BBlam*ratio_star*dnu).to(u.W/u.m**2) # this is Watts per m2
        photon_flux = flux/E_ph # This is in Photons per second per m2
        newcube[i,:,:] += photon_flux.to(1./u.s/u.m**2)
    
	# multiply by the number of wavelengths since this is the way J. Krist normalizes the cubes
    newcube *= len(lamlist)*tel_area
    return newcube


