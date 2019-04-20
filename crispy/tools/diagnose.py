from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from scipy import interpolate

def gauss(x, a, x0, sig,b,c):
    '''
    Simple gaussian function with usual inputs
    '''
    return c+b*x+a*np.exp(-(x-x0)**2/(2.*sig**2))
    

def fit_monochromatic_cube( cube,
                            lamlist,
                            returnAll = False,
                            sigma_guess = 5):
    '''
    Fits an extracted data cube with a gaussian to find the wavelength peak
    
    Parameters
    ----------
    cube: 3D ndarray
        The extracted datacube where all bad pixels are NaNs
    lamlist: 1D array
        List of wavelengths corresponding to the slices of the cube
        Suggested units: nanometers (in which sigma_guess is about 5)
    returnAll: boolean
        If True, return the full results of the curve fit function (popt,pcov)
        If False, return only the central wavelength (Default)
    sigma_guess: float
        Guess at the width of the gaussian fit in same units as lamlist (Default 5)
    '''
    vals = np.nanmean(cube,axis=(1,2))
    popt, pcov = curve_fit( gauss,
                            lamlist,
                            vals,
                            p0=[np.amax(vals),lamlist[np.argmax(vals)],sigma_guess,0,0]
                            )
    if returnAll: return popt,pcov
    else: return popt[1]


    ## to reconstruct gaussian function, use;
    # gauss(lamlist,*popt)

def plotfit(cube,lamlist):
    vals = np.nanmean(cube,axis=(1,2))
    popt, pcov = fit_monochromatic_cube(cube,lamlist,returnAll=True)
    curve_vals = gauss(lamlist, *popt)
    print popt[1]
    plt.figure(figsize=(10,10))
    plt.plot(lamlist,curve_vals,label='Fitted')
    plt.plot(lamlist,vals,label='Data')
    plt.legend()
