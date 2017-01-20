from astropy.io import fits as pyf
import numpy as np
import logging as log
from scipy.interpolate import interp1d
from photutils import RectangularAperture
from photutils import aperture_photometry
from scipy import ndimage

def simpleReduction(par,name,ifsimage):
    '''
    Basic cube reduction using an IFS image and a wavecal cube
    Equivalent to method 1 in the IDL primitive 'pisces_assemble_spectral_datacube'
    '''
    log.info('Simple reduction')
    calCube = pyf.open(par.wavecalDir+par.wavecalName)
    
    waveCalArray = calCube[0].data
    waveCalArray = waveCalArray/1000.
    
    xcenter = calCube[1].data
    nlens = xcenter.shape[1]
    ycenter = calCube[2].data
    good = calCube[3].data
    ydim,xdim = ifsimage.shape
    xdim-=2
    xcenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan
    ycenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan

    
    sdpx = 26 # length of longest spectra in pixels; need to link this to the filter setting
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
    
    wavelengths = np.arange(sdpx+1)*(lam_long-lam_short)/float(sdpx)+lam_short
    
    cube = np.zeros((sdpx,nlens,nlens))

    X = np.zeros((nlens,nlens),dtype='i4')
    Y = np.zeros((nlens,nlens),dtype='i4')
    #X *= np.NaN
    #Y *= np.NaN
    
    for wav in range(sdpx):
        lam = wavelengths[wav]
        log.info('Wavelength = %3.1f' % (lam*1000.))
        #index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-lam))
        #print index
        for i in range(nlens):
            for j in range(nlens):
                if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
                    #print xcenter[:,i,j],np.isnan(xcenter[:,i,j]).any()
                    #print np.isnan(xcenter[0,i,j])
                    fx = interp1d(waveCalArray,xcenter[:,i,j])
                    fy = interp1d(waveCalArray,ycenter[:,i,j])
                    #print (lam,fx(lam),xcenter[:,i,j])
                    Y[i,j] = np.int(fx(lam))
                    X[i,j] = np.int(fy(lam))            
        cube[wav,:,:] = ifsimage[X,Y]+ifsimage[X,Y+1]+ \
            ifsimage[X,Y-1]+ifsimage[X,Y+2]+ifsimage[X,Y-2]
    
    cube[cube==0] = np.NaN
    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)


def densifiedSimpleReduction(par,name,ifsimage,ratio=10.):
    '''
    Basic cube reduction using an IFS image and a wavecal cube
    Equivalent to method 1 in the IDL primitive 'pisces_assemble_spectral_datacube'
    '''
    calCube = pyf.open(par.wavecalDir+par.wavecalName)
    
    nx = int(ifsimage.shape[0] * ratio)
    ny = int(ifsimage.shape[1] * ratio)
    
    x = np.arange(nx)/ratio 
    y = np.arange(ny)/ratio

    x, y = np.meshgrid(x, y)
    
    log.info('Densifying image by a factor %d' % ratio)
    ifsimageDense = ndimage.map_coordinates(ifsimage, [y, x],cval=np.nan)
    ifsimageDense /= ratio**2 # conserve energy

    
    waveCalArray = calCube[0].data
    waveCalArray = waveCalArray/1000.
    
    xcenter = calCube[1].data
    nlens = xcenter.shape[1]
    ycenter = calCube[2].data
    good = calCube[3].data
    #xcenter[~good] = np.NaN
    #ycenter[~good] = np.NaN
    ydim,xdim = ifsimageDense.shape
    xmargin = (ratio*5)//2
    ymargin = (ratio)//2
    xdim-=xmargin
    ydim-=ymargin
    xcenter[~((xcenter*ratio<xdim)*(ycenter*ratio<ydim)*(xcenter*ratio>xmargin)*(ycenter*ratio>ymargin))] = np.nan
    ycenter[~((xcenter*ratio<xdim)*(ycenter*ratio<ydim)*(xcenter*ratio>xmargin)*(ycenter*ratio>ymargin))] = np.nan

    
    sdpx = 26 # length of longest spectra in pixels; need to link this to the filter setting
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
    
    wavelengths = np.arange(sdpx+1)*(lam_long-lam_short)/float(sdpx)+lam_short
    
    cube = np.zeros((sdpx,nlens,nlens))

    X = np.zeros((nlens,nlens),dtype='i4')
    Y = np.zeros((nlens,nlens),dtype='i4')
    #X *= np.NaN
    #Y *= np.NaN
    ratio = int(ratio)
    for wav in range(sdpx):
        lam = wavelengths[wav]
        log.info('Wavelength = %3.1f' % (lam*1000.))
        #index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-lam))
        #print index
        for i in range(nlens):
            for j in range(nlens):
                if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
                    #print xcenter[:,i,j],np.isnan(xcenter[:,i,j]).any()
                    #print np.isnan(xcenter[0,i,j])
                    fx = interp1d(waveCalArray,xcenter[:,i,j])
                    fy = interp1d(waveCalArray,ycenter[:,i,j])
                    #print (lam,fx(lam),xcenter[:,i,j])
                    Y[j,i] = np.floor(fx(lam)*ratio)
                    X[j,i] = np.floor(fy(lam)*ratio)
        for xcount in range(-ratio//2,-ratio//2+ratio):
            for ycount in range(-(ratio*5)//2,-(ratio*5)//2+ratio*5):
                cube[wav,:,:] += ifsimageDense[X+xcount,Y+ycount]
            #cube[wav,:,:] = ifsimage[X,Y]+ifsimage[X,Y+1]+ \
            #ifsimage[X,Y-1]+ifsimage[X,Y+2]+ifsimage[X,Y-2]
    
    cube[cube==0] = np.NaN
    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)


def apertureReduction(par,name,ifsimage):
    '''
    Basic cube reduction using an IFS image and a wavecal cube
    Equivalent to method 1 in the IDL primitive 'pisces_assemble_spectral_datacube'
    '''
    calCube = pyf.open(par.wavecalDir+par.wavecalName)
    
    waveCalArray = calCube[0].data#wavecal[0,:,:]
#     waveCalArray = waveCalArray[waveCalArray>0]/1000.
    waveCalArray = waveCalArray/1000.
    
#     xcenter = np.zeros((numWCal,nlens,nlens))
#     ycenter = np.zeros((numWCal,nlens,nlens))
#     for i in range(numWCal):
#         xcenter[i,:,:] = wavecal[i*2+1,:,:]
#         ycenter[i,:,:] = wavecal[i*2+2,:,:]
    xcenter = calCube[1].data
    nlens = xcenter.shape[1]
    ycenter = calCube[2].data
    good = calCube[3].data
    #xcenter[~good] = np.NaN
    #ycenter[~good] = np.NaN
    ydim,xdim = ifsimage.shape
    xdim-=2
    xcenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan
    ycenter[~((xcenter<xdim)*(ycenter<ydim)*(xcenter>0)*(ycenter>2))] = np.nan

    
    sdpx = 26 # length of longest spectra in pixels; need to link this to the filter setting
    lam_long = max(waveCalArray)
    lam_short = min(waveCalArray)
    
    wavelengths = np.arange(sdpx+1)*(lam_long-lam_short)/float(sdpx)+lam_short
    print (wavelengths)
    
    cube = np.zeros((sdpx,nlens,nlens))

    
    for wav in range(sdpx):
        lam = wavelengths[wav]
        log.info('Wavelength = %3.1f' % (lam*1000.))
        #index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-lam))
        #print index
        for i in range(nlens):
            for j in range(nlens):
                if not (np.isnan(xcenter[:,i,j]).any() and np.isnan(ycenter[:,i,j]).any()):
                    #print xcenter[:,i,j],np.isnan(xcenter[:,i,j]).any()
                    #print np.isnan(xcenter[0,i,j])
                    fx = interp1d(waveCalArray,xcenter[:,i,j])
                    fy = interp1d(waveCalArray,ycenter[:,i,j])
                    #print (lam,fx(lam),xcenter[:,i,j])
                    pos = (fx(lam),fy(lam))
                    ap = RectangularAperture(pos,1,5,0)
                    cube[wav,j,i] = aperture_photometry(ifsimage,ap)['aperture_sum'][0]
    
    cube[cube==0] = np.NaN
    pyf.PrimaryHDU(cube).writeto(name+'.fits',clobber=True)
