#!/usr/bin/env python

from scipy import signal
import scipy.interpolate
import scipy.ndimage

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from detutils import frebin
from initLogger import getLogger
log = getLogger('crispy')
from image import Image

def rebinDetector(par,finalFrame,clip=False):
    '''
    Rebins the dense detector map with the correct scaling while conserving flux.
    This also works with non-integer ratios.
    
    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    finalFrame : 2D ndarray
            Dense detector map to be rebinned.
    
    Returns
    -------
    detectorFrame : 2D array
            Return the detector frame with correct pixel scale.
    
    '''
    detpixprlenslet = par.pitch/par.pixsize 
    log.info('Number of detector pixels per lenslet: %f' % detpixprlenslet)
        
    newShape = (finalFrame.shape[0]//(par.pxperdetpix),finalFrame.shape[1]//(par.pxperdetpix))
    log.info('Rebinning final detector. Image has dimensions %dx%d' % newShape)
    detectorFrame = frebin(finalFrame,newShape) 
    
    if clip:
        i = int(detectorFrame.shape[0]*(1. - 1./np.sqrt(2.))/2.)
        detectorFrame = detectorFrame[i:-i, i:-i]

    return detectorFrame



def readDetector(par,IFSimage,inttime=100,append_header=False):
    '''
    Read noise, CIC, dark current; NO TRAPS
    Input is IFSimage in average photons per second
    Quantum efficiency considerations are already taken care of when
    generating IFSimage images
    '''
    if append_header and not 'RN' in par.hdr:
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('comment', '*'*60), end=True)
        par.hdr.append(('comment', '*'*22 + ' Detector readout ' + '*'*20), end=True)
        par.hdr.append(('comment', '*'*60), end=True)    
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('RN',par.RN,'Read noise (electrons/read)'), end=True) 
        par.hdr.append(('CIC',par.CIC,'Clock-induced charge'), end=True) 
        par.hdr.append(('DARK',par.dark,'Dark current'), end=True) 
        par.hdr.append(('Traps',par.Traps,'Use traps? T/F'), end=True) 
        par.hdr.append(('INTTIME',inttime,'Integration time (s)'), end=True) 
        
        
#     par.hdr.append(('INTTIME',inttime,'Integration time'), end=True)
    ### thoughts on implementing the EMGain:
    # This requires an inverse cumulative probability density which depends
    # on the number of incoming electrons in the well, with a max of 32.
    # Suggestion is to pre-compute the 32 required functions, save them
    # then apply them to the array, for example using np.vectorize
    # Another way would be to make lists of coordinates for all pixels with the same
    # values, and call this icdf a maximum of 32 times; after the random numbers
    # are generated, put them back in their right place on the detector.
    ###
    detector = np.random.poisson(IFSimage.data*inttime+par.dark*inttime+par.CIC)
    if par.RN>0:
        detector += np.random.normal(0.0,par.RN,IFSimage.data.shape)
    return detector

def averageDetectorReadout(par,filelist,detectorFolderOut,suffix = 'detector',offaxis=None,averageDivide=False):
    '''	
    Process a list of files and creates individual detector readouts
    If we want only one file, we can just make a list of 1
    '''
    det_outlist = []
	
    for reffile in filelist:
        log.info('Apply detector readout on '+reffile.split('/')[-1])
        img = Image(filename=reffile)
        if offaxis is not None:
            off = Image(offaxis)
            img.data+=off.data
        inttime = par.timeframe/par.Nreads
        img.data*=par.QE*par.losses*par.PhCountEff*par.CTE*par.pol
        #refreshes parameter header
        par.makeHeader()
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('comment', '*'*60), end=True)
        par.hdr.append(('comment', '*'*22 + ' Detector readout ' + '*'*20), end=True)
        par.hdr.append(('comment', '*'*60), end=True)    
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('RN',par.RN,'Read noise (electrons/read)'), end=True) 
        par.hdr.append(('CIC',par.CIC,'Clock-induced charge'), end=True) 
        par.hdr.append(('DARK',par.dark,'Dark current'), end=True) 
        par.hdr.append(('Traps',par.Traps,'Use traps? T/F'), end=True) 
        par.hdr.append(('QE',par.QE,'Quantum efficiency of the detector'),end=True)
        par.hdr.append(('PHCTEFF',par.PhCountEff,'Photon counting efficiency'),end=True)
        par.hdr.append(('CTE',par.CTE,'Charge transfer efficiency'),end=True)
        par.hdr.append(('TRANS',par.losses,'IFS Transmission factor'),end=True)
        par.hdr.append(('POL',par.pol,'Polarization losses'),end=True)
        par.hdr.append(('INTTIME',inttime,'Time for each infividual frame'),end=True)
        par.hdr.append(('NREADS',par.Nreads,'Number of frames averaged'),end=True)
        par.hdr.append(('EXPTIME',par.timeframe,'Total exposure time'),end=True)

        frame = np.zeros(img.data.shape)
        varframe = np.zeros(img.data.shape)
        # averaging reads
        for i in range(par.Nreads):
            newread = readDetector(par,img,inttime=inttime)
            frame += newread
            varframe += newread**2
        if averageDivide:
            frame /= par.Nreads
            varframe /= par.Nreads
            varframe -= frame**2
        outimg = Image(data=frame,ivar=1./varframe,header=par.hdr)
        # append '_suffix' to the file name
        outimg.write(detectorFolderOut+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits',clobber=True)
        det_outlist.append(detectorFolderOut+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
    return det_outlist


def noiselessDetector(par,filelist,detectorFolderOut,suffix = 'detector',offaxis=None):
    '''	
    Process a list of files and creates individual detector readouts
    If we want only one file, we can just make a list of 1
    '''
    det_outlist = []
	
    for reffile in filelist:
        log.info('Apply noiseless detector readout on '+reffile.split('/')[-1])
        img = Image(filename=reffile)
        if offaxis is not None:
            off = Image(offaxis)
            img.data+=off.data
        inttime = par.timeframe/par.Nreads
        img.data*=par.QE*par.losses
        #refreshes parameter header
        par.makeHeader()
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('comment', '*'*60), end=True)
        par.hdr.append(('comment', '*'*22 + ' Noiseless detector readout ' + '*'*20), end=True)
        par.hdr.append(('comment', '*'*60), end=True)    
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('RN',0,'Read noise (electrons/read)'), end=True) 
        par.hdr.append(('CIC',0,'Clock-induced charge'), end=True) 
        par.hdr.append(('DARK',0,'Dark current'), end=True) 
        par.hdr.append(('Traps',par.Traps,'Use traps? T/F'), end=True) 
        par.hdr.append(('QE',par.QE,'Quantum efficiency of the detector'),end=True)
        par.hdr.append(('TRANS',par.losses,'IFS Transmission factor'),end=True)
        par.hdr.append(('INTTIME',inttime,'Time for each infividual frame'),end=True)
        par.hdr.append(('NREADS',1,'Number of frames averaged'),end=True)
        par.hdr.append(('EXPTIME',par.timeframe,'Total exposure time'),end=True)

        frame = np.zeros(img.data.shape)
        # averaging reads
        frame = img.data*par.timeframe
        outimg = Image(data=frame,header=par.hdr)
        # append '_suffix' to the file name
        outimg.write(detectorFolderOut+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits',clobber=True)
        det_outlist.append(detectorFolderOut+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
    return det_outlist
