import numpy as np
from scipy.signal import medfilt
try:
    from astropy.io import fits as pyf
except:
    import pyfits as pyf


def gen_bad_pix_mask(image, filsize=3, threshold=5.0, return_smoothed_image=False):
    """
    Identify and mask bad pixels using median filter
    
    Parameters
    ----------
    image: 2D ndarray
            Image to clean
    filsize: int
            Size of median filter in pixels
    threshold: float
            Threshold in terms of standard deviations
    return_smoothed_image: boolean
    
    Returns
    -------
    goodpix: 2D int ndarray
            Good pixel mask. 1 where pixel is good
    image_sm: 2D ndarray, if return_smoothed_image

    """
    image_sm = medfilt(image, filsize)
    res = image - image_sm
    sigma = np.std(res)
    goodpix = np.abs(res)/sigma < threshold
    return (goodpix, image_sm) if return_smoothed_image else goodpix


def gen_lenslet_flat(BBcube, nsig=5):

    lenslet_flat=np.nansum(BBcube.data,axis=0)
    
    mask = (lenslet_flat.data!=0)

    x = np.arange(lenslet_flat.shape[0])
    
    # select only central region
    med_n = np.median(x)
    x -= int(med_n)	
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x**2 + y**2)
    stdmask = mask*(r<20)
    
    sig = np.std(lenslet_flat[stdmask])
    ave = np.mean(lenslet_flat[stdmask])
    print("Mean, sig in central 20 lenslets:",ave,sig)

    mask *= (lenslet_flat<ave+nsig*sig)*(lenslet_flat>ave-nsig*sig)

    fullave = np.mean(lenslet_flat[mask])
    norm_lenslet_flat = lenslet_flat/fullave
    norm_lenslet_flat[norm_lenslet_flat==0] = np.NaN

    
    return 1./norm_lenslet_flat, mask



def bowtie(image,xc,yc,openingAngle,clocking,IWApix,OWApix,export='bowtie',twomasks=False):
    '''
    Creates one or two binary mask for a shaped pupil bowtie mask
    
    Parameters
    ----------
    image: 2D ndarray
            Image to which the mask needs to be applied
    xc: int
            X coordinate of center of bowtie
    yc: int
            Y coordinate of center of bowtie
    openingAngle: float
            Angle in degrees representing the opening angle of the bowtie mask
    clocking: float
            Angle of rotation of the mask in degrees
    IWApix: float
            Radius in pixels corresponding to the inner working angle of the mask
    OWApix: float
            Radius in pixels corresponding to the outer working angle of the mask
    export: boolean
            Whether to export the bowtie or not
    twomasks: boolean
            If True, returns two masks, one for each side of the bowtie
            If False, returns one single mask
            
    Returns
    -------
    mask: 2D ndarrays of int
            Bowtie mask with 1 inside the mask, zero everywhere else.
    mask2: 2D ndarrays of int
            If twomasks is True, mask and mask2 are the two sides of the bowtie.
    '''
    
    x = np.arange(image.shape[0])
    y = np.arange(image.shape[1])
    x -= xc
    y -= yc
    x,y = np.meshgrid(x,y)
    
    r = np.sqrt(x**2 + y**2)
    
    clocking *=np.pi/180.
    openingAngle *=np.pi/180.
    
    # rotate the phase map so that the wrapping occurs on the axis of symmetry between
    # the two bowtie sides
    theta = (np.arctan2(y, x)-clocking - np.pi/2.)%(2*np.pi)

    mask = (r<OWApix)*(r>IWApix)
    mask *= (theta<openingAngle/2.+np.pi/2.)
    mask *= (theta>-openingAngle/2.+np.pi/2.)
    mask2 = (r<OWApix)*(r>IWApix)
    mask2 *= (theta>-openingAngle/2.+np.pi*1.5)
    mask2 *= (theta<openingAngle/2.+np.pi*1.5)
    
    if twomasks:
        if export is not None:
            out = pyf.HDUList(pyf.PrimaryHDU(mask.astype(np.int)))
            out.writeto(export+'1.fits', clobber=True)
            out = pyf.HDUList(pyf.PrimaryHDU(mask2.astype(np.int)))
            out.writeto(export+'2.fits', clobber=True)
        return mask,mask2
    else:
        mask += mask2
        if export is not None:
            out = pyf.HDUList(pyf.PrimaryHDU(mask.astype(np.float)))
            out.writeto(export+'.fits', clobber=True)
        return mask,mask    
    
    
    
def scale2imgs(img1,img2,mask=None,returndiff = True):
    '''
    Finds the slice-by-slice best-fit scale factor between two images.
    Optionally returns the difference between the two. 
    Images can be cubes.
    
    
    '''
    # make local copies of data
    c1 = img1.data.copy()
    c2 = img2.data.copy()
    c1[np.isnan(c1)]=0.0
    c2[np.isnan(c2)]=0.0
    
    res = []
    for i in range(c1.shape[0]):
        if mask is not None:
            c1[i] *= mask
            c2[i] *= mask
        refslice = np.reshape(c1[i], (1, -1))
        targetslice = np.reshape(c2[i],-1)
        res.append(np.linalg.lstsq(refslice.T,targetslice)[0])
    res = np.array(res).flatten()
    if returndiff:
        return res, c1*res[:,np.newaxis,np.newaxis]-c2
    else:
        return res
