from astropy.io import fits

import numpy as np
import glob
import matplotlib.pyplot as plt
import codecs
from tools.image import Image


folder = 'ReferenceFiles/simpsf/'
spotfields = glob.glob(folder+'*')

for wavel in [660,770,890]:
    f, axarr = plt.subplots(3,3)
    for field in spotfields:    
        if ".txt" in field and str(wavel) in field:
            # deals with Windows encoding
            readFile = codecs.open(field,encoding='utf-16-le')
            dat = np.loadtxt(readFile,skiprows=18)
            dat /= np.sum(dat)
            im = Image(data=dat)
            im.header['PSFWAVEL'] = (wavel,'Wavelength of PSF')

            if "_1" in field:
                axarr[1,1].imshow(dat,interpolation='nearest',origin='lower')
                im.write(folder+str(wavel)+'PSF_0_0.fits')
            if "_2" in field:
                axarr[1,0].imshow(dat,interpolation='nearest',origin='lower')
                im.write(folder+str(wavel)+'PSF_0_1.fits')
            if "_3" in field:
                axarr[1,2].imshow(dat,interpolation='nearest',origin='lower')
                im.write(folder+str(wavel)+'PSF_0_-1.fits')
            if "_4" in field:
                axarr[2,1].imshow(dat,interpolation='nearest',origin='lower')
                im.write(folder+str(wavel)+'PSF_1_0.fits')
            if "_5" in field:
                axarr[0,1].imshow(dat,interpolation='nearest',origin='lower')
                im.write(folder+str(wavel)+'PSF_-1_0.fits')
            if "_6" in field:
                axarr[2,0].imshow(dat,interpolation='nearest',origin='lower')
                im.write(folder+str(wavel)+'PSF_1_1.fits')
            if "_7" in field:
                axarr[2,2].imshow(dat,interpolation='nearest',origin='lower')
                im.write(folder+str(wavel)+'PSF_1_-1.fits')
            if "_8" in field:
                axarr[0,0].imshow(dat,interpolation='nearest',origin='lower')
                im.write(folder+str(wavel)+'PSF_-1_1.fits')
            if "_9" in field:
                axarr[0,2].imshow(dat,interpolation='nearest',origin='lower')
                im.write(folder+str(wavel)+'PSF_-1_-1.fits')
    f.savefig(folder+str(wavel)+'PSFs.png')


    