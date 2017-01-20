from tools.locate_psflets import locatePSFlets
from tools.image import Image
from ../IFS import propagateIFS
from photutils import CircularAperture
import matplotlib as mpl


def createWavecalFiles(par):
    '''
    Creates a set of monochromatic IFS images to be used in wavelength calibration step
    
    '''
    wavelist = np.arange(605,726,10)
    par.saveDetector=False
    lamoD = 3. # number of lenslet per lamoD at 660nm
    inputCube = np.ones((1,512,512),dtype=float)/9.
    filelist = []
    lamlist = []
    for wav in wavelist:
        mperpix = par.pitch/lamoD*wav/660.
        par.pixperlenslet = par.pitch/mperpix
        par.mperpix = mperpix
        detectorFrame = propagateIFS(par,[wav*1e-3],inputCube)
        lamlist.append(wav)
        filename = par.wavecalDir+'det_%3d.fits' % (wav)
        filelist.append(filename)
        Image(data=detectorFrame).write(filename)
    par.lamlist = lamlist
    par.filelist = filelist


def computeWavecal(par,lamlist=None,filelist=None,order = 3):
    '''
    Computes a wavelength calibration from a set of fits files
    Uses Tim Brandt's locate_PSFlets routine with an initial guess.
    The consecutive solutions use information from previous solutions to improve the
    initial guess.
    The process can be optimized somewhat by reducing the number of lenslets within
    the locatePSFlets function (could make this a parameter!)
    lamlist and filelist can be defined in the parameters in which case they don't need to be set
    
    Inputs
    1. par          Parameter instance with at least the key IFS parameters, interlacing and scale
    2. lamlist      list of wavelengths in nm
    3. filelist     list of fits files to open in corresponding order
    4. order        order of 2d polynomial to be fitted to lenslets
    
    '''
    # lamlist/filelist override
    if (filelist!=None) and (lamlist!=None):
        par.filelist = filelist
        par.lamlist = lamlist
    coef = None
    allcoef = []    
    xpos = []
    ypos = []
    good = []

    for i in range(len(par.lamlist)):
        inImage = Image(filename=par.filelist[i])
        _x, _y, _good, coef = locatePSFlets(inImage, polyorder=order, coef=coef,phi=par.philens,scale=par.pitch/par.pixsize)
        xpos += [_x]
        ypos += [_y]
        good += [_good]
        allcoef += [[par.lamlist[i]] + list(coef)]
    log.info("Saving wavelength solution to " + par.wavecalDir + "lamsol.dat")
    outkey = pyf.HDUList(pyf.PrimaryHDU(np.asarray(par.lamlist)))
    outkey.append(pyf.PrimaryHDU(np.asarray(xpos)))
    outkey.append(pyf.PrimaryHDU(np.asarray(ypos)))
    outkey.append(pyf.PrimaryHDU(np.asarray(good).astype(np.uint8)))
    outkey.writeto(par.wavecalDir + par.wavecalName, clobber=True)

    allcoef = np.asarray(allcoef)
    np.savetxt(par.wavecalDir + "lamsol.dat", allcoef)



def inspectWaveCal(par,slice=0,name='inspectWavecal'):
    '''
    Inspects a wavecal solution by overplotting apertures on top of the image
    
    Inputs:
    1. par          Parameter instance
    2. slice        which slice of the wavelength calibration to look at
    3. save         whether to save the figure or not
    
    '''
    calCube = pyf.open(par.wavecalDir+par.wavecalName)
    waveCalArray = calCube[0].data
    i = slice
    log.info('Displaying PSF centroid fits for wavelength %3d' % (waveCalArray[i]))
    xorig = calCube[1].data[i,:,:]
    yorig = calCube[2].data[i,:,:]
    xdim,ydim = yorig.shape

    xg,yg = xorig.shape
    vals = np.array([(xorig[m,n],yorig[m,n]) for m in range(xg) for n in range(yg)])
    pos = (vals[:,0],vals[:,1])
    aps = CircularAperture(pos, r=3)
    fig,ax = plt.subplots(figsize=(15,15))
    hdulist = pyf.open(par.filelist[i],ignore_missing_end=True)
    if hdulist[0].header['NAXIS']!=2:
        image = pyf.open(par.filelist[i],ignore_missing_end=True)[1].data
    else:
        image = pyf.open(par.filelist[i],ignore_missing_end=True)[0].data
    mean = np.mean(image)
    std = np.std(image)
    norm = mpl.colors.Normalize(vmin=mean,vmax=mean+5*std)
    ax.imshow(image, cmap='Greys',norm=norm,interpolation='nearest',origin='lower')
    aps.plot(ax=ax,color='blue', lw=1, alpha=0.5)
    fig.savefig(par.wavecalDir+name+'_'+str(waveCalArray[i]),dpi=300)
