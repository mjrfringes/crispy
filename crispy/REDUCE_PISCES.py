from PISCESparams import Params

par = Params()

# the following is to do the wavelength calibration. You don't need this now.

# from tools.wavecal import buildcalibrations
# par.lamlist = np.arange(605,725+1,10)
# par.filelist = np.sort(glob.glob(par.wavecalDir + "Calibra*.FIT"))
# buildcalibrations(par,parallel=False,hires=False)

from IFS import reduceIFSMap
fname=''    #INSERT YOUR PISCES FILENAME HERE
cube = reduceIFSMap(par,fname)

# the wavelength of each slice is retrieved by this:
import tools
from tools.reduction import calculateWaveList
lam_midpts,lam_endpts = calculateWaveList(par)
print lam_midpts