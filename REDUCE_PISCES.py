import tools
import glob
from PISCESparams import Params
import numpy as np
import logging as log
from tools.initLogger import initLogger


par = Params()

# This is the logger system to print to file and to print to console
initLogger(par.exportDir+'/IFS.log')




# the following is to do the wavelength calibration. You don't need this now.

# from tools.wavecal import buildcalibrations
# par.lamlist = np.arange(605,725+1,10)
# par.filelist = np.sort(glob.glob(par.wavecalDir + "Calibra*.FIT"))
# buildcalibrations(par,parallel=False,hires=False)

from IFS import reduceIFSMap
fname=''    #INSERT YOUR PISCES FILENAME HERE
cube = reduceIFSMap(par,fname,method='intopt')

# the wavelength of each slice is retrieved by this:
from tools.reduction import calculateWaveList
lam_midpts,lam_endpts = calculateWaveList(par)
print lam_midpts