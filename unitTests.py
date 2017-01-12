
import numpy as np
import tools
import logging as log
import matplotlib.pyplot as plt
from tools.image import Image
from params import Params



def testLoadKernels():
    '''
    Make sure the kernel interpolation with wavelength makes sense
    '''
    par = Params()
    tools.initLogger(par.exportDir+'/IFS.log',levelConsole=log.DEBUG)
    
    log.info('Import all kernels and rescale them to same plate scale')
    kernels890,locations = tools.loadKernels(par,890)
    kernels770,loc = tools.loadKernels(par,770)
    kernels660,loc = tools.loadKernels(par,660)
    refWaveList = [660,770,890]
    kernelList = np.array([kernels660,kernels770,kernels890])

    for lam in np.arange(0.6,0.9,0.05):
        kernels = tools.selectKernel(par,lam,refWaveList,kernelList)
        allkernels = tools.plotKernels(par,kernels,locations)
        Image(data=allkernels).write(par.unitTestsOutputs+'/kernels%.3f.fits' % (lam))



if __name__ == '__main__':
    testLoadKernels()
