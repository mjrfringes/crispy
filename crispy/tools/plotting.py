import numpy as np
import matplotlib.pyplot as plt
import logging as log


def plotKernels(par, kernels, locations, plot=False):
    '''
    Make plots of all the kernels
    '''
    #fig,axarr = plt.subplots(3,3,sharex=True, sharey=True)
    nx, ny = kernels[0].shape
    output = np.zeros((3 * nx, 3 * ny))

    for k in range(len(locations)):
        output[locations[k, 0] *
               2 *
               nx:(locations[k, 0] *
                   2 +
                   1) *
               nx, locations[k, 1] *
               2 *
               ny:(locations[k, 1] *
                   2 +
                   1) *
               ny] += kernels[k]
        # axarr[locations[k,0]*2,locations[k,1]*2].imshow(kernels[k])
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(output, interpolation='nearest', origin='lower')
        plt.show()
    return output
