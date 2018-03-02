import scipy.interpolate
import scipy.ndimage
import numpy as np


def rebin(a, shape):
    """
    Resizes a 2d array by averaging or repeating elements,
    new dimensions must be integral factors of original dimensions

    Parameters
    ----------
    a : array_like
        Input array.
    new_shape : tuple of int
        Shape of the output array (y, x)

    Returns
    -------
    rebinned_array : ndarray
        If the new shape is smaller of the input array, the data are averaged,
        if the new shape is bigger array elements are repeated
    """

    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def frebin(array, shape, total=True):
    """
    Function that performs flux-conservative
    rebinning of an array. Found at https://github.com/benjaminpope/pysco/blob/master/frebin.py


    Parameters
    ----------
    array: ndarray
        Numpy array to be rebinned
    shape: tuple
        (x,y) of new array size
        total: Boolean
            when True flux is conserved
    Returns
    -------
        new_array: new rebinned array with dimensions: shape
    """

    # Determine size of input image
    y, x = array.shape

    y1 = y - 1
    x1 = x - 1

    xbox = x / float(shape[0])
    ybox = y / float(shape[1])

    # Determine if integral contraction so we can use rebin
    if (x == int(x)) and (y == int(y)):
        if (x % shape[0] == 0) and (y % shape[1] == 0):
            return rebin(array, (shape[1], shape[0])) * xbox * ybox

    # Otherwise if not integral contraction
    # First bin in y dimension
    temp = np.zeros((shape[1], x), dtype=float)
    # Loop on output image lines
    for i in range(0, int(shape[1]), 1):
        rstart = i * ybox
        istart = int(rstart)
        rstop = rstart + ybox
        istop = int(rstop)
        if istop > y1:
            istop = y1
        frac1 = rstart - istart
        frac2 = 1.0 - (rstop - istop)

    # Add pixel values from istart to istop an subtract
    # fracion pixel from istart to rstart and fraction
    # fraction pixel from rstop to istop.
        if istart == istop:
            temp[i, :] = (1.0 - frac1 - frac2) * array[istart, :]
        else:
            temp[i, :] = np.sum(array[istart:istop + 1, :], axis=0)\
                - frac1 * array[istart, :]\
                - frac2 * array[istop, :]

    temp = np.transpose(temp)

    # Bin in x dimension
    result = np.zeros((shape[0], shape[1]), dtype=float)
    # Loop on output image samples
    for i in range(0, int(shape[0]), 1):
        rstart = i * xbox
        istart = int(rstart)
        rstop = rstart + xbox
        istop = int(rstop)
        if istop > x1:
            istop = x1
        frac1 = rstart - istart
        frac2 = 1.0 - (rstop - istop)
    # Add pixel values from istart to istop an subtract
    # fracion pixel from istart to rstart and fraction
    # fraction pixel from rstop to istop.
        if istart == istop:
            result[i, :] = (1. - frac1 - frac2) * temp[istart, :]
        else:
            result[i, :] = np.sum(temp[istart:istop + 1, :], axis=0)\
                - frac1 * temp[istart, :]\
                - frac2 * temp[istop, :]

    if total:
        return np.transpose(result)
    elif not total:
        return np.transpose(result) / float(xbox * ybox)
