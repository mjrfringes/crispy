#!/usr/bin/env python

import numpy as np
from scipy import ndimage


def Rotate(image, phi, clip=True, order=3):
    """
    Rotate the input image by phi about its center.  Do not resize the
    image, but pad with zeros.  Function originally from Tim Brandt

    Parameters
    ----------
    image : 2D square array
            Image to rotate
    phi : float
            Rotation angle in radians
    clip :  boolean (optional)
            Clip array by sqrt(2) to remove fill values?  Default True.
    order : integer (optional)
            Order of interpolation when rotating. Default is 1.

    Returns
    -------
    imageout: 2D array
            Rotated image of the same shape as the input image, with zero-padding

    """

    x = np.arange(image.shape[0])
    med_n = np.median(x)
    x -= int(med_n)
    x, y = np.meshgrid(x, x)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    x = r * np.cos(theta + phi) + med_n
    y = r * np.sin(theta + phi) + med_n

    imageout = ndimage.map_coordinates(image, [y, x], order=order)

    if clip:
        i = int(imageout.shape[0] * (1. - 1. / np.sqrt(2.)) / 2.)
        imageout = imageout[i:-i, i:-i]

    return imageout


def rotateCube(cube, phi, clip=True, order=1):
    outcube = [Rotate(cube[i], phi, clip, order) for i in range(cube.shape[0])]
    return np.array(outcube)


def Shift(image, dx, dy, order=3):
    """
    Shifts the input image by dx, dy.

    Parameters
    ----------
    image : 2D square array
            Image to rotate
    dx : float
            Rotation angle in radians
    dy :  boolean (optional)
            Clip array by sqrt(2) to remove fill values?  Default True.

    Returns
    -------
    imageout: 2D array
            Shifted image of the same shape as the input image

    Notes
    -----
    The functions pads the edges with the nearest value, including NaNs.

    """

    x = np.arange(image.shape[1], dtype=np.float)
    y = np.arange(image.shape[0], dtype=np.float)
    x -= image.shape[1] // 2
    y -= image.shape[0] // 2
    x -= dx
    y -= dy
    x, y = np.meshgrid(x, y)
    x += image.shape[1] // 2
    y += image.shape[0] // 2
#     if order>1:
#         img = ndimage.interpolation.spline_filter(image)
    imageout = ndimage.map_coordinates(image, [y, x], order=order)

    return imageout


def shiftCube(cube, dx, dy, order=3):
    outcube = [Shift(cube[i], dx, dy, order) for i in range(cube.shape[0])]
    return np.array(outcube)
