#!/usr/bin/env python

import numpy as np

def FT(Ein, z, a, b, u, v, lam, nx, ny):

    """
    2D Fourier transform.  Not the FFT.  
    """

    neta, nxi = Ein.shape
    dxi = a/(nxi - 1.)
    deta = b/(neta - 1.)
    dx = u/(nx - 1.)
    dy = v/(ny - 1.)
    xi = np.linspace(-(nxi - 1)/2., (nxi - 1)/2., nxi)*dxi
    eta = np.linspace(-(neta - 1)/2., (neta - 1)/2., neta)*deta

    x = np.linspace(-(nx - 1)/2., (nx - 1)/2., nx)*dx
    y = np.linspace(-(ny - 1)/2., (ny - 1)/2., ny)*dy

    x_xi = np.dot(x[:, None], xi[None, :])
    y_eta = np.dot(y[:, None], eta[None, :])
    
    return np.exp(2.j*np.pi*z/lam)/(1.j*lam*z)*np.dot(np.exp(-2.j*np.pi*y_eta/lam/z), np.dot(Ein, np.exp(-2.j*np.pi*x_xi/lam/z).T))*dxi*deta

