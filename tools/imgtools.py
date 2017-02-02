import numpy as np
from scipy.signal import medfilt


def gen_bad_pix_mask(image, filsize=3, threshold=5.0, return_smoothed_image=False):
    """
    """
    image_sm = medfilt(image, filsize)
    res = image - image_sm
    sigma = np.std(res)
    goodpix = np.abs(res)/sigma < threshold
    return (goodpix, image_sm) if return_smoothed_image else goodpix
