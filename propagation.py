"""propagation routines and wrappers around prysm propagation routines"""

from numpy import fft
import numpy as np
import coloredlogs
import logging
import sys
from prysm.propagation import focus_fixed_sampling,unfocus_fixed_sampling,psf_sample_to_pupil_sample


"""Boilerplate from snipet.py"""
# Set this flag to always see debug logs
DEBUG = False
# Set up the base logger all threads will use, once we know the debug flag
coloredlogs.DEFAULT_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
coloredlogs.DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
# Adjust color of the log depending on the DEBUG variable.
if DEBUG:
    coloredlogs.install(level='DEBUG')
else:
    coloredlogs.install(level='INFO')
#
log = logging.getLogger('')

"""Propagation Functions"""
def _angular_spectrum_transfer_function(shape, wvl, dx, z):
    """init the transfer function of free space

    Parameters
    ----------
    shape : float
        shape of the array to propagate, array.shape[0] assuming square arrays
    wvl : float
        wavelength in units of microns
    dx : float
        sample size (i.e. how big is a pixel) in microns
    z : float
        distance to propagate, microns

    Returns
    -------
    numpy.ndarray
        transfer function of free space
    """
    log.debug(sys._getframe().f_code.co_name)
    ky, kx = (fft.fftfreq(s, dx) for s in shape)
    ky = np.broadcast_to(ky, shape).swapaxes(0, 1)
    kx = np.broadcast_to(kx, shape)

    coef = np.pi * wvl * z
    transfer_function = np.exp(-1j * coef * (kx**2 + ky**2))
    return transfer_function

def _angular_spectrum_prop(field, transfer_function):
    """Propagate a field using the angular spectrum method

    Parameters
    ----------
    field : numpy.ndarray
        field to propagate
    transfer_function : numpy.ndarray
        transfer function of free space, call _angular_spectrum_transfer_function()

    Returns
    -------
    numpy.ndarray
        propagated field
    """
    # this code is copied from prysm with some modification
    log.debug(sys._getframe().f_code.co_name)
    forward = fft.fft2(field)
    return fft.ifft2(forward*transfer_function)

def ft_fwd(x):
    """ 'focus' operator, wrapper for numpy fft that conserves energy

    Parameters
    ----------
    x : numpy.ndarray
        field to focus

    Returns
    -------
    numpy.ndarray
        focused field
    """
    log.debug(sys._getframe().f_code.co_name)
    return fft.ifftshift(fft.fft2(fft.fftshift(x), norm='ortho'))

def ft_rev(x):
    """ 'unfocus' operator, wrapper for numpy fft that conserves energy

    Parameters
    ----------
    x : numpy.ndarray
        field to unfocus

    Returns
    -------
    numpy.ndarray
        unfocused field
    """
    log.debug(sys._getframe().f_code.co_name)
    return fft.ifftshift(fft.ifft2(fft.fftshift(x), norm='ortho'))

def mft_fwd(x, config):
    """focus operator, wrapper for prysm MFT

    Parameters
    ----------
    x : numpy.ndarray
        field to focus
    config : dict, optional
        dictionary containing instrument configuration for focal plane sampling, by default cred2_conf

    Returns
    -------
    numpy.ndarray
        focused field
    """

    log.debug(sys._getframe().f_code.co_name)
    conf = config.conf
    pupil_size = conf['pupil_size']#psf_sample_to_pupil_sample(conf['image_dx'],x.shape[0],conf['wavelength'],conf['efl'])
    pupil_dx = pupil_size/x.shape[0] # convert to mm
    return focus_fixed_sampling(x,pupil_dx,conf['efl'],conf['wavelength'],conf['image_dx'],x.shape[0])
    
def mft_rev(x, config):
    """unfocus operator, wrapper for prysm MFT

    Parameters
    ----------
    x : numpy.ndarray
        field to unfocus
    config : dict, optional
        dictionary containing instrument configuration for focal plane sampling, by default cred2_conf

    Returns
    -------
    numpy.ndarray
        focused field
    """

    log.debug(sys._getframe().f_code.co_name)
    conf = config.conf
    output_dx = conf['pupil_size']/x.shape[0]
    return unfocus_fixed_sampling(x,conf['image_dx'],conf['efl'],conf['wavelength'],output_dx,x.shape[0])
