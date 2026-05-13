"""Light image processing"""

from numpy import fft
import numpy as np
import coloredlogs
import logging
import sys
from scipy.ndimage import center_of_mass,shift

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

log = logging.getLogger('')


def crop_center(array, cut, center=None):
    """crop out center of array

    Parameters
    ----------
    array : numpy.ndarray
        array to crop
    cut : int
        radius of array to crop
    center : list of length 2, defaults to None
        If not none, defines the center of the array cropping. Otherwise, defaults to centering using
        scipy.ndimage.center_of_mass.

    Returns
    -------
    numpy.ndarray
        cropped array of size 2*cut x 2*cut
    """

    log.debug(sys._getframe().f_code.co_name)

    # get dimensions
    xshape = int(array.shape[0]/2)
    yshape = int(array.shape[1]/2)

    if center is not None:
        com = center_of_mass(array)
        xcom,ycom = int(com[0]),int(com[1])
    else:
        xcom,ycom = center[0],center[1]

    return array[xcom-cut:xcom+cut,ycom-cut:ycom+cut]
    
def ft_tilt(array, shiftx, shifty):
    """method to perform subpixel shifts using the fourier transformation on square arrays
    
    Parameters
    ----------
    array : numpy.ndarray
        array to shift
    shiftx : float
        number of pixels to shift in axis 0
    shifty : float
        number of pixels to shift in axis 1
        
    Returns
    -------
    numpy.ndarray
        shifted array
    """
    
    log.debug(sys._getframe().f_code.co_name)

    # TODO: consider re-working for rectangular images, will require MFT
    imshape = array.shape[0]
    
    # construct phasor
    u = np.linspace(-np.pi,np.pi,imshape)
    u,v = np.meshgrid(u,u)
    phase_ramp = shiftx*u + shifty*v
    phasor = np.exp(1j*phase_ramp)
    
    # apply tilt
    to_pupil = ft_rev(array) * phasor
    
    return ft_fwd(to_pupil)
    
def ft_center_array(array, criterion='maximum'):
    """method to perform subpixel shifts using the fourier transformation on square arrays

    FIXME: This doesn't work, it just calls SCIPY's shift
    
    Parameters
    ----------
    array : numpy.ndarray
        array to shift
    criterion : str
        'max' or 'center of mass', otherwise just returns the array
        
    Returns
    -------
    numpy.ndarray
        shifted array
    """

    log.debug(sys._getframe().f_code.co_name)

    if criterion == 'maximum':
        center = np.where(array==np.max(array))
        print('maximum at = ',center)
        centers_x = center[0]
        centers_y = center[1]
        which = 0
        center = [centers_x[which],centers_y[which]]
        print('chosen center = ',center)
    elif criterion == 'center of mass':
        center = center_of_mass(array)
        
    else:
        print('criterion {criterion} not recognized, defaulting to array center')
        return array
    
    # get current center
    cx,cy = center[0],center[1]
    
    # find array center
    imshape = array.shape[0]
    array_x = int(imshape/2)
    array_y = array_x
    
    # compute the shift
    shift_x = array_x - cx
    shift_y = array_y - cy
    print(f'shifts = {shift_x},{shift_y}')
    
    # tilt the image
    shifted_image = shift(array,[shift_x,shift_y],order=1)#ft_tilt(array,shift_x,shift_y)
    
    return np.abs(shifted_image),[shift_x,shift_y]
    

def threshold(array, threshold, threshold_value=0):
    """


    Parameters
    ----------
    array : numpy.ndarray
        image to threshold
    threshold : float
        value to threshold below
    threshold_value : float
        value to set thresholded values to
        
    Returns
    -------
    numpy.ndarray
        thresholded array
    """

    log.debug(sys._getframe().f_code.co_name)

    array_thresholded = np.copy(array)
    array_thresholded[array<threshold] = threshold_value
    return array_thresholded

def phase_unwrap_2d(phase_wrapped):

    """phase unwrapping routine based on the phaseunwrap2d.go script in IDL and the following proceedings:
    M.D. Pritt; J.S. Shipman, "Least-squares two-dimensional phase unwrapping using FFT's",
    IEEE Transactions on Geoscience and Remote Sensing ( Volume: 32, Issue: 3, May 1994),
    DOI: 10.1109/36.297989

    Uses a finite differences approach to determine the partial derivative of the wrapped phase in x and y,
    then solves the solution in the fourier domain

    TODO: Test this function against the prior in IDL, it doesn't appear to reconstruct phase well

    Parameters
    ----------
    phase_wrapped : numpy.ndarray
        array containing 2D signal to unwrap

    Returns
    -------
    numpy.ndarray
        unwrapped phase
    """

    log.debug(sys._getframe().f_code.co_name)

    imsize = phase_wrapped.shape
    M = imsize[0]
    N = imsize[1]

    Nmirror = 2 * (N )
    Mmirror = 2 * (M )

    phmirror = np.ones([Mmirror,Nmirror])

    # Quadrant 3
    phmirror[:M,:N] = phase_wrapped

    # First mirror reflection Quadrant 2
    phmirror[M:,:N] = np.flipud(phase_wrapped)

    # Second mirror reflection Quadrant 4
    phmirror[:M,N:] = np.fliplr(phase_wrapped)

    # Final reflection Quadrant 1
    phmirror[M:,N:] = np.flipud(np.fliplr(phase_wrapped))

    phroll = np.zeros_like(phmirror)
    phroll[:M,:N-1] = phmirror[:M,1:N]
    phroll[:M,N-1] = phmirror[:M,0]
    deltafd = phroll-phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)

    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdx = deltafd

    # compute forward difference
    phroll = np.zeros_like(phmirror)
    phroll[:M-1,:N] = phmirror[1:M,:N]
    phroll[M,:N] = phmirror[0,:N]
    deltafd = phroll - phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)
    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdy = deltafd

    # Solve system of equations formed by min LS -> phi
    D_n = np.fft.fft2(deltafdx)
    D_m = np.fft.fft2(deltafdy)
    inc_n = 2 * np.pi / Nmirror
    inc_m = 2 * np.pi / Mmirror

    nn = np.ones([Mmirror,1]) @ (np.arange(Nmirror))[np.newaxis]
    mm = np.ones([Nmirror,1]) @ (np.arange(Mmirror))[np.newaxis]
    mm = mm.transpose()
    print(mm.shape)
    i = 1j
    mult_n = np.ones([Mmirror,Nmirror]) - np.exp(-nn * i * inc_n)
    mult_m = np.ones([Mmirror,Nmirror]) - np.exp(-mm * i * inc_m)
    divisor = (np.cos(mm*inc_m) + np.cos(nn*inc_n) - np.ones([Mmirror,Nmirror])*2)*2
    divisor[0,0] = 1
    phi = (D_n*mult_n + D_m*mult_m) / divisor
    phi[0,0] = 0
    phi = np.fft.ifft2(phi)[:M,:N]
    phout = np.real(phi)
    return phout