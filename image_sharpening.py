"""image sharpening scripts for Keck 2 AO Bench
image_sharpening takes heavilly from the PRAISE code repository on Github,
the scripts here are largely modifications of what can be found there
https://github.com/brandondube/praise/tree/master
"""

from numpy import fft
import numpy as np
import coloredlogs
import logging
import sys

# physical optics propagation routines
from propagation import (
    _angular_spectrum_transfer_function,
    _angular_spectrum_prop,
    ft_fwd,
    ft_rev,
    mft_fwd,
    mft_rev
)


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

"""Supported Instrument Configurations"""
test_conf ={
    'image_dx':5, # um
    'efl':13*10.950e3, # mm
    'wavelength':1, # um
    'pupil_size':10.950e3 #mm
}

# CRED2 image plane config
cred2_conf ={
    'image_dx':15, # um
    'efl':437.869, # mm
    'wavelength':1.550, # um
    'pupil_size':437.869/32 #mm
}

# the instrument configuration in the IDL image sharpening script
idl_conf = {
    'image_dx':0.00995/206265*1.495830136381861e5*1e-3, # um
    'efl':1.495830136381861e5, # mm
    'wavelength':2.1686, # um
    'pupil_size':10.950e3 #mm
}


"""Hot-swappable Configuration"""
class InstrumentConfiguration:

    def __init__(self,configuration):
        """Instrument configuration class that contains data about the system needed
        for phase retrieval MFTs.

        Parameters
        ----------
        configuration : dict
            dictionary containing the following keys
            - 'image_dx' : size of pixels on the image plane, microns
            - 'efl' : effective focal length of the system, mm
            - 'wavelength' : wavelength of light, microns
            - 'pupil_size' : diameter of the entrance pupil, mm
        """
        self.conf = configuration

# run a default configuration
set_conf = InstrumentConfiguration(test_conf)

def update_instrument_configuration(configuration):
    """swap the configuration used to set the sampling for phase retrieval MFTs
    
    Parameters
    ----------
    configuration : dict
            dictionary containing the following keys
            - 'image_dx' : size of pixels on the image plane, microns
            - 'efl' : effective focal length of the system, mm
            - 'wavelength' : wavelength of light, microns
            - 'pupil_size' : diameter of the entrance pupil, mm
    """
    log.debug(sys._getframe().f_code.co_name)
    set_conf.conf = configuration
    return

"""Useful functions for setting up iterative transformers"""
def _mean_square_error(a, b, norm=1):
    """elementwise mean square error of two arrays

    Parameters
    ----------
    a : numpy.ndarray
        array 1
    b : numpy.ndarray
        _description_
    norm : int, optional
        denominator of the mean square error, by default 1

    Returns
    -------
    numpy.ndarray
        mean square error
    """
    log.debug(sys._getframe().f_code.co_name)
    diff = a - b
    mse = np.sum(diff**2)
    return mse / norm

def _init_iterative_transform(self, psf, pupil_amplitude, phase_guess=None):
    """init function for iterative transform algorithms

    Parameters
    ----------
    psf : numpy.ndarray
        contains the measured PSF to retrieve the phase for
    pupil_amplitude : numpy.ndarray
        contains a model of the pupil function
    phase_guess : numpy.ndarray, optional
        starting point for phase retrieval, by default None
    """

    log.debug(sys._getframe().f_code.co_name)
    
    if phase_guess is None:
        phase_guess = np.random.rand(*pupil_amplitude.shape)

    absF = np.sqrt(psf)
    absg = pupil_amplitude

    self.absF = fft.ifftshift(absF)
    self.absg = fft.ifftshift(absg)
    phase_guess = fft.ifftshift(phase_guess)

    self.g = self.absg * np.exp(1j*phase_guess)
    self.mse_denom = np.sum((self.absF)**2)
    self.iter = 0
    self.costF = []

def _init_iterative_transform_fixedsampling(self, psf, pupil_amplitude, phase_guess=None):
    """init function for iterative transform algorithms using matrix fourier transforms

    Parameters
    ----------
    psf : numpy.ndarray
        contains the measured PSF to retrieve the phase for
    pupil_amplitude : numpy.ndarray
        contains a model of the pupil function
    phase_guess : numpy.ndarray, optional
        starting point for phase retrieval, by default None
    """
    log.debug(sys._getframe().f_code.co_name)
    
    if phase_guess is None:
        phase_guess = np.random.rand(*pupil_amplitude.shape)

    self.absF = np.sqrt(psf)
    self.absg = pupil_amplitude

    self.g = self.absg * np.exp(1j*phase_guess)
    self.mse_denom = np.sum((self.absF)**2)
    self.iter = 0
    self.costF = []

"""Fixed Sampling Phase Retrieval

The following classes employ a matrix fourier transform (MFT) and the user-specified sampling configuration to enforce the pupil and image sampling,
they are the preferred algorithms for single-plane phase retrieval.

"""
class GerchbergSaxtonFixedSampling:
    """Gerchberg Saxton iterator, using matrix fourier transforms

    Notes
    -----
    Details about this algorithm can be found in Fienup (1982)
    https://doi.org/10.1364/AO.21.002758
    """
    def __init__(self,psf,pupil_amplitude,phase_guess=None):
        _init_iterative_transform_fixedsampling(self, psf, pupil_amplitude, phase_guess)

    def step(self):
        """Advance the algorithm one iteration."""

        log.debug(sys._getframe().f_code.co_name)

        G = mft_fwd(self.g, set_conf)
        self.G = G
        mse = _mean_square_error(np.abs(G), self.absF, self.mse_denom)

        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = mft_rev(Gprime, set_conf)
        phs_gprime = np.angle(gprime)
        gprimeprime = self.absg * np.exp(1j*phs_gprime)

        self.costF.append(mse)
        self.iter += 1
        self.g = gprimeprime
        return gprimeprime
    
class ConjugateGradientFixedSampling:
    """Conjugate Gradient iterator, using matrix fourier transforms

    Notes
    -----
    Details about this algorithm can be found in Fienup (1982)
    https://doi.org/10.1364/AO.21.002758
    """
    def __init__(self, psf, pupil_amplitude, phase_guess=None, hk=1):
        _init_iterative_transform_fixedsampling(self, psf, pupil_amplitude, phase_guess)
        self.gprimekm1 = self.g
        self.hk = hk

    def step(self):
        """Advance the algorithm one iteration."""

        log.debug(sys._getframe().f_code.co_name)
        G = mft_fwd(self.g, set_conf)
        mse = _mean_square_error(np.abs(G), self.absF, self.mse_denom)
        Bk = mse
        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = mft_rev(Gprime, set_conf)

        gprimeprime = gprime + self.hk * (gprime - self.gprimekm1)

        # finally, apply the object domain constraint
        phs_gprime = np.angle(gprimeprime)
        gprimeprime = self.absg * np.exp(1j*phs_gprime)

        self.costF.append(mse)
        self.iter += 1
        self.Bkm1 = Bk  # bkm1 = "B_{k-1}"; B for iter k-1
        # self.Dkm1 = D
        self.gprimekm1 = gprime
        self.g = gprimeprime
        return gprimeprime


"""FFT-based Phase Retrieval

The following classes utilize FFTs for propagation. The GerchbergSaxton and ConjugateGradient classes generally underperform compared to their FixedSampling 
equivalents, so we prefer that users default to those for single-plane iterators. The FocusDiversePhaseRetrieval is the default algorithm for multi-plane
iteration, and uses Misel's algorithm generalized to N PSFs

"""
class FocusDiversePhaseRetrieval:
    """Focus Diversity Phase Retrieval using Gerchberg-Saxton-like iteration.

    Algorithm inspired by Misel's two-psf algorithm [1], generalized to N psfs
    - [1] D L Misell 1973 J. Phys. D: Appl. Phys. 6 2200
    """

    def __init__(self,psflist,wvl,dxs,defocus_positions,phase_guess=None):
        """Phase Retrieval Iterator using Focus Diversity for N defocus positions

        Parameters
        ----------
        psflist : list of numpy.ndarrays of the same shape
            length N list of numpy.ndarrays that contain the defocused PSF data. Must be of the same pixel scale
            and array size
        wvl : float
            wavelength of light in microns
        dxs : float
            pixel scale of the arrays in psflist in microns
        defocus_positions : list of floats
            defocus positions in microns
        phase_guess : numpy.ndarray, optional
            phase guess of the desired pupil sampling, by default None
        """
        self.log = log
        # catch some common mistakes
        assert len(defocus_positions) == len(dxs), f"defocus_positions and dxs should have the same length, got {len(defocus_positions)} and {len(dxs)}"
        assert (len(psflist) == len(dxs)+1) and (len(psflist) == len(defocus_positions)+1), f"psflist should be one element longer than dxs and defocus_positions, got {len(psflist)}"

        try:
            if phase_guess is None:
                phase_guess = np.random.rand(*psflist[0].shape)

            self.absFlist = []
            self.mse_denom = []

            # TODO: Throw a try-except

            # Create the object domain data in field units
            for psf in psflist:
                self.absFlist.append(np.fft.ifftshift(np.sqrt(psf)))
                self.mse_denom.append(np.sum(psf))

            # Begin with a guess using the first PSF
            phase_guess = np.fft.ifftshift(phase_guess)
            self.G0 = self.absFlist[0] * np.exp(1j*phase_guess)
            
            # pre-compute transfer functions, lists of kernels
            self.forward_prop = []
            self.backward_prop = []
            self.cost_functions = [] # will be a list of lists
            for dz,dx in zip(defocus_positions,dxs):
                self.forward_prop.append(_angular_spectrum_transfer_function(psflist[0].shape,wvl,dx,dz)) # there was a 1e-3 factor here
                self.backward_prop.append(_angular_spectrum_transfer_function(psflist[0].shape,wvl,dx,-dz))
                self.cost_functions.append([])

            self.iter = 0

        except Exception as e:
            self.log.critical(f'Error in initializing iterator: \n {e}')

    def step(self):
        """use Misel's algorithm to perform an iteration between image space and the fourier plane

        Returns
        -------
        G0primeprime
            updated estimate of the image plane electric field
        """

        log.debug(sys._getframe().f_code.co_name)
        
        for i,(fwd,rev,absF1,mse_denom) in enumerate(zip(self.forward_prop,self.backward_prop,self.absFlist[1:],self.mse_denom)):

            G1 = _angular_spectrum_prop(self.G0,fwd)
            phs_G1 = np.angle(G1)
            G1prime = absF1 * np.exp(1j*phs_G1)
            G0prime = _angular_spectrum_prop(G1prime,rev)
            phs_G0prime = np.angle(G0prime)
            # G0primeprime = self.absFlist[0] * np.exp(1j*phs_G0prime)
            G0primeprime = self.absFlist[0] * np.exp(1j*phs_G0prime)

            # remember to update the phase guess for PSF
            self.G0 = G0primeprime
            self.cost_functions[i].append(_mean_square_error(np.abs(G0prime),self.absFlist[0],norm=mse_denom))
            self.iter += 1

        # return pupil_estimate
        # pupil_estimate = np.fft.ifftshift(np.fft.ifft2(G0primeprime))

        return np.fft.fftshift(G0primeprime)

class GerchbergSaxton:
    """Gerchberg Saxton phase retrieval algorithm

    Notes
    -----
    Details about this algorithm can be found in Fienup (1982)
    https://doi.org/10.1364/AO.21.002758
    """
    def __init__(self,psf,pupil_amplitude,phase_guess=None):
        _init_iterative_transform(self, psf, pupil_amplitude, phase_guess)

    def step(self):
        """Advance the algorithm one iteration."""

        log.debug(sys._getframe().f_code.co_name)

        G = fft.fft2(self.g)
        mse = _mean_square_error(abs(G), self.absF, self.mse_denom)

        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = fft.ifft2(Gprime)
        phs_gprime = np.angle(gprime)
        gprimeprime = self.absg * np.exp(1j*phs_gprime)

        self.costF.append(mse)
        self.iter += 1
        self.g = gprimeprime
        return gprimeprime
    
class ConjugateGradient:
    """Conjugate Gradient phase retrieval algorithm

    Notes
    -----
    Details about this algorithm can be found in Fienup (1982)
    https://doi.org/10.1364/AO.21.002758
    """
    def __init__(self, psf, pupil_amplitude, phase_guess=None, hk=1):
        _init_iterative_transform(self, psf, pupil_amplitude, phase_guess)
        self.gprimekm1 = self.g
        self.hk = hk

    def step(self):
        """Advance the algorithm one iteration"""

        log.debug(sys._getframe().f_code.co_name)

        G = fft.fft2(self.g)
        mse = _mean_square_error(abs(G), self.absF, self.mse_denom)
        Bk = mse
        phs_G = np.angle(G)
        Gprime = self.absF * np.exp(1j*phs_G)
        gprime = fft.ifft2(Gprime)

        gprimeprime = gprime + self.hk * (gprime - self.gprimekm1)

        # finally, apply the object domain constraint
        phs_gprime = np.angle(gprimeprime)
        gprimeprime = self.absg * np.exp(1j*phs_gprime)

        self.costF.append(mse)
        self.iter += 1
        self.Bkm1 = Bk  # bkm1 = "B_{k-1}"; B for iter k-1
        # self.Dkm1 = D
        self.gprimekm1 = gprime
        self.g = gprimeprime
        return gprimeprime
    
# Try nonlinear optimization with algorithmic differentiation
class NLOptPRModel:
    """Nonlinear optimization phase retrieval algorithm, no phase diversity

    FIXME: This algorithm doesn't perform consistently in simulation and hasn't worked on a testbed _yet_

    Notes
    -----
    This algorithm is from Jurling and Fienup (2014) 
    https://doi.org/10.1364/JOSAA.31.001348 

    """
    def __init__(self, amp, wvl, basis, data):
        """Nonlinear Optimization Phase Retrieval w/ algorithmic differentiation

        TODO: Implement focus diversity

        Parameters
        ----------
        amp : numpy.ndarray
            pupil amplitude
        wvl : float
            wavelength of light
        basis : numpy.ndarray
            array containing the basis index in the first dimension, and pixel index in the last two
        data : numpy.ndarray
            PSF to reconstruct the phase of
        """

        self.amp = amp
        self.wvl = wvl
        self.basis = basis
        self.D = data
        self.D /= np.sum(self.D)
        self.hist = []
        self.costF = []

    def logcallback(self, x):
        self.hist.append(x.copy())

    def update(self, x):
        W = np.tensordot(self.basis, x, axes=(0,0))
        W *= 2 * np.pi / self.wvl
        g = self.amp * np.exp(1j * W)
        G = ft_fwd(g)
        I = np.abs(G)**2
        I /= np.sum(I)
        # I *= self.total_energy
        E = _mean_square_error(I,self.D,norm=1)
        self.W = W
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        self.costF.append(E)

    def fwd(self, x):
        """Advance the algorithm one iteration"""
        self.update(x)
        return self.E

    def rev(self, x):
        """reverse-mode algorithmic differentiation for computing the gradient of the error metric
        with respect to the basis coefficients. Used with scipy.optimize.minimize
        
        nlo = NLOptPRModel(pupil_amplitude,wavelength,modal_basis,psf_data)
        results = minimize(nlo.fwd,jac=nlo.rev)

        not passing nlo.rev to the jac kwarg results in the use of finite differences, which also works
        
        """
        self.update(x)
        Ibar = 2 * (self.I - self.D)
        Gbar = 2 * Ibar * self.G
        gbar = ft_rev(Gbar)
        Wbar = np.imag(gbar * np.conj(self.g))
        Wbar *= 2 * np.pi / self.wvl
        abar = np.tensordot(self.basis, Wbar)
        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.Wbar = Wbar
        self.abar = abar
        return self.abar



