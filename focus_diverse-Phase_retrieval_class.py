
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
            self.cost_functions[i].append(mean_squared_error(np.abs(G0prime),self.absFlist[0],norm=mse_denom))
            self.iter += 1

        # return pupil_estimate
        # pupil_estimate = np.fft.ifftshift(np.fft.ifft2(G0primeprime))

        return np.fft.fftshift(G0primeprime)