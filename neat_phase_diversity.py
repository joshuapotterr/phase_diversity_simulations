import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from skimage.restoration import unwrap_phase
from skimage.transform import resize
from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
from processing import phase_unwrap_2d
# pep8 styl


# Convert phase into meters using the wavelength
def phase_to_m(phase, wv):
    return phase * wv / (2 * np.pi)

# Calculate defocus distance from peak-to-valley (P2V) error in meters
def p_to_delta(P, f, D):
    return 8 * P * (f/D)**2

# Convert defocus distance into phase error
def delta_to_p(delta, f, D):
    return -1 * delta / (8 * (f/D)**2)

#Build the Seal Simulation

def build_seal_simulation(seal_parameters):
    """
    Build and return the SEAL optical simulation components.

    This function initializes all necessary grids, apertures, and modal bases
    (Zernike and Fourier) for running SEAL simulations, including the focal and 
    pupil grids, telescope and masking pupils, and modal templates.

    Parameters:
        seal_parameters (dict): Dictionary of SEAL system configuration parameters.
            Required keys:
                - 'q' (int): Oversampling factor for focal grid.
                - 'Num_airycircles' (int): Number of Airy rings to include in focal grid.
                - 'pupil_size' (float): Diameter of the telescope pupil (in meters).
                - 'focal_length_meters' (float): Effective focal length (in meters).
                - 'wavelength_meter' (float): Wavelength of light (in meters).
                - 'pupil_pixel_dimension' (int): Grid resolution for pupil.
                - 'small_pupil_size_meter' (float): Diameter of the masking pupil.

    Returns:
        dict: A dictionary named `simulation_elements` containing the following:
            - 'pupil_grid': HCIPy PupilGrid object.
            - 'focal_grid': HCIPy FocalGrid object.
            - 'aperture': Callable function to create circular aperture.
            - 'telescope_pupil': 2D array of the telescope pupil mask.
            - 'masking_pupil': 2D array of the masking pupil mask.
            - 'zernike_sample_12': List of the first 12 Zernike modes (2D shaped).
            - 'fourier_sample_12': List of 12 Fourier modes (2D shaped).

    Notes:
        - Zernike modes are computed using HCIPy's `make_zernike_basis`.
        - Fourier modes are generated for specific low-order (kx, ky) spatial frequencies.
        - All shaped outputs match the pupil grid shape for direct phase application.
    """
    

    #Create a focal grid based on system parameters
    #focal_grid = make_focal_grid(q=q, num_airy=num_airy, pupil_diameter=pupil_size, focal_length=focal_length, reference_wavelength=wavelength)
    focal_grid = make_focal_grid(
        q =seal_parameters['q'], 
        num_airy=seal_parameters['Num_airycircles'], 
        pupil_diameter = seal_parameters['pupil_size'], 
        focal_length=seal_parameters['focal_length_meters'], 
        reference_wavelength =seal_parameters['wavelength_meter'] 
        )
    # Setup apertures for telescope and masking pupils
    pupil_grid = make_pupil_grid(
        seal_parameters['pupil_pixel_dimension'],
        seal_parameters['pupil_size']
    )
    
    aperture = make_circular_aperture(seal_parameters['pupil_size'])

    telescope_pupil = aperture(pupil_grid)

    small_aperture = make_circular_aperture(seal_parameters['small_pupil_size_meter'])

    masking_pupil = small_aperture(pupil_grid)
    
    #Make Zernike to store
    zernike_modes = make_zernike_basis(
         num_modes=12,
         D=seal_parameters['pupil_size'],
         grid=pupil_grid
    )
    zernike_sample_12 = [mode.shaped for mode in zernike_modes[:12]]
    
    freq_pairs = [(i, 0) for i in range(1, 7)] + [(0, i) for i in range(1, 7)]  # 12 total
    kx = np.array([kx * 2 * np.pi / seal_parameters['pupil_size'] for kx, _ in freq_pairs])
    ky = np.array([ky * 2 * np.pi / seal_parameters['pupil_size'] for _, ky in freq_pairs])
    fourier_coords = UnstructuredCoords((kx, ky))
    fourier_grid = CartesianGrid(fourier_coords)
    fourier_basis = make_fourier_basis(pupil_grid, fourier_grid)
    fourier_sample_12 = [mode.shaped for mode in fourier_basis]

    # Return all components as a dictionary
    simulation_elements = {
        'pupil_grid': pupil_grid,
        'focal_grid': focal_grid,
        'aperture': aperture,
        'telescope_pupil': telescope_pupil,
        'masking_pupil': masking_pupil,
        'zernike_sample_12' : zernike_sample_12,
        'fourier_sample_12' : fourier_sample_12 }
    

    return simulation_elements

##Core for Phase Retrieval
##########################

def convert_psf_estimate_to_phase(psf_estimate, 
                                  seal_parameters, 
                                  telescope_pupil, 
                                  phase_unwrap=None):
    """
    Convert a PSF estimate into a pupil phase map using inverse Fourier optics.

    This function performs an inverse modified Fourier transform (MFT) on the PSF estimate 
    to recover the complex electric field in the pupil plane, extracts the phase, optionally
    unwraps it, resizes to match the telescope pupil grid, and applies the pupil mask.

    Parameters:
        psf_estimate (ndarray): Estimated point spread function (PSF), assumed to be in the focal plane.
        seal_parameters (dict): Configuration dictionary containing SEAL optical system parameters.
            Required key: 'pupil_pixel_dimension' (int) for the desired pupil output size.
        telescope_pupil (Field or ndarray): The shaped pupil mask to apply to the final phase map.
        phase_unwrap (str or callable, optional): Phase unwrapping method to apply.
            Options:
                - "phase_unwrap_2d": Use custom 2D phase unwrapping function from `processing.py`.
                - "unwrap_phase": Use `skimage.restoration.unwrap_phase`.
                - callable: Any user-defined unwrapping function that accepts a 2D array.

    Returns:
        pupil_phase : ndarray
        2D array of the unwrapped and pupil-masked phase map of the retrieved pupil field (in radians).

    Notes:
        - The MFT is computed using `mft_rev()` as defined in the imaging pipeline.
        - The result is resized to the SEAL pupil grid dimensions.
        - The phase is masked by the telescope pupil to zero outside the aperture.
    """
    #retrieve the grid size
    pupil_dim = seal_parameters['pupil_pixel_dimension']
    #configure instrument for inverse transform
    seal_configuration = InstrumentConfiguration(seal_parameters)
    #compute complex field via reverse MFT, extract phase
    raw_pupil_phase = np.angle(mft_rev(psf_estimate,seal_configuration)) 

    #choose unwrap method
    if phase_unwrap == "phase_unwrap_2d":
        phase_unwrap = phase_unwrap_2d
    elif phase_unwrap == "unwrap_phase":
        phase_unwrap = unwrap_phase
    
    if phase_unwrap is not None:
        raw_pupil_phase = phase_unwrap(raw_pupil_phase)
    #resize to the pupil dimensions and mask the outside

    pupil_phase = resize(raw_pupil_phase, (pupil_dim, pupil_dim)) * telescope_pupil.shaped

    return pupil_phase

def check_phase_estimate(system_truth_phase, phase_estimate,masking_pupil):
    """
    Evaluate the accuracy of an estimated pupil phase map against the ground truth.

    This function compares the estimated phase map to the known system truth phase
    by computing pixel-wise differences within the region defined by a pupil mask.
    It returns error metrics such as RMS (root-mean-square) error and peak-to-valley (P2V) error,
    as well as diagnostic difference images.

    Parameters
    ----------
    system_truth_phase : Wavefront
        HCIPy Wavefront object containing the true phase in its `.phase` attribute.
    phase_estimate : ndarray
        2D array representing the estimated pupil phase (in radians).
    masking_pupil : Field or ndarray
        Binary mask defining the region of interest for error evaluation; pixels outside this mask are excluded from RMS.

    Returns
    -------
    dict
        Dictionary with the following keys:
            - 'rms_error' : float
                Root-mean-square error between truth and estimate (masked).
            - 'p2v_error' : float
                Peak-to-valley error over the full difference image.
            - 'difference_image' : ndarray
                Masked phase difference values (truth - estimate).
            - 'difference_true_vs_estimate' : ndarray
                Full, unmasked phase difference map.

    Notes
    -----
    - The mask ensures that the RMS error is computed only within the clear pupil region.
    - P2V is calculated on the unmasked difference (may include edge effects).
    - Optionally, the median can be subtracted from each phase map before comparison to remove piston terms.
    """
    true_phase = system_truth_phase.shaped # Get true phase
    mask = np.array(masking_pupil.shaped, dtype =bool)# Apply mask to non-zero phase region
    #implement med_subtracted, ie passing the dictionary will help with this(pupil_phase - median blah blah)
    difference_true_vs_estimate = (true_phase - phase_estimate) #Compute difference
    difference_masked = difference_true_vs_estimate[mask]
    rms_error = np.sqrt(np.mean(difference_masked ** 2))
    ##do i want difference_masked? 
    p2v_error = np.max(difference_true_vs_estimate) - np.min(difference_true_vs_estimate)
    return {'rms_error': rms_error, 
            'p2v_error': p2v_error, 
            'difference_image':difference_masked,
            'difference_true_vs_estimate':difference_true_vs_estimate}#also return the difference image
    
def make_cost_functions_plots(cost_functions, filename=None):
    """
    Plot the convergence of cost functions over optimization iterations for each defocus input.

    This function generates a semilogarithmic plot showing how the cost function decreases 
    with each iteration during phase retrieval. Each line corresponds to a different defocus 
    configuration. Useful for assessing convergence behavior.

    Parameters:
        cost_functions (list of list of float): 
            A list where each element is a list of cost values across iterations for one defocus input.
        filename (str, optional): 
            If provided, saves the figure to the specified path. Otherwise, the figure is displayed in memory.

    Returns:
        None

    Notes:
        - Uses `plt.semilogy()` for better visualization of convergence.
        - Clears the figure after plotting to avoid overlap in future plots.
    """
    plt.clf()
    for idx, costs in enumerate(cost_functions, start =1):
        plt.semilogy(costs, label=f'defocus {idx}')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.legend()
    plt.grid(True)
    if filename: 
        plt.savefig(filename)#pass the name as an argument to the function
    #before starting new plot, use plt.clf to clear and it doesnt overlap
    plt.clf()

def run_phase_retrieval(system_truth_intensity,
                        defocus_dictionary, 
                        seal_parameters,
                        num_iterations=200):
    """
    Perform iterative phase retrieval using focus-diverse PSFs.

    Runs the phase retrieval algorithm by combining a focused PSF with a set of defocused PSFs. 
    The algorithm estimates the pupil phase that best explains the observed intensity distribution, 
    using the FocusDiversePhaseRetrieval class and tracking the convergence of the cost function.

    Parameters
    ----------
    system_truth_intensity : ndarray
        Focused (zero-defocus) PSF used as the reference for phase retrieval.
    defocus_dictionary : dict
        Mapping from defocus distance (float, in meters) to defocused PSF (ndarray).
    seal_parameters : dict
        Dictionary of SEAL optical system parameters. Must include 'image_dx' and 'wavelength_meter'.
    num_iterations : int, optional
        Number of optimization iterations to run (default is 200).

    Returns
    -------
    psf_estimate : ndarray
        Estimated complex PSF after phase retrieval.
    cost_functions : list of list of float
        Convergence history of the cost function for each defocus input.

    Notes
    -----
    - The FocusDiversePhaseRetrieval class must implement a .step() method for phase update and cost tracking.
    - The focused PSF is prepended to the list of PSFs as a reference.
    - The returned cost_functions can be used to assess convergence.
    """

    distance_list = list(defocus_dictionary.keys()) 
    #could use sorted here instead, psfs and sitances may mismatch unless order
    #is guaranteed by construction

    #This needs to get passed ONLY numpy arrays, it checks the first element of psf_list and uses its shape to construct the starting guess
    psf_list = [system_truth_intensity] +[defocus_dictionary[key] for key in distance_list]
    print('psf_list:',psf_list)
    print('type of psf_list', type(psf_list))
    dx_list= [seal_parameters['image_dx'] for key in distance_list]
    #consider asserting, "assert all(isinstance(seal_parameters['image_dx'], (float, int)))

    mp= FocusDiversePhaseRetrieval(psf_list,
                                   seal_parameters['wavelength_micron'], #JARENS NOT IN METER
                                   dx_list, 
                                   distance_list)

    for i in range(num_iterations):
            psf_estimate = mp.step() ##does this for loop overwrite psf_estimate each time
    '''
    all_estimates = []
    for i in range(num_iterations):
        psf_estimate = mp.step()
        all_estimates.append(psf_estimate)
    return all_estimates, mp.cost_functions
    '''
    

    return psf_estimate, mp.cost_functions

def calculate_phase_retrieval_accuracy(
        system_truth_phase, 
        psf_estimate, 
        cost_functions, 
        seal_parameters, 
        simulation_elements,
        phase_unwrap_method = None,
        verbose = False
        ):
    """
    Evaluate the accuracy of phase retrieval by comparing estimated and ground truth pupil phases.

    This function reconstructs the pupil phase from the retrieved PSF, compares it to the ground truth phase,
    and computes error metrics such as RMS and peak-to-valley (P2V) error. Optionally, it can also plot the 
    cost function convergence history for diagnostic purposes.

    Parameters
    ----------
    system_truth_phase : Wavefront
        The known wavefront used as the ground truth for comparison.
    psf_estimate : ndarray
        The estimated PSF obtained from the phase retrieval algorithm.
    cost_functions : list of list of float
        Convergence history of the cost function across iterations and defocus inputs.
    seal_parameters : dict
        Dictionary of SEAL system configuration parameters (e.g., pupil size, image resolution).
    simulation_elements : dict
        Dictionary of precomputed simulation elements, including:
            - 'masking_pupil' : mask for error computation
            - 'telescope_pupil' : mask for phase reconstruction
    phase_unwrap_method : str or callable, optional
        Phase unwrapping method to use. Options:
            - "phase_unwrap_2d"
            - "unwrap_phase"
            - callable (user-defined)
            Default is None (no unwrapping).
    verbose : bool, optional
        If True, plots the cost function convergence history.

    Returns
    -------
    dict
        Dictionary containing:
            - 'rms_error' : float
                Root-mean-square phase difference over the pupil.
            - 'p2v_error' : float
                Peak-to-valley phase difference.
            - 'difference_image' : ndarray
                Masked residual phase map.
            - 'difference_true_vs_estimate' : ndarray
                Full residual phase map (unmasked).

    Notes
    -----
    - Internally calls convert_psf_estimate_to_phase() and check_phase_estimate().
    - The phase unwrapping method must correspond to a callable if specified.
    - Cost function history is only plotted if verbose=True.
    """

    phase_estimate = convert_psf_estimate_to_phase(psf_estimate, 
                                                   seal_parameters,
                                                   simulation_elements['telescope_pupil'],
                                                   phase_unwrap_method)# Reconstruct phase from PSF
    
    phase_estimate_metrics = check_phase_estimate(system_truth_phase, 
                                                  phase_estimate,
                                                  simulation_elements['masking_pupil'])# Compare with truth

    if verbose:
        make_cost_functions_plots(cost_functions)

    return phase_estimate_metrics

##Simulations
#############

def simulate_focused_image(wf_error_to_retrieve, 
                          simulation_elements, 
                          seal_parameters):
    """
    Simulate a focused wavefront image using a specified phase error.

    Generates a complex-valued wavefront by applying the provided phase error to the telescope pupil mask,
    and returns the resulting focused intensity image, phase, and full wavefront object.

    Parameters
    ----------
    wf_error_to_retrieve : ndarray
        Phase error to apply to the wavefront (in radians). Should be broadcastable to the pupil shape.
    simulation_elements : dict
        Dictionary containing precomputed simulation components, must include:
            - 'telescope_pupil' : ndarray
                Circular pupil aperture mask.
    wavelength : float
        Wavelength of the simulated light, in meters.

    Returns
    -------
    wf_focused_intensity : ndarray
        2D array representing the intensity distribution of the focused wavefront.
    wf_focused_phase : ndarray
        2D array of the phase map of the focused wavefront.
    wf_focused : Wavefront
        The complex-valued wavefront object after applying the phase error.

    Notes
    -----
    - The input phase error is flattened to match the expected format for constructing the Wavefront.
    - Returned intensity is shaped to match the pupil grid and can be used in downstream phase retrieval.
    """
    telescope_pupil=simulation_elements['telescope_pupil']
    wf_focused = Wavefront(telescope_pupil * np.exp(1j * wf_error_to_retrieve.flatten()), 
                   seal_parameters['wavelength_meter'])
    wf_focused_intensity = wf_focused.intensity
    wf_focused_phase = wf_focused.phase
    #assert simulation_elements['telescope_pupil'].shape == wf_error_to_retrieve.shape, \
    "Wavefront error and telescope pupil shape mismatch"
    #.intensity gives us our actual image, and .shaped formats it into an ndarray in order to pass to FDPR
    #Does this need to get propagated to Focal using Fraunhofer?
    #psf_list output is full of zeros and i only get 2 non-zero returns

    return wf_focused_intensity.shaped, wf_focused_phase, wf_focused
    

def simulate_defocused_image(defocus_phase,
                             wf_error_to_retrieve, 
                             seal_parameters, 
                             simulation_elements,
                             ):#instead of defocus_phase pass wf,test_abberattion
    """
    Simulate a defocused point spread function (PSF) by propagating a wavefront with combined aberrations.

    This function combines a specified phase error and a defocus phase map, applies them to the telescope pupil,
    and simulates propagation to the focal plane using Fraunhofer diffraction. The resulting PSF intensity 
    image is returned, resized to match the pupil grid.

    Parameters
    ----------
    defocus_phase : ndarray
        2D array representing the defocus aberration (in radians); must match the pupil shape.
    wf_error_to_retrieve : ndarray
        2D array of static wavefront phase error to retrieve (in radians); must match the pupil shape.
    seal_parameters : dict
        Dictionary of SEAL system configuration, must include:
            - 'focal_length_meters' : float
                Effective focal length, in meters.
            - 'pupil_pixel_dimension' : int
                Output grid size for resizing.
            - 'wavelength_meter' : float
                Wavelength, in meters.
    simulation_elements : dict
        Dictionary containing optical components, including:
            - 'telescope_pupil' : ndarray
                Pupil mask.
            - 'pupil_grid' : PupilGrid
                Grid over which the pupil mask is defined.
            - 'focal_grid' : FocalGrid
                Grid over which the focal plane PSF is computed.

    Returns
    -------
    focal_intensity : ndarray
        2D array representing the intensity of the simulated defocused PSF, resized to match the pupil grid.

    Notes
    -----
    - The wavefront is constructed as the product of the pupil mask and a complex exponential of the combined phase errors.
    - Fraunhofer propagation is performed using the provided pupil and focal grids.
    - The output intensity is shaped and resized to the pupil grid dimensions for consistency.
    """
    #Use dictionary again
    
    aberration_to_impart = wf_error_to_retrieve + defocus_phase

    wavefront_defocused = Wavefront(simulation_elements['telescope_pupil'] * np.exp(1j * aberration_to_impart.flatten()),
                           seal_parameters['wavelength_meter'])
    #assert aberration_to_impart.shape == simulation_elements['telescope_pupil'].shape

    
    prop2f = FraunhoferPropagator(simulation_elements['pupil_grid'],
                                    simulation_elements['focal_grid'],
                                    focal_length=seal_parameters['focal_length_meters']
                                    )
    aberrated_psf = prop2f(wavefront_defocused) # Propagate to focal plane, no dict

    focal_intensity = aberrated_psf.intensity
    resize_256 = (seal_parameters['pupil_pixel_dimension'], seal_parameters['pupil_pixel_dimension'])
    focal_intensity = resize(focal_intensity.shaped, resize_256)
    print('focal_intensity shape is :', focal_intensity.shape)

    return focal_intensity # Return intensity image, using shaped for the return as a proper array

def calculate_defocus_phase(seal_parameters,
                            simulation_elements,
                            defocus_distance):
    """
    Generate a pupil-plane defocus phase map for a given physical defocus distance.

    This function constructs a defocus phase aberration (in radians) by scaling the normalized
    Zernike defocus mode (Zernike index 3) to match the specified physical defocus distance.
    The scaling factor is determined using the optical configuration and the delta_to_p formula.

    Parameters
    ----------
    seal_parameters : dict
        Optical system parameters. Must include:
            - 'focal_length_meters' (float): System focal length in meters.
            - 'pupil_size' (float): Pupil diameter in meters.
    simulation_elements : dict
        Simulation resources. Must include:
            - 'zernike_sample_12' (list of ndarray): The first 12 shaped Zernike modes.
    defocus_distance : float
        Physical defocus distance in meters (can be positive or negative).

    Returns
    -------
    defocus_phase : ndarray
        2D array (same shape as the pupil) representing the defocus phase aberration in radians.

    Notes
    -----
    - The Zernike mode used is always index 3, corresponding to defocus.
    - The phase scaling uses the peak-to-valley (P2V) value of both the template and the target defocus.
    - This function requires delta_to_p to convert physical defocus to phase P2V.
    """

    defocus_template =simulation_elements['zernike_sample_12'][3]

    template_p2v=np.max(defocus_template)-np.min(defocus_template) 
    #Convert physical defocus distance to phase P2V using delta_to_p()
    defocus_p2v = delta_to_p(
                            delta = defocus_distance,
                            f = seal_parameters['focal_length_meters'],
                            D=seal_parameters['pupil_size']
                            )
    defocus_phase = (defocus_template*defocus_p2v)/template_p2v
    
    return defocus_phase



def focus_diverse_phase_retrieval(system_truth,
                                  phase_diverse_information, 
                                  wf_error_to_retrieve,  
                                  seal_parameters,
                                  simulation_elements):

    """
    Perform phase retrieval using a focused PSF and one or more defocused PSFs.

    This function simulates defocused point spread functions (PSFs) for a set of specified defocus distances,
    then runs a phase retrieval algorithm using both the focused (ground truth) PSF and the simulated defocused PSFs.
    The goal is to estimate the pupil phase that best explains the observed intensity distributions.

    Parameters
    ----------
    system_truth : Wavefront 
        The focused (zero-defocus) point spread function, representing the ground truth.
    phase_diverse_information : dict
        Dictionary mapping each defocus distance (float, meters) to its corresponding defocus phase map (2D ndarray, radians).
    wf_error_to_retrieve : ndarray
        The static wavefront phase error map to be retrieved (2D array, radians).
    seal_parameters : dict
        Dictionary of optical and simulation parameters (e.g., focal length, pupil size).
    simulation_elements : dict
        Dictionary containing simulation resources such as aperture masks, grids, and Zernike modes.

    Returns
    -------
    psf_estimate : ndarray
        The estimated complex PSF after phase retrieval.
    cost_functions : list
        List tracking the cost function (error metric) over optimization iterations.

    Notes
    -----
    - For each defocus distance, a defocused PSF is simulated using the provided phase map and wavefront error.
    - All PSFs (focused and defocused) are then provided to the phase retrieval algorithm.
    - The cost_functions output can be used to assess convergence or algorithm performance.
    """

    print("TYPE OF phase_diverse_information:", type(phase_diverse_information))
    print("CONTENTS:", phase_diverse_information)

    defocus_distances = phase_diverse_information.keys()
    #change to allow mulitple inptus
    defocus_psfs={}
    for defocus_distance in defocus_distances:

        defocus_phase = phase_diverse_information[defocus_distance]
        
        
        defocused_psf  = simulate_defocused_image(
            defocus_phase,
            wf_error_to_retrieve,
            seal_parameters,
            simulation_elements
        )
        #put into disctionary keyed by defocus distance
        defocus_psfs[defocus_distance]=defocused_psf

        #run it
    psf_estimate, cost_functions = run_phase_retrieval(
            system_truth,
            defocus_psfs,
            seal_parameters
        )

    return psf_estimate, cost_functions





def simulate_phase_diversity_grid(wf_error_to_retrieve,
                                  phase_diverse_inputs, 
                                  seal_parameters, 
                                  file_name_out, 
                                  simulation_elements, 
                                  wavelength):
    """
    Simulate phase diversity experiments across a grid of defocus configurations and evaluate phase retrieval accuracy.

    This function iterates over a 2D grid of phase diversity input configurations, simulating the corresponding focused and defocused
    point spread functions (PSFs), performing phase retrieval for each configuration, and calculating the accuracy (e.g., RMS error)
    of the retrieved phase compared to the known ground truth.

    Parameters
    ----------
    wf_error_to_retrieve : ndarray
        The static wavefront phase error map to be retrieved, in radians.
    phase_diverse_inputs : list or array-like
        A 2D grid (list of lists, or similar) where each entry contains a tuple: (indices..., defocus_distances...),
        with the first half being grid indices and the second half being defocus distances (float, meters).
    seal_parameters : dict
        Dictionary of optical and simulation parameters (e.g., focal length, pupil size, pixel dimensions).
    file_name_out : str
        Path or filename for saving the simulation results.
    simulation_elements : dict
        Dictionary containing simulation resources such as aperture masks, grids, and Zernike modes.
    wavelength : float
        Wavelength of the simulated light (in meters).

    Returns
    -------
    phase_diversity_grid : ndarray
        2D array of RMS phase retrieval errors for each grid point.

    Notes
    -----
    - For each grid point, the function simulates a focused PSF and one or more defocused PSFs using the supplied aberrations.
    - Phase retrieval is performed for each configuration, and the accuracy is evaluated (e.g., by comparing RMS error).
    - The resulting phase_diversity_grid is saved to file_name_out and also returned.
    """

    #would be origin, breaking step by step
    
    ##Step 1: simulate focused system truth, everything is getting passed from main
    system_truth_intensity, system_truth_phase, system_truth = simulate_focused_image(wf_error_to_retrieve,
                                          simulation_elements,
                                          wavelength) 

    ##Step2: empty error grid
    dim = seal_parameters['grid_dim'] 
    phase_diversity_grid = np.zeros((dim,dim))


    ##Step 3: loop through each focus input
    #simulation_specfics would be one of the tuples in phase dverse info
    for simulation_specifics in phase_diverse_inputs:
        n_info = int((len(simulation_specifics))/2)
        indices = simulation_specifics[:n_info]
        defocus_distances = simulation_specifics[n_info:]
        #indices first half
        '''    
        index_x, index_y, defocus_dict = simulation_specifics
        indices = (index_x, index_y)
        defocus_distances = defocus_dict.keys()
        '''
        #defocus_distance is second half
        
    #for index_x in phase_diverse_inputs.shape[0]:
        defocus_dictionary= {}
        #this takes second two information in tuple of simualtion_specifics
        for defocus_distance in defocus_distances:

            defocus_phase = calculate_defocus_phase(seal_parameters,
                                                    simulation_elements,
                                                    defocus_distance)
            defocus_dictionary[defocus_distance] = defocus_phase
        print("TYPE OF defocus_dictionary:", type(defocus_dictionary))
        print("KEYS:", defocus_dictionary.keys() if hasattr(defocus_dictionary, 'keys') else 'not a dict')

        psf_estimate, cost_functions = focus_diverse_phase_retrieval(system_truth_intensity, 
                                                                     defocus_dictionary,
                                                                     wf_error_to_retrieve, 
                                                                     seal_parameters, 
                                                                     simulation_elements)
        
        metrics = calculate_phase_retrieval_accuracy(system_truth_phase, 
                                                     psf_estimate, 
                                                     cost_functions, 
                                                     seal_parameters,
                                                     simulation_elements,
                                                     phase_unwrap_method = 'phase_unwrap_2d', 
                                                     verbose =True)
        #results.append(metrics)
        #indices is first half of tuple
        #need to add index checks and make sure indices are within bounds
        
        phase_diversity_grid[indices] = metrics['rms_error']

    np.save(file_name_out, phase_diversity_grid) #will assign name when function runs
    print("phase_diversity_grid stats:")
    print("Min:", np.min(phase_diversity_grid))
    print("Max:", np.max(phase_diversity_grid))
    print("Nonzero count:", np.count_nonzero(phase_diversity_grid))
    print("Shape:", phase_diversity_grid.shape)

    return phase_diversity_grid



def plot_phase_diversity_heat_map(phase_diversity_grid,
                                  heatmap_plot_out):
    """
    Plot and save a heatmap of RMS phase retrieval errors across a 2D defocus grid.

    This visualization is useful for assessing the effectiveness of various combinations of
    defocus distances used in phase diversity wavefront sensing.

    Parameters:
        phase_diversity_grid (ndarray):
            2D array where each entry corresponds to the RMS phase retrieval error
            for a specific defocus configuration (indexed by grid location).

        heatmap_plot_out (str):
            Filename to save the plotted heatmap (e.g., 'heatmap.png').

    Returns:
        None
    """
    plt.clf()

    extent = [-10, 10, -10, 10]  # assuming defocus in mm, update if different
    plt.imshow(phase_diversity_grid, origin='lower', extent=extent)



    plt.colorbar(label='RMS Error')

    plt.title('Phase Retrieval RMS Error Heatmap')

    plt.xlabel('Defocus Distance [mm]')

    plt.ylabel('Defocus Distance [mm]')

    plt.savefig(heatmap_plot_out)

    plt.clf()
    #keep in mind extent logic, need to tell physical x y labels 

##WIP##
##WIP##
def main(seal_parameters,
         phase_diverse_inputs,#might be defocus_grid
         file_name_out,  
         heatmap_plot_out,
         zernike_index = 6
         ): ##need help to implement this grid and using scales
    """
    Main driver function for SEAL phase diversity simulation and retrieval.

    This function:
    - Builds simulation elements.
    - Injects a known Zernike aberration into the wavefront.
    - Simulates a focused PSF.
    - Performs phase diversity retrieval over a grid of defocus configurations.
    - Generates and saves a heatmap of the RMS phase retrieval error.

    Parameters
    ----------
    seal_parameters : dict
        Configuration for SEAL optical system.
    phase_diverse_inputs : list
        Each entry should be a tuple (i, j, defocus_1, defocus_2, ...)
        indicating grid indices and defocus values.
    file_name_out : str
        File path to save the 2D phase retrieval error grid (.npy).
    heatmap_plot_out : str
        File path to save the resulting RMS heatmap figure.
    zernike_index : int, optional
        Index of the Zernike mode to inject as the known aberration.
    """

    # Build simulation elements like pupil grid, aperture, Zernike/Fourier modes, etc.
    simulation_elements = build_seal_simulation(seal_parameters)
    
    # Get the simulation wavelength
    wavelength = seal_parameters['wavelength_meter']

    # DEFINE ABERRATION TO RETRIEVE (ground truth) — outside the simulation functions
    zernike_modes = simulation_elements['zernike_sample_12']
    wf_error_to_retrieve = 0.75 * zernike_modes[zernike_index]

    # Simulate focused image using that known wavefront error
    system_truth_intensity, system_truth_phase, system_truth = simulate_focused_image(wf_error_to_retrieve,
                                          simulation_elements,
                                          wavelength)

    phase_diversity_grid= simulate_phase_diversity_grid(
            phase_diverse_inputs = phase_diverse_inputs,
            simulation_elements = simulation_elements,
            seal_parameters = seal_parameters,
            wavelength = wavelength,
            wf_error_to_retrieve = wf_error_to_retrieve,
            file_name_out =file_name_out)
    #Last function is to plot, i.e 'plot phase diversity heat map'
    plot_phase_diversity_heat_map(
        phase_diversity_grid = phase_diversity_grid,
        heatmap_plot_out=heatmap_plot_out
        )

   


if __name__ == "__main__": 
    #Define Variables
    seal_parameters = {
        'image_dx': 2.0071, # 
        'efl': 500, # SEAL effective focal length, mm
        'wavelength_micron': 0.65, # SEAL center wavelength, microns- >prysm
        'wavelength_meter': 650e-9,#SEAL center wavelength, meters -> hcipy
        'pupil_size': 10.12e-3, # Keck entrance pupil diameter
        'small_pupil_size_meter': 9.5e-3,
        'pupil_pixel_dimension': 256,
        'focal_length_meters': 500e-3,
        'q': 16,
        'Num_airycircles': 16,
        'grid_dim': 100
         }
    seal_parameters['wavelength']=seal_parameters['wavelength_meter']
    #some list of defocus distances 
    f = seal_parameters['focal_length_meters']
    D = seal_parameters['pupil_size']
    simulation_elements = build_seal_simulation(seal_parameters)
    #grid_dim of a 100 implies we are testing over 10,000 different conditions
    dim = seal_parameters['grid_dim']
    #evenely spaced, 1D array
    #1D defocus pace in mm, 
    x_wise = np.linspace(-10,10, dim)
    y_wise = np.linspace(-10,10, dim)
    x_wise_m = x_wise /1000
    y_wise_m = y_wise /1000

    #unique (i,j) index pairs, upper triangle to avoid mirror
    upper_triangle = np.triu_indices(dim, 1)
    phase_diverse_inputs= []

    #build each input set
    #input should  = (index_i,index_j, {defocus1_m:phase1, defocus2_m:phase2})

    for upper_triangle_indices in upper_triangle: 
        index_x = upper_triangle_indices[0] 
        index_y = upper_triangle_indices[1] 
        match_x = x_wise_m[index_x] 
        match_y = y_wise_m[index_y] 
        #List of tuples

        '''
        could delete  and replace with 
        phase_diverse_inputs.append((index_x, index_y, match_x, match_y)) 
        to remove type error in delta_to_p
        '''
        defocus_dict = {
            match_x: calculate_defocus_phase(seal_parameters, 
                                             simulation_elements, 
                                             match_x),
            match_y: calculate_defocus_phase(seal_parameters, 
                                             simulation_elements, 
                                             match_y)
        }
        phase_diverse_inputs.append((index_x, index_y, defocus_dict))
        
    
    

main(seal_parameters,
     phase_diverse_inputs,
     file_name_out ='example_file_name.npy',
     heatmap_plot_out= 'example_heatmap.png')


