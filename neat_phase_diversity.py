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

def convert_psf_estimate_to_phase(psf_estimate, seal_parameters, telescope_pupil, phase_unwrap=None):
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
        ndarray: 2D unwrapped and pupil-masked phase map of the retrieved pupil field (in radians).

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

    This function compares the estimated phase to the known system truth phase by computing
    pixel-wise differences over the region defined by a masking pupil. It returns standard
    error metrics like RMS (root-mean-square) and P2V (peak-to-valley), as well as 
    difference images for diagnostics.

    Parameters:
        system_truth (Wavefront): HCIPy Wavefront object containing the true phase 
            in its `.phase` attribute.
        phase_estimate (ndarray): 2D array representing the estimated pupil phase (in radians).
        masking_pupil (Field or ndarray): Binary mask defining the region of interest
            for error evaluation. Pixels outside this mask are excluded from RMS.

    Returns:
        dict: Dictionary containing:
            - 'rms_error' (float): Root-mean-square error between truth and estimate (masked).
            - 'p2v_error' (float): Peak-to-valley error over the full difference image.
            - 'difference_image' (ndarray): Masked phase difference values (truth - estimate).
            - 'difference_true_vs_estimate' (ndarray): Full, unmasked phase difference map.

    Notes:
        - The mask ensures that RMS is only computed inside the clear pupil aperture.
        - P2V is calculated on the full unmasked difference, which may include edge artifacts.
        - You may optionally consider subtracting the median from each phase before comparison
          to eliminate piston effects.
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
        Run the iterative phase retrieval algorithm using focus-diverse PSFs.

    This function initializes and executes the phase retrieval process based on a set of 
    defocused PSFs and the system's true focused PSF. It uses the FocusDiversePhaseRetrieval 
    class to estimate the pupil phase that best reproduces the observed intensity patterns.

    Parameters:
        system_truth (ndarray):
            The focused PSF (no defocus), used as the reference for reconstruction.
        defocus_dictionary (dict):
            Dictionary mapping each defocus distance (float, in meters) to its corresponding defocused PSF (ndarray).
        seal_parameters (dict):
            Dictionary containing SEAL optical system configuration. Must include:
                - 'image_dx': Pixel size in the image plane (float)
                - 'wavelength': Central wavelength used in the simulation (float)
        num_iterations (int, optional):
            Number of optimization iterations to run. Default is 200.

    Returns:
        tuple:
            - psf_estimate (ndarray): Estimated complex PSF after phase retrieval.
            - cost_functions (list of list of float): Convergence history of the cost function for each defocus input.

    Notes:
        - The `FocusDiversePhaseRetrieval` object is assumed to implement a `.step()` method 
          that updates the phase estimate and stores cost history.
        - Defocus PSFs are assumed to be aligned with the `distance_list`.
        - The focused PSF is prepended to the list of PSFs as the reference.
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
                                   seal_parameters['wavelength_meter'], #JARENS NOT IN METER
                                   dx_list, 
                                   distance_list)

    for i in range(num_iterations):
            psf_estimate = mp.step() 
    

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
    Evaluate the accuracy of phase retrieval by comparing estimated and true wavefront phases.

    This function reconstructs the pupil phase from the retrieved PSF, compares it to the ground 
    truth phase, and computes error metrics such as RMS and peak-to-valley (P2V) error. 
    Optionally, it can also plot the cost function convergence history.

    Parameters:
        system_truth (Wavefront):
            The known wavefront used as the ground truth for comparison.
        psf_estimate (ndarray):
            The estimated PSF obtained from the retrieval algorithm.
        cost_functions (list of list of float):
            Convergence history of the cost function across iterations and defocus inputs.
        seal_parameters (dict):
            Dictionary of SEAL system configuration parameters, including pupil size, image resolution, etc.
        simulation_elements (dict):
            Dictionary of precomputed simulation elements, including:
                - 'masking_pupil': Boolean mask to isolate the pupil region
                - other components needed by downstream functions
        phase_unwrap_method (str, optional):
            String identifier for the unwrapping method to use. Options:
                - "phase_unwrap_2d"
                - "unwrap_phase"
                Default is None (no unwrapping).
        verbose (bool, optional):
            If True, plots the cost function history using `make_cost_functions_plots`.

    Returns:
        dict:
            Dictionary containing accuracy metrics:
                - 'rms_error': Root-mean-square phase difference over the pupil
                - 'p2v_error': Peak-to-valley phase difference
                - 'difference_image': Masked residual phase map
                - 'difference_true_vs_estimate': Full residual phase map (unmasked)

    Notes:
        - Assumes the phase unwrapping method corresponds to a callable function if specified.
        - Internally calls `convert_psf_estimate_to_phase()` and `check_phase_estimate()`.
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
                          wavelength):
    """
    Simulate a focused wavefront image using a known wavefront error.

    This function generates a complex-valued wavefront by applying the provided
    phase error to the SEAL telescope pupil and returns the corresponding wavefront
    object. This wavefront represents the focused optical field at the pupil plane.

    Parameters:
        wf_error_to_retrieve (ndarray):
            Phase error to inject into the wavefront (in radians). Should match the
            shape of the telescope pupil grid or be flattenable to match it.
        simulation_elements (dict):
            Dictionary containing precomputed simulation components. Must include:
                - 'telescope_pupil' (ndarray): Circular pupil aperture mask.
        wavelength (float):
            Wavelength of the simulated light in meters.

    Returns:
        Wavefront:
            Complex-valued wavefront object defined on the pupil with the applied phase error.

    Notes:
        - The input phase array is flattened to match the expected format for constructing
          the `Wavefront` object.
        - Assumes the pupil shape is compatible with the input phase shape.
    """
    telescope_pupil=simulation_elements['telescope_pupil']
    wf_focused = Wavefront(telescope_pupil * np.exp(1j * wf_error_to_retrieve.flatten()), 
                   seal_parameters['wavelength_meter'])
    wf_focused_intensity = wf_focused.intensity
    wf_focused_phase = wf_focused.phase
    #assert simulation_elements['telescope_pupil'].shape == wf_error_to_retrieve.shape, \
    "Wavefront error and telescope pupil shape mismatch"
    #.intensity gives us our actual image, and .shaped formats it into an ndarray in order to pass to FDPR
    return wf_focused_intensity.shaped, wf_focused_phase, wf_focused
    

def simulate_defocused_image(defocus_phase,
                             wf_error_to_retrieve, 
                             seal_parameters, 
                             simulation_elements,
                             ):#instead of defocus_phase pass wf,test_abberattion
    """
    Simulate a defocused PSF image by propagating a wavefront with combined aberrations.

    This function generates a defocused optical field by combining a known aberration 
    (e.g., Zernike or Fourier phase error) with a defocus phase term, then propagates 
    the resulting wavefront to the focal plane using Fraunhofer diffraction. The output 
    is the resulting intensity distribution in the focal plane.

    Parameters:
        defocus_phase (ndarray):
            Phase array representing the defocus aberration (in radians), same shape as aperture.
        wf_error_to_retrieve (ndarray):
            Wavefront phase error to retrieve (in radians), same shape as aperture.
        seal_parameters (dict):
            Dictionary of SEAL system configuration parameters; must contain:
                - 'focal_length_meters' (float): Effective focal length in meters.
        wavelength (float):
            Simulation wavelength in meters.
        simulation_elements (dict):
            Dictionary containing optical components including:
                - 'aperture' (ndarray): Pupil function (e.g., circular aperture mask).
                - 'pupil_grid' (Grid): Grid over which pupil is defined.
                - 'focal_grid' (Grid): Grid over which focal plane PSF is computed.

    Returns:
        ndarray:
            2D focal intensity image of the defocused PSF (same shape as focal grid).
    
    Raises:
        AssertionError:
            If the shapes of the aberration and aperture arrays do not match.

    Notes:
        - Uses the Fraunhofer propagator for far-field propagation.
        - The returned image is the `.shaped` intensity of the propagated wavefront.
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
    Convert a physical defocus distance into a Zernike-based defocus phase map.

    This function uses the normalized Zernike defocus mode (Zernike index 3) as a template 
    and scales it to match a specified physical defocus distance. The result is a phase 
    aberration map corresponding to the defocus term in radians.

    Parameters:
        seal_parameters (dict):
            Dictionary containing optical configuration parameters, must include:
                - 'focal_length_meters' (float): Effective focal length of the system [m].
                - 'pupil_size' (float): Diameter of the pupil [m].

        simulation_elements (dict):
            Dictionary of simulation components. Must include:
                - 'zernike_sample_12' (list of 2D arrays): List of the first 12 Zernike modes (shaped).

        defocus_distance (float):
            Physical defocus offset in meters (positive or negative).

    Returns:
        ndarray:
            A 2D array representing the defocus phase map in radians, scaled to match the
            specified physical defocus distance.

    Notes:
        - The Zernike mode used is hardcoded as index 3 (defocus term).
        - Phase scaling is done based on the peak-to-valley of the Zernike template.
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
    Run a single phase retrieval estimation using a focused and one defocused PSF.

    This function extracts the defocus phase term from a single entry in the 
    phase-diverse input list, simulates the corresponding defocused PSF, and 
    performs phase retrieval by comparing it with the focused "truth" PSF.

    Parameters:
        system_truth (Wavefront):
            Focused wavefront representing the ground truth PSF (complex field).

        wavelength (float):
            Wavelength in meters at which the PSFs are simulated.

        phase_diverse_information (tuple):
            A tuple of the form ((i, j), {defocus_distance: defocus_phase}), where:
                - (i, j): Grid index (used externally to track config).
                - {defocus_distance: defocus_phase}: Mapping of defocus value [m] 
                  to the corresponding phase map [radians].

        wf_error_to_retrieve (ndarray):
            The static wavefront error map to retrieve, in radians.

        seal_parameters (dict):
            Dictionary of optical configuration parameters for SEAL system.

        simulation_elements (dict):
            Dictionary of precomputed simulation components (aperture, grids, Zernike modes, etc.).

    Returns:
        tuple:
            - psf_estimate (ndarray): Final PSF estimate after phase retrieval.
            - cost_functions (list): History of cost function values during iterations.
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
    Simulate a 2D grid of phase diversity configurations and compute phase retrieval accuracy (RMS error)
    for each combination of defocus inputs.

    This function simulates one "focused" wavefront (`system_truth`) using the provided wavefront error,
    then loops over a 2D grid of phase-diverse input tuples—each containing defocus distances and associated
    grid indices—and runs a single phase retrieval using each defocus configuration. The result is a heatmap
    (2D array) of RMS error values for each simulation indexed by (i, j).

    Parameters:
        wf_error_to_retrieve (ndarray):
            Static wavefront error map to retrieve (in radians).

        phase_diverse_inputs (list of tuples):
            Each element is a tuple of the form:
                ((i, j), {defocus_distance_1: defocus_phase_1, defocus_distance_2: defocus_phase_2, ...})
            where (i, j) are 2D grid indices, and the dictionary maps defocus distances [m] to phase maps [rad].

        seal_parameters (dict):
            Dictionary containing SEAL system parameters (grids, wavelength, focal length, etc.).

        file_name_out (str):
            File path to save the resulting RMS error heatmap as a NumPy `.npy` file.

        simulation_elements (dict):
            Dictionary of precomputed simulation components (grids, apertures, modal bases, etc.).

        wavelength (float):
            Wavelength [meters] at which PSFs are simulated.

    Returns:
        phase_diversity_grid (ndarray):
            A 2D array (shape defined by `seal_parameters['grid_dim']`) of RMS phase retrieval errors.
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
        #indices first half
        #defocus_distance is second half
        
    #for index_x in phase_diverse_inputs.shape[0]:
        defocus_dictionary= {}
        #this takes second two information in tuple of simualtion_specifics
        defocus_distances = simulation_specifics[n_info:]
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
        indices = simulation_specifics[:n_info]
        #need to add index checks and make sure indices are within bounds
        
        phase_diversity_grid[indices] = metrics['rms_error']

    np.save(file_name_out, phase_diversity_grid) #will assign name when function runs

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

    plt.imshow(phase_diversity_grid)


    plt.colorbar(label='RMS Error')

    plt.title('Phase Retrieval RMS Error Heatmap')

    plt.xlabel('Defocus Distance [mm]')

    plt.ylabel('Defocus Distance [mm]')

    plt.savefig(heatmap_plot_out)

    plt.clf()
    #keep in mind extent logic, need to tell physical x y labels 

##WIP##
##WIP##
def main(seal_parameters, #might be defocus_grid
         file_name_out,  
         heatmap_plot_out,
         zernike_index = 6
         ): ##need help to implement this grid and using scales
    """
    Main driver function for SEAL phase diversity simulation and retrieval.

    This function:
    - Builds simulation elements
    - Injects a known Zernike aberration into the wavefront
    - Simulates a focused PSF
    - Performs phase diversity retrieval over a grid of defocus configurations
    - Generates a heatmap of the RMS phase retrieval error

    Parameters:
        seal_parameters (dict): Configuration for SEAL optical system.
        physical_defocus_range (list of tuples): Each tuple should be 
            (i, j, defocus_1, defocus_2, ...) indicating grid index and defocus values.
        file_name_out (str): File path to save the 2D phase retrieval error grid (.npy).
        heatmap_plot_out (str): File path to save the resulting RMS heatmap figure.
        zernike_index (int): Index of the Zernike mode to inject as the known aberration.

    Returns:
        None
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
        phase_diverse_inputs.append((index_x, index_y, match_x, match_y)) 

    

main(seal_parameters,
     file_name_out ='example_file_name.npy',
     heatmap_plot_out= 'example_heatmap.png')


