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
#equation [9] of dean/bowers paper
def dean_bowers_max_list(fringes,
                 max_n):
    v_hat= fringes
    
    return [v_hat**2 / (4*((2*n)-1)) for n in range(max_n +1)]
#inverse of normalized defocus equation
def a_hat_to_defocus(a_hat,f,D,wavelength):
    return (4 * f**2 * wavelength) / (np.pi * D**2) * a_hat

def dean_bowers_min(fringes,
                    n,
                    wavelength):
    v_hat=fringes
    return (v_hat**2)/(8*n)

#Build the Seal Simulation

def build_seal_simulation(seal_parameters):
    """
    Build and return the core optical simulation elements for SEAL phase diversity.

    Parameters
    ----------
    seal_parameters : dict
        Dictionary of system configuration. Must contain:
        - 'q' (int): Oversampling factor for focal grid.
        - 'Num_airycircles' (int): Number of Airy rings to include.
        - 'pupil_size' (float): Telescope pupil diameter (m).
        - 'focal_length_meters' (float): Effective focal length (m).
        - 'wavelength_meter' (float): Reference wavelength (m).
        - 'pupil_pixel_dimension' (int): Resolution of pupil grid.
        - 'small_pupil_size_meter' (float): Diameter of the masking pupil (m).

    Returns
    -------
    simulation_elements : dict
        Dictionary containing:
        - pupil_grid (PupilGrid): HCIPy grid for the pupil plane.
        - focal_grid (FocalGrid): HCIPy grid for the focal plane.
        - aperture (callable): Function to generate a circular aperture on any grid.
        - telescope_pupil (ndarray): Binary mask of the full telescope pupil.
        - masking_pupil (ndarray): Binary mask of the smaller (masking) pupil.
        - zernike_sample_256 (list of ndarray): First 256 Zernike mode shapes.
        - fourier_sample_84 (list of ndarray): 84 low‑order Fourier mode shapes.
        - pupil_wavefront (Wavefront): Uniform wavefront over the pupil grid.
        - original_wavefront (Wavefront): Wavefront with telescope_pupil applied.
        - original_focal_image (Wavefront): Result of forward‑propagating original_wavefront.
        - pupil_image (Wavefront): Copy of original_wavefront for later reference.
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

    pupil_grid = make_pupil_grid(
        seal_parameters['pupil_pixel_dimension'],
        seal_parameters['pupil_size']
    )
    aperture = make_circular_aperture(seal_parameters['pupil_size'])
    telescope_pupil = aperture(pupil_grid)
    small_aperture = make_circular_aperture(seal_parameters['small_pupil_size_meter'])
    masking_pupil = small_aperture(pupil_grid)

    zernike_modes = make_zernike_basis(
         num_modes=256,
         D=seal_parameters['pupil_size'],
         grid=pupil_grid
    )
    zernike_sample_256 = [mode.shaped for mode in zernike_modes[:256]]

    freq_pairs = [(i, 0) for i in range(1, 43)] + [(0, i) for i in range(1, 43)]  # 84 total
    kx = np.array([kx * 2 * np.pi / seal_parameters['pupil_size'] for kx, _ in freq_pairs])
    ky = np.array([ky * 2 * np.pi / seal_parameters['pupil_size'] for _, ky in freq_pairs])
    fourier_coords = UnstructuredCoords((kx, ky))
    fourier_grid = CartesianGrid(fourier_coords)
    fourier_basis = make_fourier_basis(pupil_grid, fourier_grid)
    fourier_sample_84 = [mode.shaped for mode in fourier_basis]
    pupil_wavefront = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), 
                                seal_parameters['wavelength'])
    original_wavefront = Wavefront(telescope_pupil, wavelength=650e-9)
    prop2f = FraunhoferPropagator(pupil_grid,
                                    focal_grid,
                                    focal_length=seal_parameters['focal_length_meters']
                                    )
    original_focal_image = prop2f.forward(original_wavefront)
    pupil_image = original_wavefront.copy()
    #pupil_image in orignal is = original_wavefront

    # Return all components as a dictionary
    simulation_elements = {
        'pupil_grid': pupil_grid,
        'focal_grid': focal_grid,
        'aperture': aperture,
        'telescope_pupil': telescope_pupil,
        'masking_pupil': masking_pupil,
        'zernike_sample_256' : zernike_sample_256,
        'fourier_sample_84' : fourier_sample_84,
        'pupil_wavefront' : pupil_wavefront,
        'original_wavefront' : original_wavefront,
        'original_focal_image' : original_focal_image,
        'pupil_image' : pupil_image
        }
    

    return simulation_elements

def propagate_image_to_focal(defocus_phase,
                            wf_error_to_retrieve,
                            simulation_elements,
                            seal_parameters):
    """
    Progagate a combined pupil‐plane aberration to the focal plane.

    Parameters
    ----------
    defocus_phase : ndarray
        2D array of defocus phase aberration (radians), matching the pupil grid shape.
    wf_error_to_retrieve : ndarray
        2D array of the static wavefront error (radians), same shape as defocus_phase.
    simulation_elements : dict
        Precomputed optical components:
        - 'pupil_grid' (PupilGrid): grid over the pupil plane.
        - 'focal_grid' (FocalGrid): grid over the focal plane.
        - 'telescope_pupil' (ndarray): binary pupil mask.
    seal_parameters : dict
        SEAL configuration parameters; must include:
        - 'focal_length_meters' (float): system focal length in meters.
        - 'wavelength' (float): propagation wavelength in meters.

    Returns
    -------
    focal_intensity : ndarray
        2D intensity image at the focal plane.
    focal_field : Wavefront
        HCIPy Wavefront object representing the complex field after propagation.
    """
    prop_p2f = FraunhoferPropagator(simulation_elements['pupil_grid'],
                                    simulation_elements['focal_grid'],
                                    seal_parameters['focal_length_meters'])
    combined_abberation = (defocus_phase + wf_error_to_retrieve).ravel()
    pupil_field = simulation_elements['telescope_pupil'] * np.exp(1j * combined_abberation)
    wavefront = Wavefront(pupil_field, seal_parameters['wavelength'])
    focal_field = prop_p2f.forward(wavefront)
    #focal_intensity = np.abs(focal_field.electric_field.reshape(simulation_elements['focal_grid'].shape))**2
    focal_intensity=focal_field.intensity
    return focal_intensity, focal_field


def calculate_defocus_params(seal_parameters,
                            simulation_elements,
                            defocus_distance, 
                            defocus_template):#All meters
    """
    Compute the 2D defocus phase map for a given physical defocus distance.

    Parameters
    ----------
    seal_parameters : dict
        - 'focal_length_meters' (float): System focal length in meters.
        - 'pupil_size' (float): Telescope pupil diameter in meters.
    simulation_elements : dict
        - 'zernike_sample_256' (list of ndarray): Zernike mode shapes; index 3 is defocus.
    defocus_distance : float
        Physical defocus distance in meters.

    Returns
    -------
    defocus_phase : ndarray
        2D phase aberration in radians, matching the pupil grid shape.

    Variables
    ---------
    defocus_template : ndarray
        The unit Zernike defocus mode (shaped array).
    template_p2v : float
        Peak-to-valley amplitude of the unit template.
    unit_defocus : ndarray
        Normalized defocus template with unit P2V.
    defocus_p2v : float
        Target phase P2V (radians) computed from physical defocus.
    defocus_phase : ndarray
        Scaled phase map = unit_defocus * defocus_p2v.
    defocus_p2v_radians : float
        Peak-to-valley of the computed defocus_phase.
    defocus_p2v_m : float
        Equivalent physical defocus distance (meters) from defocus_p2v_radians.
    delta : float
        Recomputed defocus distance via inverse mapping (should match defocus_distance).
    """
    template_p2v=np.max(defocus_template)-np.min(defocus_template) 
    assert template_p2v > 0
    unit_defocus = defocus_template / template_p2v
    defocus_p2v = delta_to_p(
                            delta = defocus_distance,
                            f = seal_parameters['focal_length_meters'],
                            D=seal_parameters['pupil_size']
                            )
    defocus_phase = unit_defocus * defocus_p2v
    defocus_p2v_radians = np.max(defocus_phase) - np.min(defocus_phase)
    defocus_p2v_m = phase_to_m(defocus_p2v_radians, 650e-9)
    delta = p_to_delta(defocus_p2v_m, f, D)
    return defocus_phase
def get_defocus_metrics(seal_parameters, defocus_distance):
    f = seal_parameters['focal_length_meters']
    D = seal_parameters['pupil_size']
    p2v_m  = p_to_delta(defocus_distance, f, D)
    p2v_rad = delta_to_p(p2v_m, f, D)
    return p2v_m, p2v_rad

def simulate_no_defocus_image(wf_error_to_retrieve, 
                            simulation_elements,
                            seal_parameters, 
                            defocus_template):
    """
    Generate a focused PSF without any added defocus aberration.Just the abberation trying to sense. 

    Parameters
    ----------
    wf_error_to_retrieve : ndarray
        2D array of static pupil-plane phase error to be retrieved (radians).
    simulation_elements : dict
        Prebuilt simulation components, including:
        - 'pupil_grid' (PupilGrid)
        - 'focal_grid' (FocalGrid)
        - 'telescope_pupil' (ndarray)
    seal_parameters : dict
        Optical configuration parameters:
        - 'wavelength' (float): propagation wavelength (meters)
        - 'focal_length_meters' (float): system focal length (meters)
    defocus_template : ndarray
        2D Zernike defocus template (radians) for grid shape reference.

    Variables
    ---------
    no_defocus_phase : ndarray
        Zero-valued phase map (same shape as defocus_template).
    no_defocus_intensity : ndarray
        Resulting focused PSF intensity (2D array).
    no_defocus_propd : Wavefront
        HCIPy Wavefront object after propagation.

    Returns
    -------
    no_defocus_intensity, no_defocus_propd
    """
    #Remind myself what Abberation trying to retrieve
    plt.figure()
    plt.imshow(wf_error_to_retrieve)
    plt.title('Error to Retrieve')
    plt.colorbar()
    plt.show()
    no_defocus_phase = np.zeros_like(defocus_template)
    no_defocus_intensity, no_defocus_propd = propagate_image_to_focal(no_defocus_phase, 
                                                                      wf_error_to_retrieve, 
                                                                      simulation_elements, 
                                                                      seal_parameters)
        #must be appended into the psf_list
    return no_defocus_intensity, no_defocus_propd
def simulate_defocused_image(defocus_phase,
                             wf_error_to_retrieve, 
                             seal_parameters, 
                             simulation_elements,
                             ):
    """
    Simulate a defocused PSF by applying a defocus phase map.

    Parameters
    ----------
    defocus_phase : ndarray
        2D phase map (radians) representing defocus aberration.
    wf_error_to_retrieve : ndarray
        2D static wavefront phase error map (radians).
    seal_parameters : dict
        Configuration parameters:
        - 'focal_length_meters' (float): focal length in meters.
        - 'wavelength' (float): propagation wavelength in meters.
    simulation_elements : dict
        Precomputed optical components:
        - 'pupil_grid' (PupilGrid): grid for the pupil plane.
        - 'focal_grid' (FocalGrid): grid for the focal plane.
        - 'telescope_pupil' (ndarray): binary pupil mask.

    Returns
    -------
    defocus_intensity : ndarray
        2D defocused PSF intensity image.
    defocus_propd : Wavefront
        HCIPy Wavefront object after propagation.
    """
    #wf_error_to_retrieve = wf_error_to_retrieve.reshape(simulation_elements['telescope_pupil'].shape)
    defocus_intensity, defocus_propd = propagate_image_to_focal(defocus_phase, 
                                                                wf_error_to_retrieve, 
                                                                simulation_elements, 
                                                                seal_parameters)
    return defocus_intensity, defocus_propd
def convert_psf_estimate_to_phase(psf_estimate,
                                  simulation_elements,  
                                  seal_parameters, 
                                  telescope_pupil, 
                                  phase_unwrap=phase_unwrap_2d):
    """
    Reconstruct a pupil plane phase map from a retrieved PSF estimate.

    Parameters
    ----------
    psf_estimate : ndarray
        2D array of retrieved PSF amplitude or intensity.
    simulation_elements : dict
        Precomputed optics, must include:
        - 'masking_pupil' (Field or ndarray): mask for valid pupil region.
        - 'original_wavefront' (Wavefront): for reference P2V calculation.
    seal_parameters : dict
        Configuration parameters:
        - 'pupil_pixel_dimension' (int): output pupil grid size.
    telescope_pupil : Field
        HCIPy Field defining the telescope pupil mask.
    phase_unwrap : {"phase_unwrap_2d", "unwrap_phase", callable}, optional
        Method to unwrap raw phase:
        - "phase_unwrap_2d": custom 2D unwrapping.
        - "unwrap_phase": skimages unwrap_phase.
        - callable: user provided function mapping 2D→2D.

    Variables
    ---------
    pupil_dim : int
        Size of the square pupil output.
    seal_configuration : InstrumentConfiguration
        Object carrying optics parameters for inverse MFT.
    raw_pupil_phase : ndarray
        Phase (radians) after inverse modified Fourier transform.
    med_subtracted : ndarray
        Pupil phase with median over valid mask removed.

    Returns
    -------
    pupil_phase : ndarray
        Final unwrapped, resized, and masked phase map (radians).
    """
    pupil_dim = seal_parameters['pupil_pixel_dimension']
    seal_configuration = InstrumentConfiguration(seal_parameters)
    raw_pupil_phase = np.angle(mft_rev(psf_estimate,seal_configuration)) 

    if phase_unwrap == "phase_unwrap_2d":
        phase_unwrap = phase_unwrap_2d
    elif phase_unwrap == "unwrap_phase":
        phase_unwrap = unwrap_phase
    if phase_unwrap is not None:
        raw_pupil_phase = phase_unwrap(raw_pupil_phase)

    pupil_phase = resize(raw_pupil_phase, (pupil_dim, pupil_dim)) * telescope_pupil.shaped
    med_subtracted = pupil_phase - np.median(pupil_phase[np.array(masking_pupil.shaped, dtype=bool)])
    print(f'P2V error of pupil_phase: {np.max(pupil_phase) - np.min(pupil_phase)}')

    print(f"P2V error: {np.max(simulation_elements['original_wavefront'].phase.shaped) - np.min(simulation_elements['original_wavefront'].phase.shaped)}")
    return pupil_phase
def phase_metrics(wf_error_to_retrieve,
                  no_defocus_propd, 
                  phase_estimate,
                  masking_pupil, 
                  simulation_elements
                  ):
    """
    Compute phase‐retrieval error metrics comparing true and estimated pupil phases.

    Parameters
    ----------
    wf_error_to_retrieve : ndarray
        2D array of the injected pupil‐plane phase error (radians).
    no_defocus_propd : Wavefront
        HCIPy Wavefront of the zero‐defocus propagation (ground truth field).
    phase_estimate : ndarray
        2D array of the retrieved pupil phase (radians).
    masking_pupil : Field or ndarray
        Binary mask defining the valid pupil region.
    simulation_elements : dict
        Contains:
        - 'pupil_image' (Wavefront): copy of original_wavefront for true phase.
        - 'telescope_pupil' (Field): pupil mask for phase injection.

    Returns
    -------
    dict
        {
            'rms_error_1' : float,
                RMS error (radians) over masked region between true and estimate.
            'rms_error_2' : float,
                RMS error (radians) of median‐removed difference within pupil.
            'difference_image' : ndarray,
                1D‐masked array of phase differences (true–estimate).
            'difference_true_vs_estimate' : ndarray,
                Full 2D difference map (no mask).
            'p2v_error' : float,
                Peak‐to‐valley of the true‐vs‐estimate difference (radians).
        }

    Internal Variables
    ------------------
    mask : ndarray (bool)
        Flattened mask array for valid pupil pixels.
    pupil_image : Wavefront
        True wavefront with injected error for reference phase.
    med_subtracted : ndarray
        Estimated phase with median over mask removed.
    difference_true_vs_estimate : ndarray
        2D phase difference between propagated truth and estimate.
    difference_masked : ndarray
        1D array of masked differences for RMS compute.
    rms_error_diff_masked : float
        RMS of `difference_masked` (radians).
    p2v_error_diff_true_vs_estimate : float
        Peak‐to‐valley of `difference_true_vs_estimate` (radians).
    difference_image : ndarray
        2D array of (true phase – median‐removed estimate) for visualization.
    check_error_region : ndarray
        Masked 1D array of `difference_image` for nanometer stats.
    nm_med : float
        Median error over pupil in nanometers.
    rms_error_region : float
        RMS of `check_error_region` (radians).
    rms_nm : float
        RMS error converted to nanometers.

    """
    mask = np.array(masking_pupil.shaped, dtype =bool)
    pupil_image = simulation_elements['pupil_image']
    pupil_image.electric_field = np.exp(complex(0, 1)*simulation_elements['telescope_pupil']*(wf_error_to_retrieve))
    print(f'P2V of Original injected error: {np.max(pupil_image.phase.shaped) - np.min(pupil_image.phase.shaped)}')
    
    #or pupil_phasE?
    med_subtracted = phase_estimate - np.median(phase_estimate[np.array(masking_pupil.shaped, dtype=bool)])
    difference_true_vs_estimate = (no_defocus_propd.phase - phase_estimate)
    difference_masked = difference_true_vs_estimate[mask]
    rms_error_diff_masked = np.sqrt(np.mean(difference_masked ** 2))
    p2v_error_diff_true_vs_estimate = np.max(difference_true_vs_estimate) - np.min(difference_true_vs_estimate)


    difference_image = pupil_image.phase.shaped - med_subtracted#Masking pupil
    check_error_region = (pupil_image.phase.shaped - med_subtracted)[np.array(masking_pupil.shaped, dtype=bool)]
    nm_med = phase_to_m(np.median(check_error_region), 650e-9) * 1e9
    rms_error_region = np.sqrt(np.mean((check_error_region) ** 2))
    rms_nm= phase_to_m(rms_error_region, 650e-9)*1e9


    plt.imshow(difference_image * masking_pupil.shaped)
    plt.title(f'Difference Image * Masking Pupil.shaped')
    plt.show()
    print(f"RMS error:{rms_nm} Nanometers")
    print(f"RMS error: {rms_error_region} radians")
    print(f'Median error in nanometers: {nm_med} nm')
    return {
        'rms_error_1' : rms_error_diff_masked,
        'rms_error_2' : rms_error_region, 
        'difference_image':difference_masked,
        'difference_true_vs_estimate':difference_true_vs_estimate, 
        'p2v_error' : p2v_error_diff_true_vs_estimate
    }


def phase_retrieval_accuracy(
        pupil_phase, 
        wf_error_to_retrieve, 
        no_defocus_propd, 
        psf_estimate, 
        cost_functions, 
        seal_parameters, 
        simulation_elements, 
        phase_unwrap_method = None,
        verbose = True,
        ):
    """
    Reconstruct pupil phase from a PSF estimate and compute retrieval error metrics.

    Parameters
    ----------
    pupil_phase : ndarray
        Ground‐truth pupil phase map (radians).
    wf_error_to_retrieve : ndarray
        Injected static pupil‐plane phase error map (radians).
    no_defocus_propd : Wavefront
        HCIPy Wavefront of the zero‐defocus propagation (true field).
    psf_estimate : ndarray
        Retrieved PSF estimate (2D intensity or amplitude array).
    cost_functions : list of list of float
        History of cost function values for each defocus channel.
    seal_parameters : dict
        Optical system parameters (e.g., focal length, wavelength).
    simulation_elements : dict
        Precomputed optics/components (e.g., grids, pupils).
    phase_unwrap_method : {"phase_unwrap_2d", "unwrap_phase", callable}, optional
        Method to unwrap phase during conversion.
    verbose : bool, default True
        If True, plot cost‐function convergence.

    Returns
    -------
    phase_estimate : ndarray
        Recovered pupil‐plane phase map (radians).
    metrics : dict
        Error‐metrics returned by `phase_metrics`, including:
        - 'rms_error_1': RMS error over masked region (radians)
        - 'rms_error_2': RMS error after median‐removal (radians)
        - 'p2v_error': peak‐to‐valley of phase difference (radians)
        - 'difference_image': 1D array of masked differences
        - 'difference_true_vs_estimate': full 2D difference map
    """
    phase_estimate = convert_psf_estimate_to_phase(psf_estimate,
                                                   simulation_elements, 
                                                   seal_parameters,
                                                   simulation_elements['telescope_pupil'],
                                                   phase_unwrap_method)
    phase_estimate_metrics = phase_metrics(
        wf_error_to_retrieve,
        no_defocus_propd, 
        phase_estimate, 
        simulation_elements,
        psf_estimate) 
    if verbose:
        make_cost_functions_plots(cost_functions)

    return phase_estimate, phase_estimate_metrics

def make_cost_functions_plots(cost_functions,
                               filename='Jul_9.jpg'):
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

def run_phase_retrieval(no_defocus_intensity,
                        defocus_dictionary, 
                        seal_parameters,
                        num_iterations=200):
    """
    Perform iterative focus‑diverse phase retrieval using Gerchberg–Saxton iterations.

    Parameters
    ----------
    no_defocus_intensity : ndarray
        2D focused PSF intensity image without any applied defocus.
    defocus_dictionary : dict
        Mapping from defocus_distance (float, same units as used in FDPR) to
        its corresponding 2D defocused PSF intensity ndarray.
    seal_parameters : dict
        SEAL system parameters:
        - 'image_dx' (float): pixel scale of the PSF images in microns.
        - 'wavelength_micron' (float): wavelength for phase retrieval in microns.
    num_iterations : int, optional
        Number of phase‑retrieval iterations to perform (default=200).

    Variables
    ---------
    distance_list : list of float
        Ordered list of defocus distances (keys of defocus_dictionary).
    psf_list : list of ndarray
        Concatenation of [no_defocus_intensity] + defocused PSFs in distance_list order.
    dx_list : list of float
        List of image_dx repeated once per defocus channel.
    mp : FocusDiversePhaseRetrieval
        Instance of the FDPR solver, initialized with psf_list, wavelengths, dxs, and defocus_positions.
    psf_estimate : ndarray
        The complex field estimate (as a 2D array) after the final iteration.

    Returns
    -------
    psf_estimate : ndarray
        Final retrieved complex field (FFT‑shifted) in the focal plane.
    cost_functions : list of list of float
        Convergence histories of the cost function for each defocus channel.
    """

    distance_list = list(defocus_dictionary.keys()) 
    psf_list = [no_defocus_intensity] +[defocus_dictionary[key] for key in distance_list]
    for psf in psf_list[1:]:
        print(type(psf), psf.shape)
    print("PSF shapes:", [psf.shape for psf in psf_list])


    print('psf_list:',psf_list)
    print('type of psf_list', type(psf_list))
    dx_list= [seal_parameters['image_dx'] for key in distance_list]
    mp= FocusDiversePhaseRetrieval(psflist= psf_list,
                                   wvl= seal_parameters['wavelength_micron'], #JARENS NOT IN METER
                                   dxs =dx_list, 
                                   defocus_positions= distance_list)
    for i in range(num_iterations):
            psf_estimate = mp.step() ##does this for loop overwrite psf_estimate each time
    return psf_estimate, mp.cost_functions

def focus_diverse_phase_retrieval(no_defocus_intensity,
                                  phase_diverse_information, 
                                  wf_error_to_retrieve,  
                                  seal_parameters,
                                  simulation_elements):
    """
    Perform phase retrieval using one no defocus PSF and multiple defocused PSFs.

    Parameters
    ----------
    no_defocus_intensity : ndarray
        2D focused PSF intensity image (no defocus applied).
    phase_diverse_information : dict
        Mapping from defocus_distance (float, meters) to its phase map (2D ndarray, radians).
    wf_error_to_retrieve : ndarray
        Static pupil-plane phase error map (radians) to be retrieved.
    seal_parameters : dict
        SEAL system parameters, e.g. focal length, wavelengths, pixel scales.
    simulation_elements : dict
        Prebuilt optical components including:
        - 'pupil_grid', 'focal_grid'
        - 'telescope_pupil', 'masking_pupil', etc.

    Variables
    ---------
    defocus_distances : dict_keys
        All defocus distances to simulate.
    defocus_psfs : dict
        Maps each defocus_distance to its simulated PSF intensity (2D ndarray).
    defocus_phase : ndarray
        The defocus phase map for the current distance.
    defocused_psf : ndarray
        Intensity from `simulate_defocused_image` for this defocus_phase.

    Returns
    -------
    psf_estimate : ndarray
        Final retrieved complex field estimate (FFT-shifted) from the solver.
    cost_functions : list of list of float
        Convergence history of the cost function for each defocus channel.
    """
    defocus_distances = phase_diverse_information.keys()
    #change to allow mulitple inptus
    defocus_psfs={}
    for defocus_distance in defocus_distances:

        defocus_phase = phase_diverse_information[defocus_distance]
        
        
        defocused_psf,_  = simulate_defocused_image( # the fcn returns focal_intensity, we reassign here
            defocus_phase,
            wf_error_to_retrieve,
            seal_parameters,
            simulation_elements
        )
        #put into disctionary keyed by defocus distance
        defocus_psfs[defocus_distance]=defocused_psf# just the intensity

        #run it
    psf_estimate, cost_functions = run_phase_retrieval(
            no_defocus_intensity,
            defocus_psfs,
            seal_parameters
        )

    return psf_estimate, cost_functions

def simulate_phase_diversity_grid(wf_error_to_retrieve,
                                  defocus_template, 
                                  phase_diverse_inputs, 
                                  seal_parameters, 
                                  file_name_out, 
                                  simulation_elements):
    no_defocus_intensity, no_defocus_propd = simulate_no_defocus_image(wf_error_to_retrieve,
                                                                       simulation_elements,
                                                                       seal_parameters, 
                                                                       defocus_template
                                                                       )
    dim = seal_parameters['grid_dim'] 
    phase_diversity_grid = np.zeros((dim,dim))

    for simulation_specifics in phase_diverse_inputs:
        index_x, index_y, defocus_dictionary = simulation_specifics
        indices = (index_x, index_y)
        defocus_distances = defocus_dictionary.keys()
        defocus_dictionary= {}

        for defocus_distance in defocus_distances:


            defocus_phase = \
            calculate_defocus_params(seal_parameters,
                                    simulation_elements,
                                    defocus_distance, 
                                    defocus_template)
            params = get_defocus_metrics(seal_parameters, defocus_distance)
            print(params)
            #print(f"Defocus: {defocus_distance:.5f}, delta: {D:.2e}, P2V: {(np.max(defocus_phase)-np.min(defocus_phase)):.2e}, max phase: {np.max(defocus_phase):.2f}")
            #defocused_psf = simulate_defocused_image(defocus_phase,
            #                                         wf_error_to_retrieve,
            #                                         seal_parameters,
            #                                         simulation_elements)
            defocus_dictionary[defocus_distance] = defocus_phase
        psf_estimate, cost_functions = focus_diverse_phase_retrieval(no_defocus_intensity, 
                                                                     defocus_dictionary,
                                                                     wf_error_to_retrieve, 
                                                                     seal_parameters, 
                                                                     simulation_elements)
        
        phase_estimate, metrics = phase_retrieval_accuracy(wf_error_to_retrieve, 
                                                            no_defocus_propd, 
                                                            psf_estimate, 
                                                            cost_functions, 
                                                            seal_parameters,
                                                            simulation_elements,
                                                            phase_unwrap_method = 'phase_unwrap_2d', 
                                                            verbose =True)
        phase_diversity_grid[indices] = metrics['rms_error']
        assert 0 <= indices[0] < dim and 0 <= indices[1] < dim


    np.save(file_name_out, phase_diversity_grid) #will assign name when function runs
    print("phase_diversity_grid stats:")
    print("Min:", np.min(phase_diversity_grid))
    print("Max:", np.max(phase_diversity_grid))
    print("Nonzero count:", np.count_nonzero(phase_diversity_grid))
    print("Shape:", phase_diversity_grid.shape)

    return phase_diversity_grid                                                            

def plot_phase_diversity_heat_map(phase_diversity_grid,
                                  heatmap_plot_out):
    plt.clf()
    extent = [-10, 10, -10, 10]  # assuming defocus in mm, update if different
    plt.imshow(phase_diversity_grid, origin='lower', extent=extent)
    plt.colorbar(label='RMS Error')
    plt.title('Phase Retrieval RMS Error Heatmap')
    plt.xlabel('Defocus Distance [m]')
    plt.ylabel('Defocus Distance [m]')
    plt.savefig(heatmap_plot_out)
    plt.clf()

def main(seal_parameters, 
         phase_diverse_inputs,
         defocus_template,#might be defocus_grid
         file_name_out,  
         heatmap_plot_out,
         zernike_index = 6,
         fourier_index_low= 1, 
         fourier_index_high = 10
         ): ##need help to implement this grid and using scales
    # Build simulation elements like pupil grid, aperture, Zernike/Fourier modes, etc.
    simulation_elements = build_seal_simulation(seal_parameters)
    # Get the simulation wavelength
    wavelength = seal_parameters['wavelength_meter']
    # DEFINE ABERRATION TO RETRIEVE (ground truth) — outside the simulation functions
    zernike_modes = simulation_elements['zernike_sample_256']
    wf_error_to_retrieve = 0.75 * zernike_modes[zernike_index]
    sinusoidal_abberation_low = .75 * simulation_elements['fourier_sample_84'][fourier_index_low]
    sinusoidal_abberation_high = .75 * simulation_elements['fourier_sample_84'][fourier_index_high]
    # Simulate focused image using that known wavefront error
    phase_diversity_grid= simulate_phase_diversity_grid(
            phase_diverse_inputs = phase_diverse_inputs,
            defocus_template=defocus_template,
            simulation_elements = simulation_elements,
            seal_parameters = seal_parameters, 
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
        'grid_dim':10
         }
    seal_parameters['wavelength']=seal_parameters['wavelength_meter']
    simulation_elements = build_seal_simulation(seal_parameters)
    defocus_template =simulation_elements['zernike_sample_256'][3]
    #some list of defocus distances 
    f = seal_parameters['focal_length_meters']
    D = seal_parameters['pupil_size']
    simulation_elements = build_seal_simulation(seal_parameters)
    #grid_dim of a 100 implies we are testing over 10,000 different conditions
    dim = seal_parameters['grid_dim']
    #evenely spaced, 1D array
    fringes=21
    max_n=1
    wavelength=seal_parameters['wavelength_meter']
    v_hat=fringes
    a_hats = dean_bowers_max_list(fringes,max_n)
    deltas=[a_hat_to_defocus(a,f,D,wavelength) for a in a_hats]

    #1D defocus pace in mm, 
    x_wise = np.linspace(-10,10, dim)
    y_wise = np.linspace(-10,10, dim)
    x_wise_m = x_wise /1000
    y_wise_m = y_wise /1000
    phase_diverse_inputs=[]

    upper_triangle = np.triu_indices(dim,1)
    upper_triangle_list = [[upper_triangle[0][i], upper_triangle[1][i]] for i in range(np.shape(upper_triangle)[1])]

    for index in upper_triangle_list:
        index_x, index_y = index
        x,y = x_wise_m[index_x], y_wise_m[index_y]
        defocus_dictionary = {
            x: calculate_defocus_params(seal_parameters, 
                                             simulation_elements, 
                                             x, 
                                             defocus_template),
            y: calculate_defocus_params(seal_parameters, 
                                             simulation_elements, 
                                             y, 
                                             defocus_template)
        }
        phase_diverse_inputs.append((index_x, index_y, defocus_dictionary))       
    

main(seal_parameters,
     phase_diverse_inputs,
     defocus_template,
     file_name_out ='jul9_neat.npy',
     heatmap_plot_out= 'example_heatmap.png')
