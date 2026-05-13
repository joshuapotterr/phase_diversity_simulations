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
    Function to setup the SEAL optical system parameters
    

    
    Parameters:
        seal_parameters(dict): Config for SEAL system
    
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
    This Function converts a PSF estimate into a pupil phase map by utilizing an Inverse Fourier Transform

    Parameters:
        psf_estimate(ndarray): Point Spread Function, in previous work defined as psf00
        seal_parameters(dict): Dictionary that represents SEAL test bed
        phase_unwrap: method to unwrap phase
    
    Returns: 
        ndarray: Unwrapped and Resized Pupil phase
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

def check_phase_estimate(system_truth, phase_estimate,masking_pupil):
    """
    This Function will determine the accuracy of our estimated phase
    Parameters:
        system_truth (Wavefront): Wavefront object with known phase.
        phase_estimate (ndarray): Estimated phase map.

    Returns:
        dict: Dictionary with error metrics like RMS and P2V.
    """
    true_phase = system_truth.phase.shaped # Get true phase
    mask = np.array(masking_pupil.shaped, dtype =bool)# Apply mask to non-zero phase region
    #implement med_subtracted, ie passing the dictionary will help with this(pupil_phase - median blah blah)
    difference_true_vs_estimate = (true_phase - phase_estimate) #Compute difference
    difference_masked = difference_true_vs_estimate[mask]
    rms_error = np.sqrt(np.mean(difference_masked ** 2))
    p2v_error = np.max(difference_true_vs_estimate) - np.min(difference_true_vs_estimate)
    return {'rms_error': rms_error, 
            'p2v_error': p2v_error, 
            'difference_image':difference_masked,
            'difference_true_vs_estimate' : difference_true_vs_estimate}#also return the difference image
    
def make_cost_functions_plots(cost_functions, filename=None):
    """
    This Function will plot the cost functions for each defocus input
    Parameters:
        cost_functions (list): List of cost function values per iteration.
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

def run_phase_retrieval(system_truth,
                         defocus_dictionary, 
                         seal_parameters,
                        num_iterations =200):
    """
    This function will run the actual phase retrieval algorithm
    
    Parameters:
        system_truth (ndarray): PSF with no defocus
        defocus_dictionary (dict): Mapping from defocus distances to PSF images.
        seal_parameters (dict): SEAL system configuration.

    Returns:
        tuple: Estimated PSF and cost function history.
    """

    #consider using sorted in order ensure psfs and distance arent mismatched?
    distance_list = list(defocus_dictionary.keys())

    psf_list = [system_truth] +[defocus_dictionary[key] for key in distance_list]
    
    #consider an assert statement 
    #assert all(isinstance(seal_parameters['image_dx'], (float, int)))
    dx_list= [seal_parameters['image_dx'] for key in distance_list]

    mp= FocusDiversePhaseRetrieval(psf_list,seal_parameters['wavelength_meter'], dx_list, distance_list)

    for i in range(num_iterations):
            psf_estimate = mp.step() 
    

    return psf_estimate, mp.cost_functions

def calculate_phase_retrieval_accuracy(
        system_truth, 
        psf_estimate, 
        cost_functions, 
        seal_parameters, 
        simulation_elements,
        phase_unwrap_method = None,
        verbose = False
        ):
    """
    This function will compare the retrieved phase to the ground truth and determine the accuracy based on
    the comparison
    
    Parameters: 
        system_truth (ndarray): Ground truth wavefront.
        psf_estimate (ndarray): Estimated PSF after retrieval.
        cost_functions (list): History of the cost functions from retrieval iterations.
        verbose (bool): If True, plots cost function.
    Returns:
        dict: Metrics for phase estimate
        """

    phase_estimate = convert_psf_estimate_to_phase(psf_estimate, 
                                                   seal_parameters,
                                                   simulation_elements,
                                                   phase_unwrap_method)# Reconstruct phase from PSF
    
    phase_estimate_metrics = check_phase_estimate(system_truth, 
                                                  phase_estimate,
                                                  simulation_elements)# Compare with truth

    if verbose:
        make_cost_functions_plots(cost_functions)

    return phase_estimate_metrics



##Simulations
#############

def simulate_focused_image(wf_error_to_retrieve, 
                            simulation_elements, 
                            wavelength):
    """
    This function will simulate a focused PSF image

    Returns:
        Wavefront: A wavefront with the known focus/defocus
    """
    ##Use a dictionary for all of this
    wf = Wavefront(simulation_elements['telescope_pupil'] * np.exp(1j * wf_error_to_retrieve.ravel()), wavelength)
    return wf
    ##check with cleaner version that J will send


def simulate_defocused_image(defocus_phase,
                             wf_error_to_retrieve, 
                             seal_parameters, 
                             wavelength, 
                             simulation_elements):#instead of defocus_phase pass wf,test_abberattion
    """
    Simulate a defocus PSF, Function to propagate the wavefront from pupil to focal plane and return focal intensity
    #Want to add another parameter to determine if fourier or zernike
    Parameters:
        defocus_phase (ndarray): Defocus phase term.

    Returns:
        ndarray: Focal Intensity
    """
    
    aberration_to_impart = wf_error_to_retrieve + defocus_phase

    aperture = simulation_elements['aperture'](simulation_elements['pupil_grid'])
    wavefront = Wavefront(aperture * np.exp(1j * aberration_to_impart), wavelength)

    
    prop2f = FraunhoferPropagator(simulation_elements['pupil_grid'],
                                   simulation_elements['focal_grid'], 
                                   focal_length=seal_parameters['focal_length_meters'])
    aberrated_psf = prop2f(wavefront) # Propagate to focal plane, no dict

    focal_intensity = aberrated_psf.intensity

    return focal_intensity.shaped # Return intensity image, using shaped for the return as a proper array

def calculate_defocus_phase(seal_parameters,
                            simulation_elements,
                            defocus_distance):
    '''Take a zernike template that represenets the defocus shape '''
    defocus_template =simulation_elements['zernike_sample_12'][3]
    template_p2v=np.max(defocus_template)-np.min(defocus_template)    
    defocus_p2v = delta_to_p(
                            delta = defocus_distance,
                            f = seal_parameters['focal_length_meters'],
                            D=seal_parameters['pupil_size']
                            )
    defocus_phase = (defocus_template*defocus_p2v)/template_p2v
    
    return defocus_phase



def focus_diverse_phase_retrieval(system_truth,
                                  wavelength, 
                                  phase_diverse_information, 
                                  wf_error_to_retrieve, 
                                  seal_parameters,
                                  simulation_elements):
    
    """
    This function will generate and prep the PSFs to run the phase retrieval
    
    Parameters:
        system_truth (ndarray): Ground truth PSF.
        phase_diverse_inputs (list): Defocus distances to simulate.

    Returns:
        tuple: Estimated PSF and cost function history.
    """
    
    '''
    for defocus_distance in phase_diverse_input: #defocus_distance -> defocus_waves

        #make the call to the function for defocus_phase here 
        defocus_phase = calculate_defocus_phase(seal_parameters,
                                                simulation_elements,
                                                  defocus_distance)
        psf = simulate_defocused_image(defocus_phase,
                                        wf_error_to_retrieve, 
                                        seal_parameters,
                                        wavelength,
                                        simulation_elements)
        defocus_dictionary[defocus_distance] = simulate_defocused_image(defocus_phase)


    psf_estimate, cost_functions = run_phase_retrieval(system_truth, defocus_dictionary, seal_parameters)
    #can check t make sure what we return is a square array
    '''
    # Unpack grid index and defocus template
    (index_x, index_y), defocus_dictionary = phase_diverse_information
    
    # Expecting one defocus distance per entry
    defocus_distance, defocus_phase = list(defocus_dictionary.items())[0]
    
    # Simulate corresponding defocused PSF
    defocused_psf = simulate_defocused_image(
        defocus_phase,
        wf_error_to_retrieve,
        seal_parameters,
        wavelength,
        simulation_elements
    )
    
    # Wrap into a dictionary keyed by physical defocus
    defocus_psfs = {defocus_distance: defocused_psf}
    
    # Run retrieval
    psf_estimate, cost_functions = run_phase_retrieval(
        system_truth,
        defocus_psfs,
        seal_parameters
    )
    return psf_estimate, cost_functions





def simulate_phase_diversity_grid(phase_diverse_inputs, 
                                  seal_parameters, 
                                  file_name_out, 
                                  simulation_elements, 
                                  wavelength,
                                  wf_error_to_retrieve):
    """
    This Function will simulate configs of the defocuses and determine the retrieval accuracy
    Parameters:
        phase_diverse_inputs(list): group of defocus distances
    
    """
    # --- Step 1: Simulate the focused system truth (no defocus)
 
    system_truth = simulate_focused_image(
        wf_error_to_retrieve=wf_error_to_retrieve,
        simulation_elements=simulation_elements,
        wavelength=wavelength
    )

    # --- Step 2: Prepare empty error grid
    dim = seal_parameters['grid_dim']
    phase_diversity_grid = np.zeros((dim, dim))  # To hold RMS errors

    # --- Step 3: Loop through each defocus input
    for phase_diverse_information in phase_diverse_inputs:
        try:
            (index_x, index_y), defocus_dictionary = phase_diverse_information
        except ValueError:
            print(f"Invalid entry: {phase_diverse_information}")
            raise
        print("Sample input format:")
        print(phase_diverse_inputs[0])
        print("Type:", type(phase_diverse_inputs[0]))

        print(f"Entry: {phase_diverse_information}")
        print(f"Type: {type(phase_diverse_information)}")
        (index_x,index_y),_ = phase_diverse_information

        #phase_diverse_information is the 
    
    # Step 3: Iterate over each defocus configuration
    for entry in phase_diverse_inputs:
        try:
            (index_x,index_y), defocus_dict = entry
        except ValueError:
            raise ValueError(f"Expected ((i,j), {{defocus: phase}}), got: {entry}")

        psf_estimate, cost_functions = focus_diverse_phase_retrieval(system_truth, 
                                                                     phase_diverse_information,
                                                                     wf_error_to_retrieve, 
                                                                     seal_parameters,
                                                                     wavelength, 
                                                                     simulation_elements)
        
        metrics = calculate_phase_retrieval_accuracy(system_truth, 
                                                     psf_estimate, 
                                                     cost_functions, 
                                                     seal_parameters,
                                                     simulation_elements,
                                                     phase_unwrap_method = 'phase_unwrap_2d', 
                                                     verbose =True)
        #results.append(metrics)
        phase_diversity_grid[index_x,index_y] = metrics['rms_error']

    np.save(file_name_out, phase_diversity_grid) #will assign name when function runs

    return phase_diversity_grid



def plot_phase_diversity_heat_map(phase_diversity_grid,
                                  heatmap_plot_out):
    """
    This function will visualize the RMS error across a grid
    Parameters:
        results (list): List of dictionaries containing RMS errors for each defocus configuration.
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
def generate_zernike_templates(seal_parameters, 
                               simulation_elements, 
                               physical_defocus_range,
                               zernike_mode_indices=[3],
                               plot_modes=True,
                               p2v_target=1.0):
    """
    Generate Zernike phase templates scaled to match a physical defocus range.

    Returns:
        list: phase_diverse_inputs [((i, mode_idx), {defocus_distance: defocus_phase})]
    """
    pupil_grid = simulation_elements['pupil_grid']
    pupil_size = seal_parameters['pupil_size']
    pupil_dim = seal_parameters['pupil_pixel_dimension']
    focal_length_meters = seal_parameters['focal_length_meters']

    phase_diverse_inputs = []

    # Generate full Zernike basis
    zernike_modes = make_zernike_basis(pupil_dim, pupil_size, pupil_grid)

    # Plot Zernike modes if requested
    if plot_modes:
        n_modes = len(zernike_mode_indices)
        n_cols = min(n_modes, 4)
        n_rows = int(np.ceil(n_modes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, mode_idx in enumerate(zernike_mode_indices):
            mode = zernike_modes[mode_idx].shaped
            im = axes[idx].imshow(mode)
            axes[idx].set_title(f'Zernike Mode {mode_idx}')
            axes[idx].axis('off')
            fig.colorbar(im, ax=axes[idx])
        for ax in axes[n_modes:]:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # Scale each mode to match defocus range
    for mode_idx in zernike_mode_indices:
        template = zernike_modes[mode_idx].shaped
        template_p2v = np.max(template) - np.min(template)

        for i, defocus_distance in enumerate(physical_defocus_range):
            defocus_p2v = delta_to_p(defocus_distance, focal_length_meters, pupil_size)
            defocus_phase = (template * defocus_p2v) / template_p2v
            phase_diverse_inputs.append(((i, mode_idx), {defocus_distance: defocus_phase}))

    return phase_diverse_inputs

##WIP##
##WIP##
def main(seal_parameters,
         phase_diverse_inputs,
         file_name_out,  
         heatmap_plot_out): ##need help to implement this grid and using scales
    
    #BUild Seal Sim
    simulation_elements = build_seal_simulation(seal_parameters)
    wavelength = seal_parameters['wavelength_meter']

    #abberation to retrieve
    # Make sure aberration is a flat Field on the pupil grid
    wf_error_to_retrieve = 0.75 * simulation_elements['zernike_sample_12'][6]
    #run the phase grid
    phase_diversity_grid = simulate_phase_diversity_grid(
        wf_error_to_retrieve=wf_error_to_retrieve,
        phase_diverse_inputs=phase_diverse_inputs,
        seal_parameters=seal_parameters,
        file_name_out=file_name_out,
        simulation_elements=simulation_elements,
        wavelength=wavelength
    )   
    #Last function is to plot, i.e 'plot phase diversity heat map'
    plot_phase_diversity_heat_map(phase_diversity_grid,
                                  heatmap_plot_out)

   


if __name__ == "__main__": 

    #freqs = [(1,0),(0,1),(1,1),(2,0),(0,2)]
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
        'grid_dim': 100,
         }
    #some list of defocus distances 
    seal_parameters['wavelength'] = seal_parameters['wavelength_meter']

    f = seal_parameters['focal_length_meters']
    D = seal_parameters['pupil_size']
        # Set defocus range and generate templates
    simulation_elements = build_seal_simulation(seal_parameters)
    physical_defocus_range = np.linspace(-5e-6, 5e-6, 5)

    grid_dim = seal_parameters['grid_dim']
    defocus_grid = np.linspace(-5e-6, 5e-6, seal_parameters['grid_dim'])
    phase_diverse_inputs = generate_zernike_templates(
        seal_parameters,
        simulation_elements,
        defocus_grid,
        zernike_mode_indices=[3],
        plot_modes=True
)



main(seal_parameters,
     phase_diverse_inputs,
     file_name_out ='example_file_name.npy ',
     heatmap_plot_out= 'example_heatmap.png')