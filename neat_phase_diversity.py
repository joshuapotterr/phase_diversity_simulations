import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from skimage.restoration import unwrap_phase
from skimage.transform import resize
from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
from processing import phase_unwrap_2d
# pep8 styl

##Math Function
###############


# Convert phase into meters using the wavelength
def phase_to_m(phase, wv):
    return phase * wv / (2 * np.pi)

# Calculate defocus distance from peak-to-valley (P2V) error in meters
def p_to_delta(P, f, D):
    return 8 * P * (f/D)**2

# Convert defocus distance into phase error
def delta_to_p(delta, f, D):
    return -1 * delta / (8 * (f/D)**2)
    
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
    #have 256,256 be a variable that i set in seal parameters, this is pupil size in optical simulation

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
    #pass a dictionary for the masking pupil, etc. we use masking pupil purely for the statistics and error analysis
    true_phase = system_truth.phase.shaped # Get true phase
    mask = np.array(masking_pupil.shaped, dtype =bool)# Apply mask to non-zero phase region
    #implement med_subtracted, ie passing the dictionary will help with this(pupil_phase - median blah blah)
    difference_true_vs_estimate = (true_phase - phase_estimate) #Compute difference
    difference_masked = difference_true_vs_estimate[mask]
    rms_error = np.sqrt(np.mean(diff ** 2))
    p2v_error = np.max(diff) - np.min(diff)
    return {'rms_error': rms_error, 'p2v_error': p2v_error, 'difference_image':difference_masked}#also return the difference image
    
def make_cost_functions_plots(cost_functions, filename=None):
    """
    This Function will plot the cost functions for each defocus input
    Parameters:
        cost_functions (list): List of cost function values per iteration.
    """
    plt.clf()
    for i in range(len(cost_functions)):
        plt.semilogy(cost_functions[i], label=f'defocus {i+1}')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.legend()
    plt.grid(True)
    if filename: 
        plt.savefig(filename)#pass the name as an argument to the function
    #before starting new plot, use plt.clf to clear and it doesnt overlap
    plt.clf()


def calculate_phase_retrieval_accuracy(system_truth, psf_estimate, cost_functions, verbose = False):
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

    phase_estimate = convert_psf_estimate_to_phase(psf_estimate, seal_parameters)# Reconstruct phase from PSF
    phase_estimate_metrics = check_phase_estimate(system_truth, phase_estimate)# Compare with truth

    if verbose:
        make_cost_functions_plots(cost_functions)

    return phase_estimate_metrics

##Simulations
#############

def simulate_focused_image(wf_error_to_retrieve, pupil_grid, telescope_pupil, wavelength):#wf_error_to_retrieve will be renamed to wf_error_to_retrieve
    """
    This function will simulate a focused PSF image

    Returns:
        Wavefront: A wavefront with the known focus/defocus
    """
    ##Use a dictionary for all of this
    influence_functions = make_zernike_basis(grid_size, pupil_size, pupil_grid)  # Generate Zernike basis
    wf = Wavefront(telescope_pupil * np.exp(1j * wf_error_to_retrieve), wavelength)
    return wf
    ##check with cleaner version that J will send

'''
def calculate_defocus_params(example_defocus, scale, f, D):
    ##Try to feed it just the wavefront, not example_defocus or scale
    #"give me one wave of p2v at this focal legnth and this diameter"(dictionary)
    #defocus_phase = example_defocus * scale#make it a paramter that is changed regulrly, and is taken by all functions
    p2v_radians = np.max(defocus_phase) - np.min(defocus_phase)
    p2v_m = phase_to_m(p2v_radians, 650e-9)
    delta = p_to_delta(p2v_m, f, D)
    delta = delta if scale > 0 else -1 * delta
    return p2v_radians, delta
    pass
'''
#rescale array so p2v is a certain constant
#figure out p2v, then divide it by itself, and multiply it by whatever constant we want

#want to pass defocus distance instead of phase, and we define phase in this functions
def simulate_defocused_image(defocus_phase,wf_error_to_retrieve, seal_parameters, wavelength, telescope_pupil,focal_grid,pupil_grid):#instead of defocus_phase pass wf,test_abberattion
    """
    Simulate a defocus PSF, Function to propagate the wavefront from pupil to focal plane and return focal intensity
    #Want to add another parameter to determine if fourier or zernike
    Parameters:
        defocus_phase (ndarray): Defocus phase term.

    Returns:
        ndarray: Focal Intensity
    """
    #Use dictionary again
    pupil_grid = make_pupil_grid(seal_parameters['pupil_pixel_dimension'], seal_parameters['pupil_size'])
    focal_grid = make_focal_grid(q=seal_parameters['q'], num_airy=seal_parameters['Num_airycircles'],
                                 pupil_diameter=seal_parameters['pupil_size'],
                                 focal_length=seal_parameters['focal_length'],
                                 reference_wavelength=seal_parameters['wavelength'])
    aperture = make_circular_aperture(seal_parameters['pupil_size'])
    telescope_pupil = aperture(pupil_grid)
    wf_error_to_retrieve = 0.75 * make_zernike_basis(grid_size, seal_parameters['pupil_size'], pupil_grid)[6] # pass it as an argument to be able use it for fourier as well
    aberration_to_impart = wf_error_to_retrieve + defocus_phase # make sure it is square when passed
    wavefront = Wavefront(aperture * np.exp(1j * aberration_to_impart), wavelength)##no dict 
    prop2f = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=seal_parameters['focal_length'])
    aberrated_psf = prop2f(wavefront) # Propagate to focal plane, no dict
    focal_intensity = aberrated_psf.intensity
    return focal_intensity.shaped # Return intensity image, using shaped for the return as a proper array
    ##cleaner version again, 
    ##take the defocus dist in this function and convert to phase we need to impart
    ## zernike modes as well as fourier modes in the simulation_paramaters dict, 
# given you have defined your aperture, wavelength, and some aberration to impart 

def run_phase_retrieval(system_truth, defocus_dictionary, seal_parameters):
    """
    This function will run the actual phase retrieval algorithm
    
    Parameters:
        system_truth (ndarray): PSF with no defocus
        defocus_dictionary (dict): Mapping from defocus distances to PSF images.
        seal_parameters (dict): SEAL system configuration.

    Returns:
        tuple: Estimated PSF and cost function history.
    """

    distance_list = list(defocus_dictionary.keys())
    psf_list = [system_truth] +[defocus_dictionary[key] for key in distance_list]
    dx_list= [seal_parameters['image_dx'] for key in distance_list]
   


    mp= FocusDiversePhaseRetrieval(psf_list,seal_parameters['wavelength'], dx_list, distance_list)
    num_iterations =200
    for i in range(num_iterations):
            psf_estimate = mp.step() 
    

    return psf_estimate, mp.cost_functions


def focus_diverse_phase_retrieval(system_truth,wavelength, phase_diverse_input, wf_error_to_retrieve, defocus_inputs, defocus_dictionary, 
                                  seal_parameters,defocus_phase,simulation_elements):
    """
    This function will generate and prep the PSFs to run the phase retrieval
    
    Parameters:
        system_truth (ndarray): Ground truth PSF.
        phase_diverse_inputs (list): Defocus distances to simulate.

    Returns:
        tuple: Estimated PSF and cost function history.
    """
    #Build defocused wavefront in tzis function, ie this is where we iterate through and build inputs we want
    #Makes most sense to do this in waves
    #So keep defocus_phase here
    #make another function that rescales the array/wavefront and than call it here 
    

    #Defocus Dictionary
    """
    defocus_dictionary = {
        delta: simulate_defocused_image(defocus_phase, wf_error_to_retrieve, pupil_grid, focal_grid, telescope_pupil, 
                                        seal_parameters['wavelength'],seal_parameters['focal_length']))
    
        for delta, defocus_phase in defocus_inputs.items()
    }
    return run_phase_retrieval(system_truth, defocus_dictionary, seal_parameters)
    """
#test this code

    for defocus_distance in phase_diverse_input: #defocus_distance -> defocus_waves
        #make the call to the function for defocus_phase here 
        psf = simulate_defocused_image(defocus_phase, wf_error_to_retrieve, seal_parameters, wavelength, simulation_elements)
        defocus_dictionary[defocus_distance] = simulate_defocused_image(defocus_phase)


    psf_estimate, cost_functions = run_phase_retrieval(system_truth, defocus_dictionary, seal_parameters)
    #can check t make sure what we return is a square array

    return psf_estimate, cost_functions





def simulate_phase_diversity_grid(phase_diverse_inputs, seal_parameters, file_name_out, simulation_elements, wavelength):
    """
    This Function will simulate configs of the defocuses and determine the retrieval accuracy
    Parameters:
        phase_diverse_inputs(list): group of defocus distances
    
    """
    system_truth = simulate_focused_image() #true wf
    #results = []
    dim = seal_parameters['grid_dim'] 
    phase_diversity_grid = np.zeros((dim,dim))
    #could we define pupil and focal grid here, generate the influence function and test ab, and system truth then run 
    #a for loop for i,j, defocus_inouts in defocus_input_grid.items and run focus_diverse_phase retrieval 
    #for psf_estimate and cost_function, than run convert_psf_estimate to phase for pupil phase
    #than metrics with check_phase_estimate, than fill grid[i,j] with metrics[rms_error], than save grid
    #so psf_estimate,cost_functions,pupil_phase,metrics would all get run in the for loop iterating 

    for (index_x, index_y), phase_diverse_information in phase_diverse_inputs:
        index, phase_diverse_input = phase_diverse_information
        index_x, index_y = index
        psf_estimate, cost_functions = focus_diverse_phase_retrieval(system_truth, phase_diverse_input, seal_parameters, wavelength, simulation_elements)
        metrics = calculate_phase_retrieval_accuracy(system_truth, psf_estimate, cost_functions, seal_parameters,simulation_elements,
                                                     phase_unwrap_method = 'phase_unwrap_2d', verbose =True)
        #results.append(metrics)
        phase_diversity_grid[index_x,index_y] = metrics['rms_error']

    np.save(file_name_out, phase_diversity_grid) #will assign name when function runs
    return phase_diversity_grid

#want final plot to be in 


def plot_phase_diversity_heat_map(phase_diversity_grid,heatmap_plot_out):
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
    plt.save(heatmap_plot_out)
    plt.clf()
    #keep in mind extent logic, need to tell physical x y labels 

#in main, spin up an array that shows what parameters we are running over, which will be defocus dist
def main(seal_parameters, phase_diverse_inputs, file_name_out, phase_diversity_grid, heatmap_plot_out): ##need help to implement this grid and using scales
    
    #Necessary to get Results for a Plot, using a predescribed unit
    simulate_phase_diversity_grid(phase_diverse_inputs, seal_parameters, file_name_out)
    #Last function is to plot, i.e 'plot phase diversity heat map'
    plot_phase_diversity_heat_map(phase_diversity_grid,heatmap_plot_out)

def build_seal_simulation(seal_parameters):
    """
    Function to setup the SEAL optical system parameters
    
    Parameters:
        seal_parameters(dict): Config for SEAL system
    
    """

    #Create a focal grid based on system parameters
    #focal_grid = make_focal_grid(q=q, num_airy=num_airy, pupil_diameter=pupil_size, focal_length=focal_length, reference_wavelength=wavelength)
    focal_grid = make_focal_grid(
        q=seal_parameters['q'], 
        num_airy=seal_parameters['Num_airycircles'], 
        pupil_diameter = seal_parameters['pupil_size'], 
        focal_length=seal_parameters['focal_length_meter'], 
        reference_wavelength =seal_parameters['wavelength'] )
    # Setup apertures for telescope and masking pupils
    pupil_grid = make_pupil_grid(seal_parameters['pupil_pixel_dimension'], seal_parameters['pupil_size'])
    aperture = make_circular_aperture(seal_parameters['pupil_size'])
    telescope_pupil = aperture(pupil_grid)
    small_aperture = make_circular_aperture(seal_parameters['small_pupil_size'])
    masking_pupil = small_aperture(pupil_grid)
    # Return all components as a dictionary
    simulation_elements = {
        'pupil_grid': pupil_grid,
        'focal_grid': focal_grid,
        'aperture': aperture,
        'telescope_pupil': telescope_pupil,
        'masking_pupil': masking_pupil
    }

    return simulation_elements
    

    ##Call for Main Function
    ########################
    #in main, spin up an array that shows what parameters we are running over, which will be defocus dist
def main(seal_parameters,simulation_elements, phase_diverse_inputs, file_name_out, heatmap_plot_out): ##need help to implement this grid and using scales
    wavelength = seal_parameters['wavelength_meter']
    #Necessary to get Results for a Plot, using a predescribed unit
    phase_diversity_grid = simulate_phase_diversity_grid(phase_diverse_inputs, seal_parameters, file_name_out)
    #Last function is to plot, i.e 'plot phase diversity heat map'
    plot_phase_diversity_heat_map(phase_diversity_grid,heatmap_plot_out)

    '''
    ##Defining phase Diverse Inputs
    phase_diverse_inputs = []
    num_configs = 5  # Adjust the number of defocus configurations as desired.
    for i in range(num_configs):
        # Create a defocus phase with a constant phase offset.
        defocus_value = i * 0.1  # Example defocus distance
        defocus_phase = np.ones(simulation_elements['pupil_grid'].shape) * defocus_value
        # Assign the same defocus configuration to grid positions
        phase_diverse_inputs.append(((i, i), [(defocus_value, defocus_phase)]))
    '''
    zernike_template()

        #So, remember that defocus is a zernike polynomial -- it has a particular shape.
        #So the goal is to start with some template of that particular shape and then rescale it so that the peak to valley of that shape is useful to you

###WIP###
#generating a zernike basis over SEAL pupil gird
#phase diverse values need to be passed


#placeholder for range of defocus distance
#implemented a step,
#take this as an input as a fcn:   physical_defocus_range = np.arange(1,1+1,1) ##make not into a function, maybe right above the main call
#instead of pupil_pixel_dimension set e.g num_zernike_modes = 4, 
#check to make sure using right make_zernike_basis

def zernike_template(seal_parameters, simulation_elements):
    zernike_modes = make_zernike_basis(seal_parameters['pupil_pixel_dimension'], seal_parameters['pupil_size'], simulation_elements['pupil_grid'])
    defocus_template = zernike_modes[3].shaped  
#Calculate the original peak-to-valley (P2V) value of this Zernike defocus shape which is the template value we normalize against.
    template_peak_to_valley = np.max(defocus_template) - np.min(defocus_template)


# Create a dictionary mapping each physical defocus distance to a properly scaled Zernike phase
#build a list of phase maps corresponding to each physical defocus value

    phase_diverse_inputs = []
    for i, defocus_distance_value in enumerate(physical_defocus_range):
    #required P2V phase shift (in radians) for this physical defocus distance
        defocus_peak_to_valley = delta_to_p(defocus_distance_value, seal_parameters['focal_length_meters'], seal_parameters['pupil_size'])

    #Rescale the Zernike defocus template so its P2V matches the desired one
        defocus_phase = defocus_template * defocus_peak_to_valley / template_peak_to_valley
    #run this in ipython 

    # Append into list as ((i, i), {defocus_distance: defocus_phase})
        phase_diverse_inputs.append(((i, i), {defocus_distance_value: defocus_phase}))

        return phase_diverse_inputs


def delta_to_p(delta, f, D):
    return -1 * delta / (8 * (f/D)**2)

# Create a dictionary mapping each physical defocus distance to a properly scaled Zernike phase
#build a list of phase maps corresponding to each physical defocus value

###WIP###


    #build something similar to line 368 of old code
    #given a defocus dist, finding p2v, then rescale the zernike mode 
    # to match the p2v of the defocus distance 
    #goal: to have defocus distance and the zernike mode have the same p2v by rescaling the zernike mode to match
    #pulling zernike mode from the dictionary i will be defining, and passing it to this fucntion 
    #defocus distance is going to be define d from scratch
    #recreate what J sent with fourier modes 

    ##Fourier Modes
    '''
    make_fourier_transform(input_grid[, ...]) ->fourier transform object
    make_fft_grid(input_grid[, q, fov, shift]) -> calculate grid returned by a fast fourier transform
    get_fft_parameters(fft_grid, input_grid) - >Try to reconstruct the FFT parameters of a grid.
    is_fft_grid(grid, input_grid)-> Returns whether grid is a native FFT grid of input_grid or not.

    ###use make_fourier_basis

    should we be using a fast, matrix, naive, chirp-z, zoom fast fourier transform object?
    
    
    '''

##WIP##
def make_fourier_modes(simulation_elements, seal_parameters, num_fourier_modes, p2v_target=1.0):
    """
    Create a list of scaled Fourier phase modes.

    Parameters:
        simulation_elements (dict): Dictionary containing simulation grids and pupils.
        seal_parameters (dict): Dictionary of SEAL parameters including pupil size.
        num_fourier_modes (list of tuples): List of (kx, ky) frequency index pairs to extract from the basis.
        p2v_target (float): Desired Peak-to-Valley phase in radians.

    Returns:
        list of ndarray: Each entry is a 2D Fourier phase map scaled to match p2v_target.
    """
    pupil_grid = simulation_elements['pupil_grid']
    aperture = simulation_elements['telescope_pupil']

    # Generate the Fourier basis
    fourier_modes = make_fourier_basis(seal_parameters['pupil_pixel_dimension'], seal_parameters['pupil_size'], pupil_grid)
    #ipython this and compare with make_fourier_basis(blah[pupil_grid], blah[pupil_size], sort_by_energy=true)


    #make_fourier_basis returns a list object of modal functions, where each mode is accessed usin index
    #we loop so that we standardize the access pattern, we could use just 'i' in the loop but other hcipy things use idx so i want to be consistent
    #is there an imaginary part because both sine and cosine is returned?
    ##--> dont have to deal with them lol

    # Use all modes if no specific num_fourier_modes provided
    if num_fourier_modes is None:
        num_fourier_modes = [(i,) for i in range(len(fourier_modes))]

    scaled_modes = []

    mode = fourier_modes.shaped  # Extract and reshape mode

        # compute original Peak-to-Valley
    mode_p2v = np.max(mode) - np.min(mode)

        # scale to  P2V that we want, i.e the target
    scaled_mode = (mode * p2v_target) / mode_p2v

        #append again
    scaled_mode.append(scaled_mode)

    return scaled_mode
##WIP##
#pull whatever we need, use.shaped, try and model after the zernike basis function above
if __name__ == "__main__": 
    phase_diverse_inputs = [(1,2)]
    file_name_out = 'test_run.py'
    heatmap_plot_out = 'heatmap_test_run.png'

    #write a different dictionary for this
    f = focal_length
    D = pupil_size
    #Define Variables
    seal_parameters = {
        'image_dx': 2.0071, # 
        'efl_mm': 500, # SEAL effective focal length, mm
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
    
    main(seal_parameters, phase_diverse_inputs, file_name_out, heatmap_plot_out)