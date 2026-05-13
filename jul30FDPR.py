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
seal_parameters = {
        'image_dx': 2.0071, # 
        'efl': 500, # SEAL effective focal length, mm
        'wavelength_micron': 0.65, # SEAL center wavelength, microns- >prysm
        'wavelength_meter': 650e-9,#SEAL center wavelength, meters -> hcipy
        'pupil_size': 10.12e-3, # Keck entrance pupil diameter
        'small_pupil_size_meter': 9.5e-3,
        'pupil_pixel_dimension': 256,
        'focal_length_meters': 500e-3,#this is 500mm
        'q': 16,
        'Num_airycircles': 16,
        'grid_dim':10
         }
def build_seal_simulation(seal_parameters):
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
    defocus_template = zernike_sample_256[3]
    wf_error_to_retrieve = zernike_sample_256[6]
    simulation_elements = {
        'pupil_grid': pupil_grid,
        'focal_grid': focal_grid,
        'aperture': aperture,
        'telescope_pupil': telescope_pupil,
        'masking_pupil': masking_pupil,
        'zernike_sample_256' : zernike_sample_256,
        'fourier_sample_84' : fourier_sample_84,
        'defocus_template':defocus_template, 
        'wf_error_to_retrieve':wf_error_to_retrieve
        }
    return simulation_elements

def calculate_defocus_params(simulation_elements, seal_parameters, defocus_distance):
    f= seal_parameters['focal_length_meters']
    D= seal_parameters['pupil_size']
    template_p2v=np.max(simulation_elements['defocus_template'])-np.min(simulation_elements['defocus_template']) 
    unit_defocus = simulation_elements['defocus_template'] / template_p2v
    defocus_p2v = delta_to_p(
                            delta = defocus_distance,
                            f = seal_parameters['focal_length_meters'],
                            D=seal_parameters['pupil_size']
                            )
    defocus_phase = defocus_p2v * unit_defocus
    p2v_radians = np.max(defocus_phase) - np.min(defocus_phase)
    p2v_m = phase_to_m(p2v_radians, 650e-9)
    delta = p_to_delta(p2v_m, f, D)
    return defocus_phase, p2v_radians, delta

def propagate_image(defocus_phase, simulation_elements):
    # Set up the propagator to simulate wavefront propagation
    prop_p2f = FraunhoferPropagator(simulation_elements['pupil_grid'],
                                    simulation_elements['focal_grid'], 
                                    seal_parameters['focal_length_meters'])
    # Combine the test aberration and defocus phase
    combined_phase = (simulation_elements['wf_error_to_retrieve'] 
                      + defocus_phase).ravel() 
    # Apply the phase to the telescope pupil
    pupil_field = (simulation_elements['telescope_pupil'] 
                   * np.exp(1j * combined_phase))
    # Create a wavefront object
    wavefront = Wavefront(pupil_field, 
                          seal_parameters['wavelength_meter'])
    # Propagate the wavefront to the focal plane
    focal_field = prop_p2f.forward(wavefront)
    # Calculate the intensity of the focal plane image
    focal_intensity = np.abs(focal_field.electric_field.reshape(simulation_elements['focal_grid'].shape))**2
    return focal_intensity
def focused_image_generation(simulation_elements):
    focused_image_phase= np.zeros_like(simulation_elements['defocus_template'])
    return propagate_image(focused_image_phase, simulation_elements)
def defocused_image_generation(defocus_distance, 
                               simulation_elements, 
                               seal_parameters,
                               verbose = False):
    defocus_phase, p2v_radians, delta = calculate_defocus_params(simulation_elements, 
                                                   seal_parameters, 
                                                   defocus_distance)
    focal_intensity = propagate_image(defocus_phase, simulation_elements)
    if verbose:
        return focal_intensity, defocus_phase, p2v_radians, delta
    return focal_intensity
def generate_psf_batch(defocus_distances, 
                       simulation_elements, 
                       seal_parameters, 
                       verbose=False):
    # Focused image
    focused = focused_image_generation(simulation_elements)
    defocused_list = []
    phase_list = []
    p2v_list = []
    delta = []
    for dz in defocus_distances:
        defocus_phase, p2v_radians, delta = calculate_defocus_params(simulation_elements, seal_parameters, dz)
        defocused = propagate_image(defocus_phase, simulation_elements)
        defocused_list.append(defocused)

    psf_dict = {
        'focused': focused,
        'defocused': defocused_list,
    }
    if verbose:
        psf_dict.update({
            'phases': phase_list,
            'p2v_radians': p2v_list,
            'delta': delta,
        })
dz_list = np.linspace(-0.01, 0.01, 5)  # e.g. 5 symmetric defocus values
sim_elements = build_seal_simulation(seal_parameters)

results = generate_psf_batch(dz_list, sim_elements, seal_parameters, verbose=True)

# View the PSFs
plt.figure()
plt.imshow(np.log10(results['focused'] / results['focused'].max()), vmin=-5)
plt.title('Focused PSF')
plt.colorbar()

for i, def_psf in enumerate(results['defocused']):
    plt.figure()
    plt.imshow(np.log10(def_psf / def_psf.max()), vmin=-5)
    plt.title(f'Defocused PSF: Δz = {dz_list[i]:.4f} m')
    plt.colorbar()
