import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from skimage.restoration import unwrap_phase
from skimage.transform import resize
from image_sharpening import FocusDiversePhaseRetrieval, mft_rev, InstrumentConfiguration
from processing import phase_unwrap_2d

# -------------------------------------------------------------------
# 1) MATH‐HELPERS (phase <-> meters, p2v <-> defocus distance)
# -------------------------------------------------------------------

def phase_to_m(phase, wv):
    """
    Convert a phase (in radians) to meters, given wavelength wv (in meters).
    """
    return phase * wv / (2 * np.pi)

def p_to_delta(P, f, D):
    """
    Given a P2V phase error P (in meters), focal length f, and pupil diameter D,
    compute the defocus distance delta (in meters).
    """
    return 8 * P * (f / D) ** 2

def delta_to_p(delta, f, D):
    """
    Given a defocus distance delta (in meters), focal length f, and pupil diameter D,
    compute the corresponding P2V phase error (in meters).
    """
    return -1 * delta / (8 * (f / D) ** 2)

# -------------------------------------------------------------------
# 2) BUILD THE SEAL “SIMULATION ELEMENTS” ONCE
# -------------------------------------------------------------------

def build_seal_simulation(seal_parameters):
    """
    Create and return a dictionary of all reusable simulation objects:
      - pupil_grid      : HCIPy PupilGrid
      - focal_grid      : HCIPy FocalGrid
      - aperture        : HCIPy CircularAperture
      - telescope_pupil : boolean mask of the pupil
      - masking_pupil   : boolean mask of a smaller pupil (if needed)
      - zernike_modes_12: list of first 12 Zernike 2D arrays (mode.shaped)
      - fourier_modes_12: list of first 12 Fourier 2D arrays (mode.shaped)
    """
    # Pull needed parameters (in SI units)
    pupil_dim = seal_parameters['pupil_pixel_dimension']   # e.g. 256
    pupil_size = seal_parameters['pupil_size']            # e.g. 10.12e-3 m
    focal_length = seal_parameters['focal_length_meters'] # e.g. 0.5 m
    q = seal_parameters['q']
    num_airy = seal_parameters['Num_airycircles']
    wavelength = seal_parameters['wavelength_meter']

    # 2A) Build PupilGrid and Circular Apertures
    pupil_grid = make_pupil_grid(pupil_dim, pupil_size)
    aperture = make_circular_aperture(pupil_size)
    telescope_pupil = aperture(pupil_grid)  # boolean mask
    small_aperture = make_circular_aperture(seal_parameters['small_pupil_size_meter'])
    masking_pupil = small_aperture(pupil_grid)

    # 2B) Build FocalGrid
    focal_grid = make_focal_grid(
        q=q,
        num_airy=num_airy,
        pupil_diameter=pupil_size,
        focal_length=focal_length,
        reference_wavelength=wavelength
    )

    # 2C) Precompute first 12 Zernike modes (shaped)
    raw_zernikes = make_zernike_basis(pupil_dim, pupil_size, pupil_grid)
    zernike_modes_12 = [mode.shaped for mode in raw_zernikes[:12]]

    # 2D) Precompute first 12 Fourier modes (6 along x, 6 along y)
    #     We choose freq_pairs = [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),
    #                              (0,1),(0,2),(0,3),(0,4),(0,5),(0,6)]
    freq_pairs = [(n, 0) for n in range(1, 7)] + [(0, n) for n in range(1, 7)]
    # Convert each (kx, ky) into rad/m: kx_phys = kx * 2π / D, etc.
    kx = np.array([pair[0] * 2 * np.pi / pupil_size for pair in freq_pairs])
    ky = np.array([pair[1] * 2 * np.pi / pupil_size for pair in freq_pairs])
    fourier_coords = UnstructuredCoords((kx, ky))
    fourier_grid = CartesianGrid(fourier_coords)
    raw_fourier = make_fourier_basis(pupil_grid, fourier_grid)
    fourier_modes_12 = [mode.shaped for mode in raw_fourier]  # yields 12 modes

    # Pack everything into a single dictionary:
    simulation_elements = {
        'pupil_grid': pupil_grid,
        'focal_grid': focal_grid,
        'aperture': aperture,
        'telescope_pupil': telescope_pupil,
        'masking_pupil': masking_pupil,
        'zernike_modes_12': zernike_modes_12,
        'fourier_modes_12': fourier_modes_12
    }
    return simulation_elements

# -------------------------------------------------------------------
# 3) GENERATE ZERNIKE PHASE TEMPLATES
# -------------------------------------------------------------------

def generate_zernike_templates(
        seal_parameters,
        simulation_elements,
        physical_defocus_range,
        zernike_mode_indices=[3],
        plot_modes=True,
        p2v_target=1.0
    ):
    """
    For each requested Zernike mode index and each physical defocus distance,
    produce a scaled 2D phase map whose P2V matches delta_to_p(defocus_distance).

    Returns a list of tuples:
      [ ((i, mode_idx), {defocus_distance: phase_2D_array}), ... ]

    - zernike_mode_indices: list of integers (e.g. [3,4,5]).
    - physical_defocus_range: 1D array of defocus distances (meters).
    - p2v_target: if you want each template's P2V to be exactly this value (rad),
      you can override "delta_to_p" scaling—instead we use actual “delta_to_p”
      so p2v_target is not strictly necessary here.
    """
    pupil_grid = simulation_elements['pupil_grid']
    pupil_size = seal_parameters['pupil_size']
    focal_length = seal_parameters['focal_length_meters']

    phase_diverse_inputs = []

    # Grab the entire Zernike basis once:
    raw_zernikes = make_zernike_basis(
        seal_parameters['pupil_pixel_dimension'],
        seal_parameters['pupil_size'],
        pupil_grid
    )

    # 3A) (Optional) Plot the unscaled Zernike modes:
    if plot_modes:
        n_modes = len(zernike_mode_indices)
        n_cols = min(n_modes, 4)
        n_rows = int(np.ceil(n_modes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes_flat = axes.flatten() if n_modes > 1 else [axes]

        for ax, mode_idx in zip(axes_flat, zernike_mode_indices):
            mode2D = raw_zernikes[mode_idx].shaped
            im = ax.imshow(mode2D, cmap='RdBu')
            ax.set_title(f'Zernike #{mode_idx}')
            ax.axis('off')
            fig.colorbar(im, ax=ax)

        # Turn off any leftover subplots if n_modes < n_cols*n_rows
        for ax in axes_flat[n_modes:]:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # 3B) For each requested Zernike index and each physical defocus:
    for mode_idx in zernike_mode_indices:
        template = raw_zernikes[mode_idx].shaped
        template_p2v = np.max(template) - np.min(template)

        for i, defocus_distance in enumerate(physical_defocus_range):
            # “delta_to_p” gives P2V phase in meters → convert to radians: phase_to_m(...)
            p2v_phase = delta_to_p(
                defocus_distance,
                focal_length,
                pupil_size
            ) * (2 * np.pi / seal_parameters['wavelength_meter'])
            # scale the template so it has exactly p2v_phase radians P2V:
            scaled_phase = (template * p2v_phase) / (template_p2v)

            # append: ((i, mode_idx), {defocus_distance: scaled_phase})
            phase_diverse_inputs.append(
                ((i, mode_idx), {defocus_distance: scaled_phase})
            )

    return phase_diverse_inputs

# -------------------------------------------------------------------
# 4) GENERATE FOURIER PHASE TEMPLATES
# -------------------------------------------------------------------

def generate_fourier_templates(
        seal_parameters,
        simulation_elements,
        num_modes=3,
        plot_modes=True,
        p2v_target=1.0
    ):
    """
    Build the first `num_modes` horizontal Fourier modes (sine/cosine along x)
    scaled so each has exactly P2V = p2v_target (in radians).  

    Returns a list of tuples:
      [ ((mode_idx, mode_idx), {p2v_target: phase_2D_array}), ... ]

    - num_modes: how many (kx,0) pairs to generate (i.e. modes at kx=1..num_modes).
    - p2v_target: desired P2V (radians) of each returned Fourier template.
    """
    pupil_grid = simulation_elements['pupil_grid']
    pupil_size = seal_parameters['pupil_size']

    # 4A) Construct kx_ky pairs = [(1,0), (2,0), ..., (num_modes,0)]
    freq_pairs = [(n, 0) for n in range(1, num_modes+1)]

    # Convert them into physical rad/m coordinates:
    kx = np.array([n * 2 * np.pi / pupil_size for (n, _) in freq_pairs])
    ky = np.array([0  * 2 * np.pi / pupil_size for (_, _) in freq_pairs])  # all zeros for ky
    fourier_coords = UnstructuredCoords((kx, ky))
    fourier_grid = CartesianGrid(fourier_coords)

    # 4B) Generate exactly those Fourier basis modes over pupil_grid
    raw_fourier_basis = make_fourier_basis(pupil_grid, fourier_grid)
    # raw_fourier_basis is a list of ModeBasis objects (both cos and sin).
    # Because we passed N frequency pairs, make_fourier_basis returns 2*N modes:
    #   [cos(k1·r), sin(k1·r), cos(k2·r), sin(k2·r), ... cos(kN·r), sin(kN·r)]
    # We'll take the first 2*num_modes entries to form our “num_modes” pairs.

    # 4C) (Optional) Plot the unscaled Fourier modes:
    if plot_modes:
        total_modes = len(raw_fourier_basis)
        n_cols = min(total_modes, 4)
        n_rows = int(np.ceil(total_modes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes_flat = axes.flatten() if total_modes > 1 else [axes]

        for idx, mode in enumerate(raw_fourier_basis[:2*num_modes]):
            mode2D = mode.shaped
            im = axes_flat[idx].imshow(mode2D, cmap='RdBu')
            axes_flat[idx].set_title(f'Fourier #{idx+1}')
            axes_flat[idx].axis('off')
            fig.colorbar(im, ax=axes_flat[idx])

        for ax in axes_flat[2*num_modes:]:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # 4D) Now scale each of those 2*num_modes Fourier modes so P2V = p2v_target:
    phase_diverse_inputs = []
    for idx, mode in enumerate(raw_fourier_basis[:2*num_modes]):
        template = mode.shaped
        template_p2v = np.max(template) - np.min(template)
        # scale so that P2V(template) → p2v_target:
        scaled = (template * p2v_target) / template_p2v
        phase_diverse_inputs.append(
            ((idx, idx), {p2v_target: scaled})
        )

    return phase_diverse_inputs

# -------------------------------------------------------------------
# 5) FOCUS‐DIVERSE PHASE RETRIEVAL CORE
# -------------------------------------------------------------------

def convert_psf_estimate_to_phase(psf_estimate, seal_parameters, telescope_pupil, phase_unwrap=None):
    """
    Convert the retrieved PSF estimate (complex field) into a “pupil‐plane” phase map.
    Internally uses mft_rev(...) to invert the MFT, then unwraps/resizes.

    Returns a 2D array of the pupil‐phase (radians), masked by telescope_pupil.
    """
    pupil_dim = seal_parameters['pupil_pixel_dimension']
    seal_configuration = InstrumentConfiguration(seal_parameters)

    # Run reverse multi‐frequency transform to get complex pupil‐field estimate
    raw_field = mft_rev(psf_estimate, seal_configuration)
    raw_phase = np.angle(raw_field)

    # Optionally unwrap:
    if phase_unwrap == "phase_unwrap_2d":
        raw_phase = phase_unwrap_2d(raw_phase)
    elif phase_unwrap == "unwrap_phase":
        raw_phase = unwrap_phase(raw_phase)

    # Resize back to (pupil_dim × pupil_dim) and mask outside pupil:
    phase_resized = resize(raw_phase, (pupil_dim, pupil_dim))
    return phase_resized * telescope_pupil.shaped

def check_phase_estimate(system_truth, phase_estimate, masking_pupil):
    """
    Compare “system_truth” (Wavefront object) to your 2D array “phase_estimate.”  
    Returns:
      - rms_error  : sqrt(mean( (truth – estimate)^2 )) inside the pupil
      - p2v_error  : peak‐to‐valley of (truth – estimate) inside pupil
      - difference_image: the 1D array of pixel‐by‐pixel differences inside the mask
    """
    true_phase = system_truth.phase.shaped
    mask = np.array(masking_pupil.shaped, dtype=bool)

    diff = (true_phase - phase_estimate)[mask]
    rms_error = np.sqrt(np.mean(diff**2))
    p2v_error = np.max(diff) - np.min(diff)
    return {
        'rms_error': rms_error,
        'p2v_error': p2v_error,
        'difference_image': diff
    }

def make_cost_functions_plots(cost_functions, filename=None):
    """
    cost_functions is a list of lists, one per defocus image.
    Plot each on a semilog y‐axis. Optionally save to disk.
    """
    plt.clf()
    for idx, costs in enumerate(cost_functions, start=1):
        plt.semilogy(costs, label=f'Defocus {idx}')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.legend()
    plt.grid(True)
    if filename:
        plt.savefig(filename)
    plt.clf()

def run_phase_retrieval(system_truth_psf, defocus_dictionary, seal_parameters):
    """
    Runs FocusDiversePhaseRetrieval:
      - system_truth_psf: 2D numpy array (in‐focus PSF intensity)
      - defocus_dictionary: {defocus_distance: psf_2D_array, ...}
      - seal_parameters: must contain 'image_dx' and 'wavelength'
    Returns (psf_estimate_complex, cost_functions_list).
    """
    distance_list = list(defocus_dictionary.keys())
    psf_list = [system_truth_psf] + [defocus_dictionary[d] for d in distance_list]
    dx_list = [seal_parameters['image_dx']] * len(distance_list)

    mp = FocusDiversePhaseRetrieval(
        psf_list,
        seal_parameters['wavelength_meter'],
        dx_list,
        distance_list
    )
    for _ in range(200):
        psf_estimate = mp.step()
    return psf_estimate, mp.cost_functions

def calculate_phase_retrieval_accuracy(
        system_truth_wavefront,
        psf_estimate_complex,
        cost_functions,
        seal_parameters,
        simulation_elements,
        phase_unwrap_method=None,
        verbose=False
    ):
    """
    Given:
      - system_truth_wavefront: Wavefront object containing true pupil‐phase
      - psf_estimate_complex: the output of run_phase_retrieval (complex 2D array)
      - cost_functions: list of cost hist lists
      - seal_parameters & simulation_elements
    Return a dictionary of errors (rms and p2v).
    """
    telescope_pupil = simulation_elements['telescope_pupil']
    phase_estimate = convert_psf_estimate_to_phase(
        psf_estimate_complex,
        seal_parameters,
        telescope_pupil,
        phase_unwrap_method
    )
    metrics = check_phase_estimate(
        system_truth_wavefront,
        phase_estimate,
        simulation_elements['masking_pupil']
    )
    if verbose:
        make_cost_functions_plots(cost_functions)
    return metrics

# -------------------------------------------------------------------
# 6) EXAMPLE USAGE (“main”)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # 6A) Re‐define seal_parameters here (redundant if already in your script):
    seal_parameters = {
        'pupil_size': 10.12e-3,
        'small_pupil_size_meter': 9.5e-3,
        'pupil_pixel_dimension': 256,
        'focal_length_meters': 500e-3,
        'q': 16,
        'Num_airycircles': 16,
        'wavelength_meter': 650e-9,
        'image_dx': 2.0071,
        'grid_dim': 100
    }

    # 6B) Build all grids, pupils, and store first 12 modes:
    simulation_elements = build_seal_simulation(seal_parameters)

    # 6C) Choose a test aberration (e.g. “coma” from the precomputed Zernikes):
    test_aberration = 0.75 * simulation_elements['zernike_modes_12'][6]

    # 6D) Simulate the in‐focus “ground truth” PSF:
    pupil_grid = simulation_elements['pupil_grid']
    telescope_pupil = simulation_elements['telescope_pupil']
    wavelength = seal_parameters['wavelength_meter']
    f_length = seal_parameters['focal_length_meters']
    prop_to_focal = FraunhoferPropagator(
        pupil_grid,
        simulation_elements['focal_grid'],
        focal_length=f_length
    )

    wf_focused = Wavefront(
        telescope_pupil * np.exp(1j * test_aberration),
        wavelength
    )
    focal_field_focused = prop_to_focal.forward(wf_focused)
    psf_focused = np.abs(focal_field_focused.electric_field.reshape(
        simulation_elements['focal_grid'].shape
    ))**2

    # 6E) Generate a handful of Zernike defocus templates over –5µm to +5µm:
    physical_defocus_range = np.linspace(-5e-6, 5e-6, 5)  # 5 points
    zernike_indices = [3]  # pure defocus (Zernike index 3)
    zernike_templates = generate_zernike_templates(
        seal_parameters,
        simulation_elements,
        physical_defocus_range,
        zernike_mode_indices=zernike_indices,
        plot_modes=True,
        p2v_target=1.0   # not strictly used, since delta_to_p is computed from defocus_distance
    )

    # 6F) Build a defocus‐to‐PSF dictionary using the first Zernike mode:
    defocus_dict = {}
    for ((i, mode_idx), mapping) in zernike_templates:
        # mapping is {defocus_distance: scaled_phase_2D}
        defocus_distance, zern_phase = next(iter(mapping.items()))
        # Combine test_aberration + zernike defocus phase:
        wf_def = Wavefront(
            telescope_pupil * np.exp(1j * (test_aberration + zern_phase)),
            wavelength
        )
        focal_field_def = prop_to_focal.forward(wf_def)
        psf_def = np.abs(focal_field_def.electric_field.reshape(
            simulation_elements['focal_grid'].shape
        ))**2
        defocus_dict[defocus_distance] = psf_def

    # 6G) Run the phase retrieval:
    psf_estimate, cost_hist = run_phase_retrieval(
        system_truth_psf=psf_focused,
        defocus_dictionary=defocus_dict,
        seal_parameters=seal_parameters
    )

    # 6H) Compare retrieved phase to known truth:
    metrics = calculate_phase_retrieval_accuracy(
        system_truth_wavefront=wf_focused,
        psf_estimate_complex=psf_estimate,
        cost_functions=cost_hist,
        seal_parameters=seal_parameters,
        simulation_elements=simulation_elements,
        phase_unwrap_method='phase_unwrap_2d',
        verbose=True
    )
    print("RMS error (rad):", metrics['rms_error'])
    print("P2V error (rad):", metrics['p2v_error'])

    # 6I) (Optional) If you want a heatmap over a 100×100 defocus grid,
    #      use simulate_phase_diversity_grid & plot_phase_diversity_heat_map
    #      (omitted here for brevity).

