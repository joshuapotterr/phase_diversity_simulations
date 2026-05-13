
from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
from skimage.transform import resize

def phase_to_m(phase, wv):
    return phase * wv / (2 * np.pi)

def p_to_delta(P, f, D):
    return 8 * P * (f/D)**2

def delta_to_p(delta, f, D):
    return -1 * delta / (8 * (f/D)**2)

def calculate_defocus_params(example_defocus, scale, f, D, wavelength):
    defocus_phase = example_defocus * scale
    p2v_radians = np.max(defocus_phase) - np.min(defocus_phase)
    p2v_m = phase_to_m(p2v_radians, wavelength)
    delta = p_to_delta(p2v_m, f, D)
    delta = delta if scale > 0 else -1 * delta
    return p2v_radians, delta

def propagate_image(defocus_phase, test_ab, telescope_pupil, wavelength, focal_grid, prop_p2f):
    combined_phase = (test_ab + defocus_phase).ravel()
    pupil_field = telescope_pupil * np.exp(complex(0, 1) * combined_phase)
    wavefront = Wavefront(pupil_field, wavelength)
    focal_field = prop_p2f.forward(wavefront)
    focal_intensity = np.abs(focal_field.electric_field.reshape(focal_grid.shape))**2
    return focal_intensity

def generate_defocus_lists(example_defocus, scales, f, D, test_ab, telescope_pupil, wavelength, focal_grid, prop_p2f):
    psf_list = []
    distance_list = []

    example_defocus = example_defocus.reshape(telescope_pupil.shape)
    test_ab = test_ab.reshape(telescope_pupil.shape)

    no_defocus_phase = np.zeros_like(example_defocus)
    no_defocus_image = propagate_image(no_defocus_phase, test_ab, telescope_pupil, wavelength, focal_grid, prop_p2f)
    psf_list.append(no_defocus_image)

    for scale in scales:
        p2v_radians, delta = calculate_defocus_params(example_defocus, scale, f, D, wavelength)
        print(f'Scale {scale}: P2V error: {p2v_radians} rad, {p2v_radians/(2*np.pi)} waves, defocus distance: {delta*1e6} microns')
        defocus_image = propagate_image(example_defocus * scale, test_ab, telescope_pupil, wavelength, focal_grid, prop_p2f)
        psf_list.append(defocus_image)
        distance_list.append(delta * 1e6)

    dx_list = [2.0071, 2.0071]
    return psf_list, distance_list, dx_list

def run_focus_diverse_phase_retrieval(scales=[2, 1], test_ab_scale=0.75):
    pupil_size = 10.12e-3
    focal_length = 500e-3
    wavelength = 650e-9
    grid_size = 256
    q = 16
    num_airy = 16

    pupil_grid = make_pupil_grid(grid_size, pupil_size)
    aperture = circular_aperture(pupil_size)
    telescope_pupil = aperture(pupil_grid)

    wavefront = Wavefront(telescope_pupil, wavelength=wavelength)
    focal_grid = make_focal_grid(q=q, num_airy=num_airy, pupil_diameter=pupil_size, focal_length=focal_length, reference_wavelength=wavelength)
    
    prop_p2f = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

    pupil_image = wavefront.copy()

    influence_functions = make_zernike_basis(256, pupil_size, pupil_grid)
    
    test_ab = test_ab_scale * influence_functions[6]

    example_defocus = test_ab
    psf_list, distance_list, dx_list = generate_defocus_lists(example_defocus, scales, focal_length, pupil_size, test_ab, telescope_pupil, wavelength, focal_grid, prop_p2f)

    mp = FocusDiversePhaseRetrieval(psf_list, 650e-3, dx_list, distance_list)
    for i in range(200):
        psf00 = mp.step()

    seal_params = {
        'image_dx': 2.0071,
        'efl': focal_length * 1e3,
        'wavelength': 0.65,
        'pupil_size': pupil_size * 1e3
    }
    conf = InstrumentConfiguration(seal_params)

    raw_pupil_phase = np.angle(mft_rev(psf00, conf))
    
    pupil_phase = resize(raw_pupil_phase, (256, 256)) * telescope_pupil.shaped

    masking_pupil = telescope_pupil.shaped
    med_subtracted = pupil_phase - np.median(pupil_phase[np.array(masking_pupil, dtype=bool)])
    difference_image = pupil_image.phase.shaped - med_subtracted

    check_error_region = (pupil_image.phase.shaped - med_subtracted)[np.array(masking_pupil, dtype=bool)]
    print(f'Median error of {np.median(check_error_region)} radians.')
    nm_med = phase_to_m(np.median(check_error_region), 650e-9) * 1e9
    print(f'Median error in nano: {nm_med} nm')

    valid_phase_values = med_subtracted[telescope_pupil > 0]
    mean_phase = np.mean(valid_phase_values)
    rms_error = np.sqrt(np.mean((valid_phase_values - mean_phase) ** 2))

    print(f"RMS error: {rms_error} radians")

    return {
        "pupil_phase": pupil_phase,
        "difference_image": difference_image,
        "psf_list": psf_list,
        "p2v_error": np.max(pupil_phase) - np.min(pupil_phase),
        "rms_error": rms_error,
        "nm_med": nm_med
    }
