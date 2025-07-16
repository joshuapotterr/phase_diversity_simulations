#equation [9] of dean/bowers paper, dimensionless
def dean_bowers_max_list(fringes,
                 max_n, 
                 wavelength):
    
    return [fringes**2 / (4*((2*n)-1)) for n in range(max_n +1)]

#inverse of normalized defocus equation, output in m
def a_hat_to_defocus(a_hat,f,D,wavelength):
    return (4 * f**2 * wavelength) / (np.pi * D**2) * a_hat

def dean_bowers_min(fringes,
                    n,
                    wavelength):

    return (fringes**2)/(8*n)
deltas = [a_hat_to_defocus(a, f, D, wavelength) for a in a_hats]

defocus_distances = [-deltas[0], +deltas[0]]
phase_diverse_inputs = [(0, 0, *defocus_distances)]

##all in one
def dean_bowers_defocus_deltas(fringes, max_n, f, D, wavelength):
    v_hat = fringes  # unitless normalized spatial frequency (cycles/aperture)
    a_hats = [(v_hat**2) / (4*(2*n + 1)) for n in range(max_n+1)]
    deltas = [(4 * f**2 * wavelength / (np.pi * D**2)) * a_hat for a_hat in a_hats]
    return deltas

##SINGLE EXXAMPLE RUN
fringes = 21
max_n = 1
f = seal_parameters['focal_length_meters']
D = seal_parameters['pupil_size']
wavelength = seal_parameters['wavelength_meter']
a_hats = dean_bowers_max_list(fringes, max_n, wavelength)
a_hat= a_hat_list[0]
defocus_distance = a_hat_to_defocus(a_hat,f,D, wavelength)
defocus_phase = calculate_defocus_phase(seal_parameters, simulation_elements, defocus_distance)
phase_diverse_information = {
  defocus_distance:defocus_phase
}
psf_estimate, cost_functions = focus_diverse_phase_retrieval(
    system_truth=system_truth_intensity,
    phase_diverse_information=phase_diverse_information,
    wf_error_to_retrieve=wf_error_to_retrieve,
    seal_parameters=seal_parameters,
    simulation_elements=simulation_elements
)
metrics = calculate_phase_retrieval_accuracy(
    system_truth_phase,
    psf_estimate,
    cost_functions,
    seal_parameters,
    simulation_elements,
    phase_unwrap_method='phase_unwrap_2d'
)
plt.imshow(metrics['difference_true_vs_estimate'])
plt.colorbar()
plt.title('Full Phase Difference')



