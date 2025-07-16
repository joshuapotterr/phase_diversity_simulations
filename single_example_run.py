
simulation_elements = build_seal_simulation(seal_parameters)

# Choose two physical defocus distances in meters
defocus_1 = -3.0e-2  # -30 mm
defocus_2 = 2.75e-3  # 2.75 mm

# Construct defocus dictionary
defocus_dict = {
    defocus_1: calculate_defocus_phase(seal_parameters, simulation_elements, defocus_1),
    defocus_2: calculate_defocus_phase(seal_parameters, simulation_elements, defocus_2),
}

# Define wavefront error to inject
wf_error_to_retrieve = 0.75 * simulation_elements['zernike_sample_12'][6]  # for example

# Simulate focused PSF (truth)
system_truth_intensity, system_truth_phase, system_truth = simulate_focused_image(
    wf_error_to_retrieve, simulation_elements, seal_parameters['wavelength_meter']
)

# Run retrieval
psf_estimate, cost_functions = focus_diverse_phase_retrieval(
    system_truth_intensity, defocus_dict, wf_error_to_retrieve, seal_parameters, simulation_elements
)

# Evaluate result
metrics = calculate_phase_retrieval_accuracy(
    system_truth_phase, psf_estimate, cost_functions, seal_parameters, simulation_elements,
    phase_unwrap_method="phase_unwrap_2d", verbose=True
)
