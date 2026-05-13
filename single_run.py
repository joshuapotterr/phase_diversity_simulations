
simulation_elements = build_seal_simulation(seal_parameters)


# Choose two physical defocus distances in meters
defocus_1 = 13.994  # in mm
defocus_2 = 3.498  
defocus_3 = -6.997

# Construct defocus dictionary
defocus_dictionary = {
    defocus_1: calculate_defocus_phase(seal_parameters, simulation_elements, defocus_1),
    defocus_2: calculate_defocus_phase(seal_parameters, simulation_elements, defocus_2),
    defocus_3: calculate_defocus_phase(seal_parameters, simulation_elements, defocus_3),
}

# Define wavefront error to inject
wf_error_to_retrieve = 0.75 * simulation_elements['zernike_sample_256'][6]  # for example

# Simulate focused PSF (truth)
system_truth_intensity, system_truth_phase, system_truth = simulate_focused_image(
    wf_error_to_retrieve, simulation_elements, seal_parameters['wavelength_meter']
)

system_truth_intensity=resize(system_truth_intensity, (256,256))

                             
# Run retrieval
psf_estimate, cost_functions = focus_diverse_phase_retrieval(
    system_truth_intensity, defocus_dictionary, wf_error_to_retrieve, seal_parameters, simulation_elements
)

# Evaluate result
phase_estimate, metrics = calculate_phase_retrieval_accuracy(
    system_truth_phase, psf_estimate, cost_functions, seal_parameters, simulation_elements,
    phase_unwrap_method="phase_unwrap_2d", verbose=True
)
def plot_phase_diversity_summary(system_truth_intensity,
                                 simulation_elements,
                                 defocus_dictionary,
                                 psf_estimate,
                                 wf_error_to_retrieve,
                                 phase_estimate,
                                 metrics,
                                 cost_functions):
    #0 
    plt.figure()
    imshow_field(simulation_elements['original_wavefront'].intensity)
    plt.title('perfect pupil plane intensity')
    plt.show()

    #focal image as defined in original
    plt.figure()
    imshow_field(np.log10(simulation_elements['original_focal_image'].intensity
                           / simulation_elements['original_focal_image'].intensity.max()), vmin=-5)
    plt.title('perfect focal image intensity')
    #Focused PSF, truth
    plt.figure()
    plt.imshow(system_truth_intensity)
    plt.title('Focused PSF Intensity (Truth)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    #Pupil Plane of Focused WF
    plt.figure()
    plt.imshow(system_truth_phase.shaped)
    plt.title('Pupil Plane Focused WF Phase')
    plt.show()

    #just the zernike defocus from dictionary
    zernike_field = simulation_elements['zernike_sample_256'][3]
    print("Field shape:", zernike_field.shape)
    print("Grid shape:", zernike_field.grid.shape)

    plt.imshow(zernike_field)
    plt.title('Zernike Sample 3')
    plt.colorbar()
    plt.show()
    # 2. Defocused PSFs in the dict
    for z, defocused_psf in defocus_dictionary.items():
        print("shape:",   defocused_psf.shape)
        print("dtype:",   defocused_psf.dtype)
        print("min,max:", defocused_psf.min(), defocused_psf.max())
        print("sum:",     defocused_psf.sum())
        print("nonzero:", np.count_nonzero(defocused_psf), "/", defocused_psf.size)
    # 3. Estimated PSF
    plt.figure()
    plt.imshow(np.abs(psf_estimate),)
    plt.title('Estimated PSF (Final)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.imshow(np.angle(psf_estimate))##this is psf00 in original
    plt.colorbar(label='Phase [rad]')
    plt.title('output intensity')
    plt.show()
    # 4. Injected Phase (Truth)
    plt.figure()
    plt.imshow(wf_error_to_retrieve)
    plt.title('Injected wf_error_to_retrieve')
    plt.colorbar(label='Phase [rad]')
    plt.tight_layout()
    plt.show()
    # 5. Retrieved Phase
    plt.figure()
    plt.imshow(phase_estimate)
    plt.title('Retrieved Phase estimate')
    plt.colorbar(label='Phase [rad]')
    plt.tight_layout()
    plt.show()
    # 6. Residual Phase
    plt.figure()
    plt.imshow(wf_error_to_retrieve - phase_estimate)
    plt.title('Residual Phase (Truth - Estimate)')
    plt.colorbar(label='Phase Error [rad]')
    plt.tight_layout()
    plt.show()
    # 7. Phase Difference from Metrics
    if 'difference_true_vs_estimate' in metrics:
        plt.figure()
        plt.imshow(metrics['difference_true_vs_estimate'])
        plt.title('Metrics: Phase Difference')
        plt.colorbar(label='Difference [rad]')
        plt.tight_layout()
        plt.show()
    # 8. Cost Function Convergence
    if cost_functions and isinstance(cost_functions[0], (list, np.ndarray)):
        plt.figure()
        plt.semilogy(cost_functions[0])
        plt.title('Cost Function Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
plot_phase_diversity_summary(system_truth_intensity,
                             simulation_elements,
                             defocus_dictionary,
                             psf_estimate,
                             wf_error_to_retrieve,
                             phase_estimate,
                             metrics,
                             cost_functions)
