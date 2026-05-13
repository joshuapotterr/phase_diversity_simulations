#FOURIER and ZERNIKE TEMPLATES##

def generate_zernike_templates(seal_parameters, 
                             simulation_elements, 
                             physical_defocus_range,
                            zernike_mode_indices = [3], # list of zernike mode indices, can choose whatever number
                            plot_modes = True,
                            p2v_target=1.0): #target peak to valley in radians
    """
    Generate Zernike or Fourier phase templates scaled to match a physical defocus range.

    Parameters:
        seal_parameters (dict): Optical setup parameters.
        simulation_elements (dict): Includes pupil_grid, etc.
        physical_defocus_range (array): List of defocus distances in meters.
        mode_type (str): 'zernike' or 'fourier'.*only for fourier
        num_modes (int): Number of modes to use.
        plot_modes ( boolean) : whether to plot the desired fourier or zernikes or not
        p2v_target (float): Peak-to-valley phase shift in radians.

    Returns:
        list: phase_diverse_inputs [((i, j), {defocus_distance: defocus_phase})]
    """
    #dictionary calls
    pupil_grid = simulation_elements['pupil_grid']
    pupil_size = seal_parameters['pupil_size']
    pupil_dim = seal_parameters['pupil_pixel_dimension']
    focal_length_meters = seal_parameters['focal_length_meters']

    #stores tuples: ((i, mode_idx), {defocus_distance: scaled_phase})
    phase_diverse_inputs = []

    #ZERNIKE#
        #  multiple Zernike indices
        #gonna generate full zernike modes over pupil grid
    zernike_modes = make_zernike_basis(pupil_dim,
                                        pupil_size, 
                                        pupil_grid)
        #defocus_template = zernike_modes[zernike_mode_index].shaped
        #original peak to valley for scaling
        # Plotting
        #used subplots here to splot over selcted zernike modes
    if plot_modes:
            n_modes = len(zernike_mode_indices) #zernike modes requested
            n_cols = min(n_modes, 4) # 4 per row
            n_rows = int(np.ceil(n_modes / n_cols)) # number of rows
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for idx, mode_idx in enumerate(zernike_mode_indices):
                mode = zernike_modes[mode_idx].shaped
                im = axes[idx].imshow(mode)
                axes[idx].set_title(f'Zernike Mode {mode_idx}')
                axes[idx].axis('off')
                fig.colorbar(im, ax=axes[idx])
                plt.tight_layout()
                plt.show()

        #for each defocus value, svale to corresponding p2v
        # Create scaled Zernike templates for each defocus value
    for mode_idx in zernike_mode_indices:
            template = zernike_modes[mode_idx].shaped
            template_p2v = np.max(template) - np.min(template)

    for i, defocus_distance in enumerate(physical_defocus_range):
                defocus_p2v = delta_to_p(defocus_distance, focal_length_meters, pupil_size)
                defocus_phase = (template * defocus_p2v) / template_p2v
                phase_diverse_inputs.append(((i, mode_idx), {defocus_distance: defocus_phase}))


    return phase_diverse_inputs

def generate_fourier_templates(seal_parameters,
                                     simulation_elements,
                                     num_modes=3,
                                     p2v_target=1.0,
                                     plot_modes=True):
        
        pupil_grid = simulation_elements['pupil_grid']
        pupil_size = seal_parameters['pupil_size']
        # Generate custom grid of frequency pairs (kx, ky), ive been looking how to not use pairs but I dont understand how
        #not physical frequencies, just indec values
        #x axis only for now for easiness
        kx_ky_pairs = [(n, 0) for n in range(1, num_modes + 1)]
        # Convert to spatial frequencies in rad/m
        kx = np.array([kx * 2 * np.pi / pupil_size for kx, _ in kx_ky_pairs])
        ky = np.array([ky * 2 * np.pi / pupil_size for _, ky in kx_ky_pairs])
        # Create UnstructuredCoords using provided freq
        fourier_coords = UnstructuredCoords((kx, ky))  # Shape: (N,)
        #create cartesian grid utilizing above coords
        fourier_grid = CartesianGrid(fourier_coords)   # Interpreted as spatial frequency grid
        fourier_basis = make_fourier_basis(pupil_grid, fourier_grid)
        #shaped phase maps from the modes
        fourier_modes = (mode.shaped for mode in fourier_basis)

        phase_diverse_inputs = []

        #plotting unscaled fourier for debugging
        #same syntax as zernike
        if plot_modes:
            n_modes = len(fourier_basis)
            n_cols = min(n_modes, 4)
            n_rows = int(np.ceil(n_modes / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))



            for idx, mode in enumerate(fourier_basis):
                im = axes[idx].imshow(mode.shaped)
                axes[idx].set_title(f'Fourier Mode {idx + 1}')
                axes[idx].axis('off')
                fig.colorbar(im, ax=axes[idx])

            plt.tight_layout()
            plt.show()
        for i, mode in enumerate(fourier_modes):
            template_p2v = np.max(mode) - np.min(mode)
            scaled_mode = (mode * p2v_target) / template_p2v
            phase_diverse_inputs.append(((i, i), {p2v_target: scaled_mode}))


        return phase_diverse_inputs

def generate_zernike_defocus_grid(
    seal_parameters,
    simulation_elements,
    physical_defocus_range,
    zernike_mode_index=3,
    plot_modes=True
):
    """
    Generate a 2D grid of defocus phase templates using Zernike mode.

    Each (i,j) entry corresponds to a unique defocus value from the input grid.

    Returns:
        list: [((i, j), {defocus_distance: defocus_phase})]
    """
    pupil_grid = simulation_elements['pupil_grid']
    pupil_size = seal_parameters['pupil_size']
    focal_length = seal_parameters['focal_length_meters']

    # Get Zernike mode 3 (defocus)
    zernike_basis = make_zernike_basis(4, D=pupil_size, grid=pupil_grid)
    defocus_template = zernike_basis[zernike_mode_index].shaped
    template_p2v = np.max(defocus_template) - np.min(defocus_template)

    if plot_modes:
        plt.imshow(defocus_template, cmap='RdBu')
        plt.colorbar()
        plt.title(f'Zernike Mode {zernike_mode_index} (Defocus)')
        plt.show()

    phase_diverse_inputs = []

    # Build 2D defocus grid
    N = len(physical_defocus_range)
    for i, d1 in enumerate(physical_defocus_range):
        for j, d2 in enumerate(physical_defocus_range):
            # You could use d1 for x-defocus, d2 for y-defocus, or combine them
            defocus_distance = (d1 + d2) / 2.0  # simplification
            defocus_p2v = delta_to_p(defocus_distance, focal_length, pupil_size)
            defocus_phase = (defocus_template * defocus_p2v) / template_p2v
            phase_diverse_inputs.append(((i, j), {defocus_distance: defocus_phase}))

    return phase_diverse_inputs
def generate_zernike_templates(seal_parameters, 
                               simulation_elements, 
                               physical_defocus_range,
                               zernike_mode_indices=[3],  # list of zernike mode indices
                               plot_modes=True,
                               p2v_target=1.0):
    """
    Generate Zernike phase templates scaled to match a physical defocus range.

    Parameters:
        seal_parameters (dict): Optical setup parameters.
        simulation_elements (dict): Includes pupil_grid, etc.
        physical_defocus_range (array): List of defocus distances in meters.
        zernike_mode_indices (list): Indices of Zernike modes to use.
        plot_modes (bool): Whether to plot the Zernike templates.
        p2v_target (float): Target peak-to-valley phase shift in radians.

    Returns:
        list: phase_diverse_inputs [((i, mode_idx), {defocus_distance: defocus_phase})]
    """
    # Unpack from simulation and config
    pupil_grid = simulation_elements['pupil_grid']
    pupil_size = seal_parameters['pupil_size']
    pupil_dim = seal_parameters['pupil_pixel_dimension']
    focal_length = seal_parameters['focal_length_meters']

    # Generate Zernike basis
    zernike_modes = make_zernike_basis(pupil_dim, pupil_size, pupil_grid)

    # Plot if requested
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

    # Generate templates
    phase_diverse_inputs = []
    for mode_idx in zernike_mode_indices:
        template = zernike_modes[mode_idx].shaped
        template_p2v = np.max(template) - np.min(template)

        for i, defocus_distance in enumerate(physical_defocus_range):
            defocus_p2v = delta_to_p(defocus_distance, focal_length, pupil_size)
            defocus_phase = (template * defocus_p2v) / template_p2v
            phase_diverse_inputs.append(((i, mode_idx), {defocus_distance: defocus_phase}))
    print("Sample input format:", phase_diverse_inputs[:3])

    return phase_diverse_inputs