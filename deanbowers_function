#equation [9] of dean/bowers paper, dimensionless
def dean_bowers_max_list(fringes=11,
                 max_n=1, 
                 wavelength=simulation_elements['wavelength_meter']):
    
    return [fringes**2 / (4*((2*n)-1)) for n in range(max_n +1)]

#inverse of normalized defocus equation, output in m
def a_hat_to_defocus(a_hat,f,D,wavelength):
    return (4 * f**2 * wavelength) / (np.pi * D**2) * a_hat

def dean_bowers_min(fringes=11,
                    n=1,
                    wavelength=simulation_elements['wavelength_meter']):

    return (fringes**2)/(8*n)
a_hats = dean_bowers_max_list(fringes=11, max_n=2, wavelength=wavelength)
deltas = [a_hat_to_defocus(a, f, D, wavelength) for a in a_hats]

defocus_distances = [-deltas[0], +deltas[0]]
phase_diverse_inputs = [(0, 0, *defocus_distances)]

##all in one
def dean_bowers_defocus_deltas(fringes, max_n, f, D, wavelength):
    v_hat = fringes  # unitless normalized spatial frequency (cycles/aperture)
    a_hats = [(v_hat**2) / (4*(2*n + 1)) for n in range(max_n+1)]
    deltas = [(4 * f**2 * wavelength / (np.pi * D**2)) * a_hat for a_hat in a_hats]
    return deltas
