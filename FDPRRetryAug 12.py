#FDPR Retry Aug 12
import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from skimage.restoration import unwrap_phase
from skimage.transform import resize
from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
from processing import phase_unwrap_2d
from astropy.io import fits
'''
Helper Functions
'''
# Convert phase into meters using the wavelength
def phase_to_m(phase, wv):
    return phase * wv / (2 * np.pi)

# Calculate defocus distance from peak-to-valley (P2V) error in meters
def p_to_delta(P, f, D):
    return 8 * P * (f/D)**2

# Convert defocus distance into phase error
def delta_to_p(delta, f, D):
    return -1 * delta / (8 * (f/D)**2)
'''
Set some hardcoded variables. For reference, these will be in MM
These will be used to create the simulation elements
'''
seal_parameters = {
        'image_dx': 2.0071, # pixel image 
        'efl': 500, # SEAL effective focal length, mm # SEAL center wavelength, microns- >prysm
        'wavelength_meter': 650e-9,#SEAL center wavelength, meters -> hcipy
        'pupil_size': 10.12e-3, # Keck entrance pupil diameter
        'small_pupil_size_meter': 9.5e-3,
        'pupil_pixel_dimension': 256,
        'focal_length_meters': 500e-3,
        'q': 16,
        'Num_airycircles': 16,
        'grid_dim':10
         }
'''
This dictionary is specifically for Instrument Configuration Class
'''
seal_param_config = {'image_dx': 2.0071, # 
               'efl': seal_parameters['focal_length_meters']*1e3, # SEAL effective focal length, mm
               'wavelength': 0.65, # SEAL center wavelength, microns
                'pupil_size': seal_parameters['pupil_size']*1e3, # Keck entrance pupil diameter
                    }
conf = InstrumentConfiguration(seal_param_config)
'''
Now We build the simulation elements
'''
conf = InstrumentConfiguration(seal_param_config)
pupil_grid = make_pupil_grid(256, seal_parameters['pupil_size'])
focal_grid = make_focal_grid(q=seal_parameters['q'],
                             num_airy=seal_parameters['Num_airycircles'],
                             pupil_diameter=seal_parameters['pupil_size'],
                             focal_length=seal_parameters['focal_length_meters'],
                             reference_wavelength=seal_parameters['wavelength_meter'])
aperture = make_circular_aperture(seal_parameters['pupil_size'])
telescope_pupil = aperture(pupil_grid)
small_aperture = make_circular_aperture(seal_parameters['small_pupil_size_meter'])
masking_pupil = small_aperture(pupil_grid)
zernike_modes = make_zernike_basis(
         num_modes=256,
         D=seal_parameters['pupil_size'],
         grid=pupil_grid
    )
#Repeating Testing from ipynb, confirm for aperture. CONFIRMED
#imshow_field(telescope_pupil, cmap='gray')

'''
Now for propagation that goes focal -> pupil
'''
wavefront = Wavefront(telescope_pupil,
                      seal_parameters['wavelength_meter']
                      )
prop_p2f = FraunhoferPropagator(pupil_grid,
                                focal_grid,
                                seal_parameters['focal_length_meters']
                                )
#Testing from ipynb, confirm for pupil plane. CONFIRMED
#imshow_field(wavefront.intensity)
pupil_image = wavefront.copy()
focal_image = prop_p2f.forward(wavefront)
#Testing from ipynb, confirm for Focal plane. CONFIRMED
#imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5)
perfect_focal = focal_image.copy()
defocus_template = zernike_modes[3]
error_to_retrieve = -.75 * zernike_modes[6]
#Testing from ipynb, confirm for Defocus/coma. CONFIRMED
#imshow_field(error_to_retrieve)
#plt.colorbar()

'''
Now I will Simulate a Focused Image, using Wavefront object
We will use the .intensity and .phase attributes of the wavefront
'''
focused_wavefront_pupil = Wavefront(telescope_pupil * np.exp(1j * error_to_retrieve.ravel()), #or flatten
                   seal_parameters['wavelength_meter'])
focused_wavefront_focal = prop_p2f(focused_wavefront_pupil)
#Testing to see System truth, confirm. CONFIRMED
#plt.imshow(focused_wavefront_focal.intensity.shaped)

#THIS SHOULD BE IN MM
def calculate_defocus_phase(seal_parameters,
                            defocus_distance):
    mask = np.array(telescope_pupil.shaped, dtype=bool)
    defocus_template_s=defocus_template.shaped
    template_p2v=defocus_template_s[mask].max() - defocus_template_s[mask].min()
    unit_defocus = defocus_template / template_p2v
    delta_m = defocus_distance * 1e-3
    defocus_p2v = delta_to_p(
                            delta = delta_m,
                            f = 500e-3,
                            D=10.12e-3
                            )
    phase_p2v = defocus_p2v * (2*np.pi/seal_parameters['wavelength_meter'])
    defocus_phase = unit_defocus * phase_p2v
    print('defocus_p2v value:',defocus_p2v)
    print('phase_p2v value', phase_p2v)
    return defocus_phase



#Rebuild lists to satisfy FDPR: psf_list = N+1, dxs = N, defocus_positions = N ---

# defocus list (mm)
defocus_mm_list = np.array([0, -12.0, -8.0, -4.0, 4.0, 8.0, 12.0])  # mm

# Calculate the PSF for each defocus.
psf_list = []      # length N

'''
This is where I construct the defocused PSFs 
'''
for dz_mm in defocus_mm_list:
    defocus_phase = calculate_defocus_phase(seal_parameters, dz_mm)  # radians on pupil grid

    wf_pupil_defocused = Wavefront(
        telescope_pupil * np.exp(1j * (error_to_retrieve.ravel() + defocus_phase.ravel())),
        seal_parameters['wavelength_meter']
    )
    wf_focal_defocused = prop_p2f(wf_pupil_defocused)
    psf = np.asarray(wf_focal_defocused.intensity.shaped)

    psf_list.append(psf)

    plt.imshow(psf)
    plt.title(f'Defocus: {dz_mm} mm')
    plt.show()
    
dx_list = [2.0071] * (len(psf_list) - 1)

print(f"Final lengths → psf_list: {len(psf_list)} (should be N+1), "
      f"dx_list: {len(dx_list)} (N), defocus_positions: {len(defocus_mm_list)} (N)")
print("First two PSF shapes:", psf_list[0].shape, psf_list[1].shape)

# 5) Run FDPR with consistent units
try:
    #FDPR expects wavelength in um and defocus in um:
    wavelength_um = seal_param_config['wavelength']  # 0.65
    mp = FocusDiversePhaseRetrieval(psf_list, wavelength_um, dx_list, defocus_mm_list[1:] * 1e3)#excludes the first item in list(0)

    for _ in range(200):
        psf00=mp.step()
    
    plt.imshow(np.angle(psf00))
    plt.title('psf00 after 200 iterations')
    plt.colorbar()
    plt.show()

    raw_pupil_phase = np.angle(mft_rev(psf00, conf))
    plt.imshow(raw_pupil_phase)
    plt.title('Raw Pupil Phase')
    plt.colorbar()
    plt.show()

    pupil_phase = resize(raw_pupil_phase, (256, 256))*telescope_pupil.shaped
    plt.imshow(pupil_phase)
    plt.title('Pupil Phase Reconstruction')
    plt.colorbar()
    plt.show()   
    print(f'P2V error of Reconstruction: {np.max(pupil_phase) - np.min(pupil_phase)}')
    masked_phase = pupil_phase[telescope_pupil.shaped == 1]
    print(f'RMS of masked Reconstruction: {np.sqrt(np.mean(masked_phase**2))}')

    # RMS and P2V in radians
    rms_rad = np.sqrt(np.mean(masked_phase**2))
    p2v_rad = np.max(masked_phase) - np.min(masked_phase)

    # Convert to nanometers
    rad_to_nm = seal_parameters['wavelength_meter'] / (2 * np.pi) * 1e9
    rms_nm = rms_rad * rad_to_nm
    p2v_nm = p2v_rad * rad_to_nm

    print(f'RMS error: {rms_nm:.2f} nm')
    print(f'P2V error: {p2v_nm:.2f} nm')

    '''
    Now for Some Error Metrics
    '''

    ##NEEDS WORK
    med_subtracted = pupil_phase - np.median(pupil_phase[np.array(masking_pupil.shaped, dtype=bool)])
    #difference_image = pupil_image.phase.shaped - med_subtracted
    #this would be the system truth
    truth_resize256 = resize(focused_wavefront_pupil.phase.shaped, (256, 256), preserve_range=True)
    plt.imshow(truth_resize256 - med_subtracted)
    plt.title('difference image')
    plt.colorbar()
    plt.axis('off')
    plt.show()

    check_error_region = (pupil_image.phase.shaped - med_subtracted)[np.array(telescope_pupil.shaped, dtype=bool)]
    print(f'Median error of {np.median(check_error_region)} radians.')
    # Compute residual
    residual = focused_wavefront_pupil.phase.shaped - med_subtracted
    masked_residual = residual[telescope_pupil.shaped == 1]
    median_error = np.median(masked_residual)
    rms_error_rad = np.sqrt(np.mean(masked_residual**2))

    rad_to_nm = seal_parameters['wavelength_meter'] / (2 * np.pi) * 1e9
    rms_error_nm = rms_error_rad * rad_to_nm
    median_error_nm = median_error * rad_to_nm

    # Print results
    print(f"Median error: {median_error:.4f} rad ({median_error_nm:.2f} nm)")
    print(f"RMS error: {rms_error_rad:.4f} rad ({rms_error_nm:.2f} nm)")
    #Graphing the Cost Functions, make it iterative, for i in range len[mp.cost_func]
    for i in range(len(mp.cost_functions)):
        plt.semilogy(mp.cost_functions[i], label=f'defocus {i+1}')
        
    plt.legend()
    plt.show()

    '''
    Grid Logic
    '''    

    '''
    1d Grid Logic
    '''  
    oned_grid= True  
    if oned_grid:
        truth_pupil = (telescope_pupil.shaped)
        mask = np.array(telescope_pupil.shaped, dtype=bool)
        dz_1d = 5e-3 #5mm defocus
        defocus_phase = calculate_defocus_phase(seal_parameters, dz_1d)
        wf_focused = Wavefront(
            telescope_pupil * np.exp(1j*0), #or flatten
            seal_parameters['wavelength_meter'])
        
        wf_aberrated = Wavefront(
            telescope_pupil * np.exp(1j * (defocus_phase.ravel())),
            seal_parameters['wavelength_meter'])
        psf_focused = prop_p2f(wf_focused).intensity.shaped
        psf_defocused = prop_p2f(wf_aberrated).intensity.shaped
        psf_list_1d = [psf_focused, psf_defocused]
        dx_list_1d = [seal_parameters['image_dx']]
        defocus_list_1d = [dz_1d]
        mp_1d = FocusDiversePhaseRetrieval(
            psf_list_1d,
            seal_param_config['wavelength'],
            dx_list_1d,
            defocus_list_1d
        )
        n_iter_1d= 200
        for _ in range(n_iter_1d):
            retrieval_1d = mp_1d.step()

        raw_pupil_phase = np.angle(mft_rev(retrieval_1d, conf))

        if raw_pupil_phase.ndim == 1:
            raw_pupil_phase = raw_pupil_phase.reshape(telescope_pupil.shaped.shape)

                # Resize to pupil grid if needed and mask
        pupil_phase_rec = resize(raw_pupil_phase, (256, 256), preserve_range=True) * telescope_pupil.shaped
        rec_masked = pupil_phase_rec[mask].copy()
        rec_masked -= np.median(rec_masked)

        truth_masked = truth_pupil[mask]

        rms_rad = np.sqrt(np.mean((rec_masked - truth_masked)**2))
        p2v_error = np.max(pupil_phase_rec) - np.min(pupil_phase_rec)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(np.log10(psf_focused / psf_focused.max()), vmin=-5)
        axs[0].set_title("Focused PSF")

        axs[1].imshow(np.log10(psf_defocused / psf_defocused.max()), vmin=-5)
        axs[1].set_title(f"Defocused PSF (Δz={dz_1d*1e3:.2f} mm)")

        axs[2].imshow(pupil_phase_rec, cmap='RdBu')
        axs[2].set_title("Retrieved Phase")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        None
        
    twod_gridsweep = False
    if twod_gridsweep: 
        # Sweep over defocus in mm
        dim = seal_parameters['grid_dim']  # 10
        defocus_mm_vals = np.linspace(-10.0, 10.0, dim)  # mm

        # Heatmap to store RMS (rad) fill upper triangle (i<j leave others as NaN
        rms_heatmap_nm = np.full((dim, dim), np.nan)

        n_iter_grid = 80

        truth_pupil = (error_to_retrieve.shaped * telescope_pupil.shaped)
        mask = np.array(telescope_pupil.shaped, dtype=bool)
        rad2nm = seal_parameters['wavelength_meter'] / (2*np.pi) * 1e9
        data = np.random.rand(10,10, 256, 256)
        # Loop upper-triangular pairs (i<j)
        for i in range(dim):
            for j in range(i+1, dim):
                dz1_mm = float(defocus_mm_vals[i])
                dz2_mm = float(defocus_mm_vals[j])

                # Build PSFs: focused (0 mm) + two defocus states
                psf_list_pair = []

                # Focused (0 mm)
                phi0 = calculate_defocus_phase(seal_parameters, 0.0)
                wf0 = Wavefront(telescope_pupil * np.exp(1j * (error_to_retrieve + phi0)),
                                seal_parameters['wavelength_meter'])
                psf0 = np.asarray(prop_p2f(wf0).intensity.shaped)
                psf_list_pair.append(psf0)

                # Defocus 1
                phi1 = calculate_defocus_phase(seal_parameters, dz1_mm)
                wf1 = Wavefront(telescope_pupil * np.exp(1j * (error_to_retrieve + phi1)),
                                seal_parameters['wavelength_meter'])
                psf1 = np.asarray(prop_p2f(wf1).intensity.shaped)
                psf_list_pair.append(psf1)

                # Defocus 2
                phi2 = calculate_defocus_phase(seal_parameters, dz2_mm)
                wf2 = Wavefront(telescope_pupil * np.exp(1j * (error_to_retrieve + phi2)),
                                seal_parameters['wavelength_meter'])
                psf2 = np.asarray(prop_p2f(wf2).intensity.shaped)
                psf_list_pair.append(psf2)

                # dx list: one per defocused image (exclude the first focused)
                dx_list_pair = [seal_param_config['image_dx'], seal_param_config['image_dx']]

                # defocus_positions for FDPR in µm (exclude focused)
                defocus_positions_um = np.array([dz1_mm, dz2_mm]) * 1e3

                # Run FDPR on this pair
                mp_pair = FocusDiversePhaseRetrieval(psf_list_pair,
                                                    seal_param_config['wavelength'],  # µm
                                                    dx_list_pair,
                                                    defocus_positions_um)             # µm

                psf_rec = None
                for _ in range(n_iter_grid):
                    psf_rec = mp_pair.step()

                # Reverse transform to pupil phase 
                raw_pupil_phase = np.angle(mft_rev(psf_rec, conf))
                if raw_pupil_phase.ndim == 1:
                    raw_pupil_phase = raw_pupil_phase.reshape(telescope_pupil.shaped.shape)

                # Resize to pupil grid if needed and mask
                pupil_phase_rec = resize(raw_pupil_phase, (256, 256), preserve_range=True) * telescope_pupil.shaped
                

                #more error metric
                rec_masked = pupil_phase_rec[mask].copy()
                rec_masked -= np.median(rec_masked)

                truth_masked = truth_pupil[mask]

                rms_rad = np.sqrt(np.mean((rec_masked - truth_masked)**2))
                rms_heatmap_nm[i, j] = rms_rad * rad2nm
                data[i,j] = pupil_phase_rec


        hdu = fits.PrimaryHDU(data)
        hdul =fits.HDUList([hdu])
        hdul.writeto("example.fits", overwrite=True)


        # Plot heatmap (upper triangle filled)
        plt.figure(figsize=(6,5))
        extent = [defocus_mm_vals[0], defocus_mm_vals[-1], defocus_mm_vals[0], defocus_mm_vals[-1]]  # mm axes
        im = plt.imshow(rms_heatmap_nm, origin='lower', extent=extent, cmap='inferno')
        plt.colorbar(im, label='RMS error [nm]')
        plt.title('Phase Retrieval RMS Error Heatmap with no Phase error (defocus pairs)')
        plt.xlabel('Defocus 1 [mm]')
        plt.ylabel('Defocus 2 [mm]')
        plt.tight_layout()
        plt.savefig('heatmap_rms_nm.png', dpi=200)
        plt.show()
        '''
        End of the Grid
        '''
    else:
        None

except Exception as e:
    print("[FDPR] run failed:", repr(e))

