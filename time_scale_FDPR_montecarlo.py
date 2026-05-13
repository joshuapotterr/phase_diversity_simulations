import time
import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from skimage.restoration import unwrap_phase
from skimage.transform import resize
from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
from processing import phase_unwrap_2d
from astropy.io import fits
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

'''
Just make own heatmap and save respetive thigns for this script, separate from the previous script. Than have a plotting script that can plot both of them, have the output 
of both of these scripts put into a thiurd script for plotting. i.e svaing the heatmaps and loading them into another script solely for debugging and plotting--faster
Consider saving every trial of monte carlo'''


'''Helper Functions'''
# Convert defocus distance into phase error
def delta_to_p(delta, f, D):
    return -1 * delta / (8 * (f/D)**2)

def make_sinusoidal_phase_waves(pupil_grid, pupil_diameter_m, cycles_per_aperture, m_waves):
        """
        Single-frequency sinusoid along x. 'cycles/aperture' means cycles across diameter D.
        Returns HCIPy Field [waves] on pupil_grid.
        """
        x = pupil_grid.x  # physical  x coordinate per sample on pupil grid in [m], spans D
        D = pupil_diameter_m # clear aperture diamter
        phase_waves = m_waves * np.sin(2 * np.pi * cycles_per_aperture * (x / D)) # 1D sinusoid along x axis with spatial frequency we provide
            #Units for phase_waves is [waves]
        return Field(phase_waves, pupil_grid) # wraps the array into a HCIpy Field on same pupil grid

def psf_from_wavefront(wf):
        """
        We input wf as a wavefront on the pupil_grid with a set wavelength. We than use 
        prop_p2f to do a physcial scaled FFT propagation. Get the intensity, than make into a 2D array
        with focal grid's smapling. Than normalize to a peak=1 for handyness. 
        """
        I = prop_p2f(wf).intensity.shaped

        #I = I / np.max(I) if np.max(I) > 0 else I
        #global normalization... intensity array normalized so peak =1
        #Removes information about Strehl ratio (absolute image sharpness).
        # Every PSF looks “equally bright” at the core
        return np.asarray(I)

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


'''
Simulation Setup
'''
seal_parameters = {
        'image_dx': 2.0071, # pixel image 
        'efl': 500, # SEAL effective focal length, mm # SEAL center wavelength, microns- >prysm
        'wavelength_meter': 650e-9,#SEAL center wavelength, meters -> hcipy
        'pupil_size': 10.12e-3, # Keck entrance pupil diameter
        'small_pupil_size_meter': 9.5e-3,
        'pupil_pixel_dimension': 256,
        'focal_length_meters': 500e-3,
        'q': 4,
        'Num_airycircles': 64,
        'grid_dim':10
         }
seal_param_config = {'image_dx': 2.0071, # 
               'efl': seal_parameters['focal_length_meters']*1e3, # SEAL effective focal length, mm
               'wavelength': 0.65, # SEAL center wavelength, microns
                'pupil_size': seal_parameters['pupil_size']*1e3, # Keck entrance pupil diameter
                    }

'''
Build Monte Carlo Grid

'''
        #Constants
f_m=seal_parameters['focal_length_meters']
D_m = seal_parameters['pupil_size']
lam_m = seal_parameters['wavelength_meter']
N_airy = seal_parameters['Num_airycircles']
F = f_m/D_m

delta_x = 2 * N_airy* (lam_m/D_m)*f_m
dx=delta_x/focal_grid.shape[0] # pixel size in focal plane

delta_k = (2*np.pi)/dx
dk = (2*np.pi)/delta_x


dzs_mc = np.linspace(0, 250, 20)
v0s_mc = np.linspace(0, 20, 20)

monte_carlo_grid = make_uniform_grid([512,512], delta_k) # otf space?

#rows:defocus, columns:spatial, storing side peak amp per (dz, v0)
MC = np.zeros((len(dzs_mc), len(v0s_mc)))   



noise_implemented = True
if noise_implemented:
    from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
    conf = InstrumentConfiguration(seal_param_config)

    # MC parameters
    N_trials   = 5      # number of noise realizations per (dz, v0)
    sigma_e    = 11     # read noise [e- rms]
    seed       = 12345
    rng        = np.random.default_rng(seed)

    # choose how many dz / v0 points to sample
    '''
    Dont have to pull from the fixed dz, can just use the same min/max  
    '''
    n_dz_samples  = 2
    n_v0_samples  = 2

    Nd, Nv = len(dzs_mc), len(v0s_mc)

    # full-size arrays, NaN where we don't sample
    rms_mean_nm = np.full((Nd, Nv), np.nan)
    rms_std_nm  = np.full((Nd, Nv), np.nan)

    pupil_mask = np.array(telescope_pupil.shaped > 0, dtype=bool)

    def add_read_noise(image_electron, sigma_e, rng):
        """Add Gaussian read noise to a PSF in electrons."""
        read_noise = rng.normal(scale=sigma_e, size=image_electron.shape)
        return image_electron + read_noise

    # evenly spaced dz and v0
    dz_idx = np.linspace(0, Nd - 1, n_dz_samples, dtype=int)
    v0_idx = np.linspace(0, Nv - 1, n_v0_samples, dtype=int)
    #EMiel Notes
    '''
    '''

    dz_subset = dzs_mc[dz_idx]
    v0_subset = v0s_mc[v0_idx]

    print(f"Monte Carlo over {len(dz_subset)} dz values × {len(v0_subset)} v0 values "
          f"= {len(dz_subset)*len(v0_subset)} grid points")

    t_start = time.time()
    n_sample = 0

    wavelength_um = seal_param_config['wavelength']  # for FDPR class
    lam_m        = wavelength_um * 1e-6

    for ii, i_dz in enumerate(dz_idx):
        dz = float(fixed_dz_heatmap[i_dz])
        phi_def = calculate_defocus_phase(seal_parameters, dz)

        for jj, j_v0 in enumerate(v0_idx):
            v0 = float(v0_heatmap[j_v0])

            #include N_trials here ., changes the timing

            #abberated WF for this i,j
            phi_sine_waves = make_sinusoidal_phase_waves(pupil_grid, D_m, v0, m_waves)
            phi_sine_rad   = 2 * np.pi * phi_sine_waves

            wf_focus     = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), lam_m)
            psf_focus_clean = psf_from_wavefront(wf_focus)

            wf_defocused   = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
            psf_defoc_clean = psf_from_wavefront(wf_defocused)

            rms_trials = []


            for t in range(N_trials):
                n_sample += 1
                #read noise
                psf0_noisy = add_read_noise(psf_focus_clean,  sigma_e, rng)
                psf0_noisy[psf0_noisy < 0] = 0 ##SKEWS DATA, FOR LOWER FLUX LEVELS
                psfd_noisy = add_read_noise(psf_defoc_clean, sigma_e, rng)
                psfd_noisy[psfd_noisy < 0] = 0 #SKEWS DATA, FOR LOWER FLUX LEVELS
                #On each PSF anything below zero set to zero, NOTE Jaren's algorithm cant handle noise for PSF
                psf_list_stack = [psf0_noisy, psfd_noisy]
                dx_list = [2.0071] * (len(psf_list_stack) - 1)  # your existing plate-scale logic

                # FDPR
                mp = FocusDiversePhaseRetrieval(psf_list_stack,
                                                wavelength_um,
                                                dx_list,
                                                [dz])  # dz list in mm

                for _ in range(100):  # iterations per trial; tune if needed
                    psf00 = mp.step()
                    print(f'psf00 is {psf00} after {_} iterations')
                
                

                #bacl to pupil for rms
                raw_pupil  = np.angle(mft_rev(psf00, conf))
                real_pupil = resize(raw_pupil, (256, 256)) * telescope_pupil.shaped

                # mask to clear aperture
                masked_phase = real_pupil[pupil_mask]

                # phase RMS in radians
                phase_rms_rad = np.sqrt(np.nanmean(masked_phase**2))

                # convert to nm of WFE
                wfe_rms_nm = phase_rms_rad * lam_m * 1e9 / (2 * np.pi)

                rms_trials.append(wfe_rms_nm)
                print(rms_trials.shape)

            # store stats into full-grid arrays at the true indices
            rms_trials = np.array(rms_trials)
            #save rms_trials itself as well without reducing it to mean/std. will be 3dim array
            rms_mean_nm[i_dz, j_v0] = np.nanmean(rms_trials)
            rms_std_nm[i_dz, j_v0]  = np.nanstd(rms_trials)

            print(f"Sample ({ii+1}/{len(dz_subset)}, {jj+1}/{len(v0_subset)}) "
                  f"dz={dz:.1f} mm, v0={v0:.2f} cyc/ap done.")

    t_end = time.time()
    elapsed = t_end - t_start
    avg_per_sample = elapsed / n_sample


    print(f"\nMonte Carlo timing: {elapsed:.1f} s for {n_sample} grid points.")
    print(f"Average per (dz,v0) sample: {avg_per_sample:.2f} s")

    # runtime estimate 
    runtime=False
    if runtime:
        total_grid   = Nd * Nv
        half_grid    = (Nd//2) * (Nv//2)
        quarter_grid = (Nd//4) * (Nv//4)
        eighth_grid  = (Nd//8) * (Nv//8)

        est_full    = total_grid   * avg_per_sample / 60.0
        est_half    = half_grid    * avg_per_sample / 60.0
        est_quarter = quarter_grid * avg_per_sample / 60.0
        est_eighth  = eighth_grid  * avg_per_sample / 60.0

        print(f"Estimated full grid runtime:    {est_full:.1f} minutes ({total_grid} pts)")
        print(f"Estimated half grid runtime:    {est_half:.1f} minutes ({half_grid} pts)")
        print(f"Estimated quarter grid runtime: {est_quarter:.1f} minutes ({quarter_grid} pts)")
        print(f"Estimated eighth grid runtime:  {est_eighth:.1f} minutes ({eighth_grid} pts)")
    plotting=True
    if plotting:
        load_heatmap=True
        if load_heatmap:
            loaded_heatmap = np.load("OTF_heatmap_data.npz")
            H = loaded_heatmap["H"]
            fixed_dz_heatmap= loaded_heatmap["fixed_dz_heatmap"]
            v0_heatmap = loaded_heatmap["v0_heatmap"]

        extent = [v0_heatmap.min(), v0_heatmap.max(),
                fixed_dz_heatmap.min(), fixed_dz_heatmap.max()]

        # 1) Mean RMS WFE (nm) vs (dz, v0)
        plt.figure(figsize=(8,6))
        clim_mean = np.nanpercentile(rms_mean_nm, [5, 95])
        im1 = plt.imshow(rms_mean_nm, aspect='auto', origin='lower', extent=extent,
                        cmap='magma_r', vmin=clim_mean[0], vmax=clim_mean[1])
        cbar1 = plt.colorbar(im1)
        cbar1.set_label("FDPR phase RMS (nm) — mean over trials")
        plt.xlabel(r"Spatial frequency $v_0$ [cycles/aperture]")
        plt.ylabel(r"Defocus $\Delta z$ [mm]")
        plt.title(fr"FDPR Monte Carlo mean RMS, N={N_trials}, $\sigma_\mathrm{{read}}={sigma_e:.0f}$ e$^-$")
        plt.tight_layout()
        plt.show()

        # 2) Std dev of RMS WFE (nm) vs (dz, v0)
        plt.figure(figsize=(8,6))
        # guard against all-NaN or all-constant
        finite_std = np.isfinite(rms_std_nm)
        if np.any(finite_std):
            clim_std = np.nanpercentile(rms_std_nm[finite_std], [5, 95])
        else:
            clim_std = [0, 1]

        im2 = plt.imshow(rms_std_nm, aspect='auto', origin='lower', extent=extent,
                        cmap='viridis', vmin=clim_std[0], vmax=clim_std[1])
        cbar2 = plt.colorbar(im2)
        cbar2.set_label("FDPR phase RMS (nm) — std over trials")
        plt.xlabel(r"Spatial frequency $v_0$ [cycles/aperture]")
        plt.ylabel(r"Defocus $\Delta z$ [mm]")
        plt.title(fr"FDPR Monte Carlo RMS variability, N={N_trials}")
        plt.tight_layout()
        plt.show()

        # save if we want
        np.savez("FDPR_MonteCarlo_RMS_maps.npz",
                rms_mean_nm=rms_mean_nm,
                rms_std_nm=rms_std_nm,
                fixed_dz_heatmap=fixed_dz_heatmap,
                v0_heatmap=v0_heatmap,
                N_trials=N_trials,
                sigma_e=sigma_e)
