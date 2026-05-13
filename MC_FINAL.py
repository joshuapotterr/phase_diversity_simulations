#MC SCRIPT NO OTF
import time
import numpy as np
from skimage.transform import resize
from hcipy import *
from image_sharpening import FocusDiversePhaseRetrieval, mft_rev, InstrumentConfiguration
import matplotlib.pyplot as plt


'''
CONFIGURATION FLAGS
'''
verbose = False          # If True, plot everything step by step
spot_check = False        # If True, do detailed plot at specific (dz, v0)
spot_check_dz = 100.0     # mm - defocus for spot check
spot_check_v0 = 3.0     # cycles/ap - spatial frequency for spot check

'''
MONTE CARLO PARAMETERS
'''
dzs_mc = np.linspace(5, 250, 25)      # mm
v0s_mc = np.linspace(0.5, 30, 40)     # cycles/aperture

N_trials = 5
sigma_e = 11
seed = 12345
rng = np.random.default_rng(seed)
save_label = "MC_4x4.npz"

'''
HELPER FUNCTIONS
'''
def mm_to_m(x_mm: float) -> float:
    return x_mm * 1e-3

def mm_to_um(x_mm: float) -> float:
    return x_mm * 1e3

def delta_to_p(delta, f, D):
    return -1 * delta / (8 * (f/D)**2)

def make_sinusoidal_phase_waves(pupil_grid, pupil_diameter_m, cycles_per_aperture, m_waves):
    """
    Single-frequency sinusoid along x. 'cycles/aperture' means cycles across diameter D.
    Returns HCIPy Field [waves] on pupil_grid.
    """
    x = pupil_grid.x
    D = pupil_diameter_m
    phase_waves = m_waves * np.sin(2 * np.pi * cycles_per_aperture * (x / D))
    return Field(phase_waves, pupil_grid)

def psf_from_wavefront(wf, propagator):
    """
    Propagate wavefront to focal plane and return PSF intensity.
    """
    I = propagator(wf).intensity.shaped
    return np.asarray(I)

def calculate_defocus_phase(seal_parameters, defocus_distance, telescope_pupil, defocus_template):
    """
    Calculate defocus phase from mechanical defocus distance.
    defocus_distance should be in MM.
    """
    mask = np.array(telescope_pupil.shaped, dtype=bool)
    defocus_template_s = defocus_template.shaped
    template_p2v = defocus_template_s[mask].max() - defocus_template_s[mask].min()
    unit_defocus = defocus_template / template_p2v
    dz_m = mm_to_m(defocus_distance)
    defocus_p2v = delta_to_p(
        delta=dz_m,
        f=seal_parameters['focal_length_meters'],
        D=seal_parameters['pupil_size']
    )
    phase_p2v = defocus_p2v * (2*np.pi/seal_parameters['wavelength_meter'])
    defocus_phase = unit_defocus * phase_p2v
    return defocus_phase

def add_read_noise(image_electron, sigma_e, rng):
    read_noise = rng.normal(scale=sigma_e, size=image_electron.shape)
    return image_electron + read_noise


def plot_spot_check(dz_mm, v0, phi_sine_rad, phi_def, psf_focus_clean, psf_defoc_clean,
                    psf0, psfd, psf_reconstruction, real_pupil, truth_phase, residual,
                    rms_residual_nm, rms_total_nm, telescope_pupil, seal_parameters):
    """
    Detailed plotting for a single (dz, v0) point - the full pipeline.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Spot Check: dz={dz_mm:.1f} mm, v0={v0:.2f} cycles/ap', fontsize=14)
    
    # Row 1: Injected phases
    ax = axes[0, 0]
    im = ax.imshow(phi_sine_rad.shaped, cmap='RdBu_r')
    ax.set_title('Injected Sinusoidal Phase [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[0, 1]
    im = ax.imshow(phi_def.shaped, cmap='RdBu_r')
    ax.set_title('Defocus Phase [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[0, 2]
    combined = (phi_sine_rad + phi_def).shaped
    im = ax.imshow(combined, cmap='RdBu_r')
    ax.set_title('Combined Phase (sine + defocus) [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[0, 3]
    im = ax.imshow(telescope_pupil.shaped, cmap='gray')
    ax.set_title('Telescope Pupil')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    # Row 2: PSFs
    ax = axes[1, 0]
    im = ax.imshow(np.log10(psf_focus_clean + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Focused PSF (clean, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1, 1]
    im = ax.imshow(np.log10(psf_defoc_clean + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Defocused PSF (clean, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1, 2]
    im = ax.imshow(np.log10(psf0 + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Focused PSF (noisy, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1, 3]
    im = ax.imshow(np.log10(psfd + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Defocused PSF (noisy, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    # Row 3: Reconstruction
    ax = axes[2, 0]
    im = ax.imshow(np.log10(np.abs(psf_reconstruction)**2 + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('FDPR Output |psf_reconstruction|² (log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[2, 1]
    im = ax.imshow(real_pupil, cmap='RdBu_r')
    ax.set_title('Reconstructed Pupil Phase [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[2, 2]
    im = ax.imshow(truth_phase, cmap='RdBu_r')
    ax.set_title('Truth (Injected Sinusoid) [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[2, 3]
    im = ax.imshow(residual, cmap='RdBu_r')
    ax.set_title(f'Residual (Recon - Truth)\nRMS_res={rms_residual_nm:.1f}nm, RMS_tot={rms_total_nm:.1f}nm')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_verbose_grid_point(i, j, dz_mm, v0, phi_sine_rad, phi_def, 
                            psf_focus_clean, psf_defoc_clean, telescope_pupil):
    """
    Verbose plotting for each grid point (before trials).
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Grid point [{i},{j}]: dz={dz_mm:.1f} mm, v0={v0:.2f} cycles/ap')
    
    ax = axes[0]
    im = ax.imshow(phi_sine_rad.shaped, cmap='RdBu_r')
    ax.set_title('Sinusoidal Phase [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1]
    im = ax.imshow((phi_sine_rad + phi_def).shaped, cmap='RdBu_r')
    ax.set_title('Combined Phase [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[2]
    im = ax.imshow(np.log10(psf_focus_clean + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Focused PSF (log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[3]
    im = ax.imshow(np.log10(psf_defoc_clean + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Defocused PSF (log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
        

    
    plt.tight_layout()
    plt.show()


'''
DEFINING PARAMETERS
'''
seal_parameters = {
    'image_dx': 2.0071,
    'efl': 500,
    'wavelength_meter': 650e-9,
    'pupil_size': 10.12e-3,
    'small_pupil_size_meter': 9.5e-3,
    'pupil_pixel_dimension': 256,
    'focal_length_meters': 500e-3,
    'q': 4,
    'Num_airycircles': 64,
    'grid_dim': 10
}

seal_param_config = {
    'image_dx': 2.0071,
    'efl': seal_parameters['focal_length_meters'] * 1e3,
    'wavelength': 0.65,
    'pupil_size': seal_parameters['pupil_size'] * 1e3,
}

# Build simulation elements
conf = InstrumentConfiguration(seal_param_config)
pupil_grid = make_pupil_grid(256, seal_parameters['pupil_size'])
focal_grid = make_focal_grid(
    q=seal_parameters['q'],
    num_airy=seal_parameters['Num_airycircles'],
    pupil_diameter=seal_parameters['pupil_size'],
    focal_length=seal_parameters['focal_length_meters'],
    reference_wavelength=seal_parameters['wavelength_meter']
)
aperture = make_circular_aperture(seal_parameters['pupil_size'])
telescope_pupil = aperture(pupil_grid)
prop_p2f = FraunhoferPropagator(pupil_grid, focal_grid, seal_parameters['focal_length_meters'])

zernike_modes = make_zernike_basis(
    num_modes=256,
    D=seal_parameters['pupil_size'],
    grid=pupil_grid
)
defocus_template = zernike_modes[3]




m_waves = 0.10  # Peak amplitude of sinusoid in waves
D_m = seal_parameters['pupil_size']
lam_m = seal_parameters['wavelength_meter']

pupil_mask = np.array(telescope_pupil.shaped > 0, dtype=bool)
wavelength_um = seal_param_config['wavelength']

Nd, Nv = len(dzs_mc), len(v0s_mc)

# Time estimation
time_per_sample = 4
time_monte_carlo = Nd * Nv * time_per_sample * N_trials
print(f"\n{'='*50}")
print(f"Monte Carlo Configuration:")
print(f"  Grid size: {Nd} x {Nv} = {Nd*Nv} points")
print(f"  Trials per point: {N_trials}")
print(f"  Read noise: {sigma_e} e-/px")
print(f"  Estimated time: {time_monte_carlo / 3600:.2f} hours")
print(f"  Verbose mode: {verbose}")
print(f"  Spot check: {spot_check} (dz={spot_check_dz}, v0={spot_check_v0})")
print(f"{'='*50}\n")


'''
STORAGE ARRAYS
'''
# Residual RMS (reconstruction - injected sinusoid)
rms_residual_trials = np.full((Nd, Nv, N_trials), np.nan)
rms_residual_mean = np.full((Nd, Nv), np.nan)
rms_residual_std = np.full((Nd, Nv), np.nan)

# Total RMS (just the reconstruction magnitude)
rms_total_trials = np.full((Nd, Nv, N_trials), np.nan)
rms_total_mean = np.full((Nd, Nv), np.nan)
rms_total_std = np.full((Nd, Nv), np.nan)

convergence_rate = np.zeros((Nd, Nv))

t0 = time.time()


'''
MONTE CARLO LOOP
'''
for i, dz in enumerate(dzs_mc):
    dz_mm = float(dz)
    phi_def = calculate_defocus_phase(seal_parameters, dz_mm, telescope_pupil, defocus_template)
    
    for j, v0 in enumerate(v0s_mc):
        if v0 == 0:
            continue

        # Create sinusoidal aberration
        phi_sine_waves = make_sinusoidal_phase_waves(pupil_grid, D_m, float(v0), m_waves)
        phi_sine_rad = 2 * np.pi * phi_sine_waves

        # Generate clean PSFs
        wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), lam_m)
        psf_focus_clean = psf_from_wavefront(wf_focus, prop_p2f)

        wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
        psf_defoc_clean = psf_from_wavefront(wf_defoc, prop_p2f)

        # NaN check
        if not np.all(np.isfinite(psf_focus_clean)) or not np.all(np.isfinite(psf_defoc_clean)):
            print(f"WARNING: Non-finite clean PSF at dz={dz:.1f}, v0={v0:.1f} - skipping")
            continue

        if verbose and i % 1 == 0 and j % 1 == 0:  # Plot every dz and v0
            plot_verbose_grid_point(i, j, dz_mm, v0, phi_sine_rad, phi_def,
                                    psf_focus_clean, psf_defoc_clean, telescope_pupil)

        # Check if this is the spot check point
        is_spot_check = spot_check \
        and np.isclose(dz_mm, spot_check_dz, atol=5) \
        and np.isclose(v0, spot_check_v0, atol=0.5)

        n_converged = 0
        trial_results_residual = []
        trial_results_total = []

        for t in range(N_trials):
            # Add noise
            psf0 = add_read_noise(psf_focus_clean, sigma_e, rng)
            psfd = add_read_noise(psf_defoc_clean, sigma_e, rng)

            psf0 = np.clip(psf0, 0, None)
            psfd = np.clip(psfd, 0, None)

            if not np.all(np.isfinite(psf0)) or not np.all(np.isfinite(psfd)):
                print(f"WARNING: Non-finite noisy PSF at dz={dz:.1f}, v0={v0:.1f}, trial={t}")
                continue

            try:
                # Run FDPR
                dz_um = mm_to_um(dz_mm)  # mm to µm
                mp = FocusDiversePhaseRetrieval(
                    [psf0, psfd], 
                    wavelength_um, 
                    [seal_parameters['image_dx']], 
                    [dz_um]
                )

                for _ in range(100):
                    psf_reconstruction = mp.step()

                if psf_reconstruction is None or not np.any(np.isfinite(psf_reconstruction)):
                    continue

                # Reconstruct pupil phase
                raw_pupil = np.angle(mft_rev(psf_reconstruction, conf))
                
                if not np.any(np.isfinite(raw_pupil)):
                    continue

                real_pupil = resize(raw_pupil, (256, 256)) * telescope_pupil.shaped
                masked_phase = real_pupil[pupil_mask]

                if not np.any(np.isfinite(masked_phase)):
                    continue
                med_subtracted = real_pupil - np.median(real_pupil[np.array(telescope_pupil.shaped, dtype=bool)])
                # Get the injected sinusoidal phase as truth
                truth_phase = phi_sine_rad.shaped *telescope_pupil.shaped
                truth_masked = truth_phase[pupil_mask]

                # Residual RMS (how well did we recover the sinusoid)
                residual = masked_phase - truth_masked
                rms_residual_rad = np.sqrt(np.nanmean(residual**2))

                # Total RMS (magnitude of reconstruction)
                rms_total_rad = np.sqrt(np.nanmean(masked_phase**2))
                # Verbose plotting for this grid point


                if not np.isfinite(rms_residual_rad) or not np.isfinite(rms_total_rad):
                    continue
                if verbose and t==0:
                    plt.imshow(truth_phase-med_subtracted)
                    plt.title(f"Difference image for [{i},{j},{t}]")
                    plt.colorbar()
                    plt.axis('off')
                if verbose and t==5:
                    plt.imshow(truth_phase-med_subtracted)
                    plt.title(f"Difference image for [{i},{j},{t}]")
                    plt.colorbar()
                    plt.axis('off')

                # Convert to nm
                rms_residual_nm = rms_residual_rad * lam_m * 1e9 / (2*np.pi)
                rms_total_nm = rms_total_rad * lam_m * 1e9 / (2*np.pi)

                if np.isfinite(rms_residual_nm) and np.isfinite(rms_total_nm):
                    rms_residual_trials[i, j, t] = rms_residual_nm
                    rms_total_trials[i, j, t] = rms_total_nm
                    trial_results_residual.append(rms_residual_nm)
                    trial_results_total.append(rms_total_nm)
                    n_converged += 1

                    # Spot check detailed plot 
                    if is_spot_check and t == 0:
                        residual_2d = real_pupil - truth_phase
                        plot_spot_check(
                            dz_mm, v0, phi_sine_rad, phi_def,
                            psf_focus_clean, psf_defoc_clean,
                            psf0, psfd, psf_reconstruction, real_pupil, truth_phase, residual_2d,
                            rms_residual_nm, rms_total_nm, telescope_pupil, seal_parameters
                        )

            except Exception as e:
                print(f"FDPR error at dz={dz:.1f}, v0={v0:.1f}, trial={t}: {e}")
                continue

        # Store statistics
        convergence_rate[i, j] = n_converged / N_trials

        if len(trial_results_residual) > 0:
            rms_residual_mean[i, j] = np.mean(trial_results_residual)
            rms_residual_std[i, j] = np.std(trial_results_residual)
        if len(trial_results_total) > 0:
            rms_total_mean[i, j] = np.mean(trial_results_total)
            rms_total_std[i, j] = np.std(trial_results_total)

    # Progress
    elapsed = time.time() - t0
    rate = (i + 1) / elapsed if elapsed > 0 else 0
    remaining = (Nd - i - 1) / rate if rate > 0 else 0
    print(f"Done dz index {i+1}/{Nd} ({dz:.1f} mm) - "
          f"elapsed: {elapsed/60:.1f} min, remaining: {remaining/60:.1f} min")

elapsed = time.time() - t0
print(f"\nMC finished in {elapsed/60:.2f} min")


'''
SUMMARY STATISTICS
'''
print(f"\n{'='*60}")
print("MONTE CARLO TRIALS SUMMARY")
print(f"{'='*60}")
print(f"Grid shape: {Nd} dz x {Nv} v0")
print(f"Trials per point: {N_trials}")

valid_residual = np.sum(np.isfinite(rms_residual_trials))
valid_total = np.sum(np.isfinite(rms_total_trials))
print(f"Valid residual trials: {valid_residual} ({100*valid_residual/rms_residual_trials.size:.1f}%)")
print(f"Valid total trials: {valid_total} ({100*valid_total/rms_total_trials.size:.1f}%)")

valid_res_means = rms_residual_mean[np.isfinite(rms_residual_mean)]
valid_tot_means = rms_total_mean[np.isfinite(rms_total_mean)]

if len(valid_res_means) > 0:
    print(f"\nResidual RMS statistics:")
    print(f"  Mean: {np.mean(valid_res_means):.2f} nm")
    print(f"  Std: {np.std(valid_res_means):.2f} nm")
    print(f"  Range: {np.min(valid_res_means):.2f} - {np.max(valid_res_means):.2f} nm")

if len(valid_tot_means) > 0:
    print(f"\nTotal RMS statistics:")
    print(f"  Mean: {np.mean(valid_tot_means):.2f} nm")
    print(f"  Std: {np.std(valid_tot_means):.2f} nm")
    print(f"  Range: {np.min(valid_tot_means):.2f} - {np.max(valid_tot_means):.2f} nm")

print(f"{'='*60}\n")


'''
SAVE RESULTS
'''

np.savez(save_label,
         dzs_mc=dzs_mc,
         v0s_mc=v0s_mc,
         rms_residual_trials=rms_residual_trials,
         rms_residual_mean=rms_residual_mean,
         rms_residual_std=rms_residual_std,
         rms_total_trials=rms_total_trials,
         rms_total_mean=rms_total_mean,
         rms_total_std=rms_total_std,
         convergence_rate=convergence_rate,
         N_trials=N_trials,
         sigma_e=sigma_e,
         seed=seed,
         m_waves=m_waves)
print(f"Saved: {save_label}")


'''
PLOTTING
'''
extent_mc = [v0s_mc.min(), v0s_mc.max(), dzs_mc.min(), dzs_mc.max()]

# Residual RMS heatmap
plt.figure(figsize=(8, 6))
finite_vals = rms_residual_mean[np.isfinite(rms_residual_mean)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
    plt.imshow(rms_residual_mean, origin='lower', aspect='auto', extent=extent_mc,
               cmap='magma_r', vmin=vmin, vmax=vmax)
else:
    plt.imshow(rms_residual_mean, origin='lower', aspect='auto', extent=extent_mc, cmap='magma_r')
plt.colorbar(label="Residual RMS [nm]")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("MC Residual RMS (Reconstruction - Truth)")
plt.tight_layout()
plt.show()

# Total RMS heatmap
plt.figure(figsize=(8, 6))
finite_vals = rms_total_mean[np.isfinite(rms_total_mean)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
    plt.imshow(rms_total_mean, origin='lower', aspect='auto', extent=extent_mc,
               cmap='magma_r', vmin=vmin, vmax=vmax)
else:
    plt.imshow(rms_total_mean, origin='lower', aspect='auto', extent=extent_mc, cmap='magma_r')
plt.colorbar(label="Total RMS [nm]")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("MC Total RMS (Reconstruction Magnitude)")
plt.tight_layout()
plt.show()

# Convergence rate
plt.figure(figsize=(8, 6))
plt.imshow(convergence_rate, origin='lower', aspect='auto', extent=extent_mc,
           cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label="Convergence rate")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("FDPR Convergence Rate")
plt.tight_layout()
plt.show()

# Residual std heatmap
plt.figure(figsize=(8, 6))
finite_vals = rms_residual_std[np.isfinite(rms_residual_std)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
    plt.imshow(rms_residual_std, origin='lower', aspect='auto', extent=extent_mc,
               cmap='viridis', vmin=vmin, vmax=vmax)
else:
    plt.imshow(rms_residual_std, origin='lower', aspect='auto', extent=extent_mc, cmap='viridis')
plt.colorbar(label="Residual RMS Std [nm]")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("MC Residual RMS Variability")
plt.tight_layout()
plt.show()