#test 1
import time
import numpy as np
from skimage.transform import resize
from hcipy import *
from image_sharpening import FocusDiversePhaseRetrieval, mft_rev, InstrumentConfiguration
import matplotlib.pyplot as plt


'''
Notes Since last time: 

added otf stuff(clean,defocused,noisy,not noisy, combination)
ran an overnight sim, weird results
fixed nan issue
npz file saves EVERYTHING


TODO:

spot check (ie dz 100 v0 2.5)
    -actually plot the reconstruction
better try/exceprts
make a verbose plotting block -- plot every step(useful for single examples)
seperate the analytical(monte carlo) and semi-analytical (OTF)
    -one script solely MC
    -One script solely OTF space

create a difference image with the pupil phase we had initially injected(sinusoidal phase waves)
    -something like reconstruction minus the 'system truth'
true reflection of reconstruction fo sinusoidal pattern
actual reconstruction to sinusoids is missing

'''
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
        # Every PSF looks "equally bright" at the core
        return np.asarray(I)

#THIS SHOULD BE IN MM
def calculate_defocus_phase(seal_parameters,
                            defocus_distance):
    mask = np.array(telescope_pupil.shaped, dtype=bool)
    defocus_template_s=defocus_template.shaped
    template_p2v=defocus_template_s[mask].max() - defocus_template_s[mask].min()
    unit_defocus = defocus_template / template_p2v
    dz_m = mm_to_m(defocus_distance)
    # FIXED: Use parameters from dictionary instead of hardcoded values
    defocus_p2v = delta_to_p(
                            delta = dz_m,
                            f = seal_parameters['focal_length_meters'],
                            D = seal_parameters['pupil_size']
                            )
    phase_p2v = defocus_p2v * (2*np.pi/seal_parameters['wavelength_meter'])
    defocus_phase = unit_defocus * phase_p2v
    print('defocus_p2v value:',defocus_p2v)
    print('phase_p2v value', phase_p2v)
    return defocus_phase

def add_read_noise(image_electron, sigma_e, rng):
    read_noise = rng.normal(scale=sigma_e, size=image_electron.shape)
    return image_electron + read_noise

def otf_from_psf_numpy(psf):
    OTF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    mag = np.abs(OTF)
    return mag

def find_otf_sidepeaks_1D(OTF, kill_core_pix=9, subpixel=True):
    row = OTF[OTF.shape[0] // 2, :].copy()
    c = len(row) // 2
    row[c-kill_core_pix:c+kill_core_pix+1] = 0.0
    left, right = row[:c], row[c+1:]
    il, ir = int(np.argmax(left)), int(np.argmax(right))
    amp_l, amp_r = float(left[il]), float(right[ir])
    amp = 0.5*(amp_l + amp_r)
    return float(amp)

'''
DEFINING PARAMETERS, HARDCODES VALUES
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

# FIXED: Create propagator (was used in psf_from_wavefront but never defined)
prop_p2f = FraunhoferPropagator(pupil_grid, focal_grid, seal_parameters['focal_length_meters'])

zernike_modes = make_zernike_basis(
         num_modes=256,
         D=seal_parameters['pupil_size'],
         grid=pupil_grid
    )
defocus_template = zernike_modes[3]


'''
START OF MONTE CARLO 

'''
#monte carlo in mm, cycles/ap
dzs_mc = np.linspace(5, 250, 25)     # mm
v0s_mc = np.linspace(0.5, 20, 40)      # cycles/aperture

'''#zoom ine
dzs_mc = np.linspace(20, 100, 40)    # focus on low-defocus sweet spot
v0s_mc = np.linspace(2, 18, 40)      # mid-range frequencies
'''
# monte carlo parameters
N_trials = 5
sigma_e  = 11
seed     = 12345
rng      = np.random.default_rng(seed)


m_waves = 0.10  # Peak amplitude of sinusoid in waves
D_m = seal_parameters['pupil_size']
lam_m = seal_parameters['wavelength_meter']

conf = InstrumentConfiguration(seal_param_config)

pupil_mask = np.array(telescope_pupil.shaped > 0, dtype=bool)

wavelength_um = seal_param_config['wavelength']

Nd, Nv = len(dzs_mc), len(v0s_mc)

# Time estimation
time_per_sample = 4  # estimated seconds per FDPR run
time_monte_carlo = Nd * Nv * time_per_sample * N_trials
print(f"\n{'='*30}")
print(f"Monte Carlo Configuration:")
print(f"  Grid size: {Nd} x {Nv} = {Nd*Nv} points")
print(f"  Trials per point: {N_trials}")
print(f"  Read noise: {sigma_e} e-/px")
print(f"  Estimated time: {time_monte_carlo / 3600:.2f} hours")
print(f"{'='*30}\n")


#STORAGE
amp_trials = np.full((Nd, Nv, N_trials), np.nan)
amp_mean   = np.full((Nd, Nv), np.nan)
amp_std    = np.full((Nd, Nv), np.nan)

# OTF amplitudes for different PSF types
otf_amp_focus_clean = np.full((Nd, Nv), np.nan)      # focused, no noise
otf_amp_defoc_clean = np.full((Nd, Nv), np.nan)      # defocused, no noise
otf_amp_focus_noisy = np.full((Nd, Nv, N_trials), np.nan)  # focused, with noise (per trial)
otf_amp_defoc_noisy = np.full((Nd, Nv, N_trials), np.nan)  # defocused, with noise (per trial)
otf_amp_reconstructed = np.full((Nd, Nv, N_trials), np.nan)  # FDPR output (per trial)

rms_trials_nm = np.full((Nd, Nv, N_trials), np.nan)
rms_mean_nm   = np.full((Nd, Nv), np.nan)
rms_std_nm    = np.full((Nd, Nv), np.nan)
convergence_rate = np.zeros((Nd, Nv))

t0 = time.time()


'''

Wrap alot of the into functions:

function to :
make clean psfs
make defocus psfs '''
for i, dz in enumerate(dzs_mc):
    dz_mm = float(dz)
    phi_def = calculate_defocus_phase(seal_parameters, dz_mm)
    for j, v0 in enumerate(v0s_mc):
        
        if v0 == 0:
            continue

        phi_sine_waves = make_sinusoidal_phase_waves(pupil_grid, D_m, float(v0), m_waves)
        phi_sine_rad   = 2*np.pi * phi_sine_waves

        wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), lam_m)
        psf_focus_clean = psf_from_wavefront(wf_focus)

        wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
        psf_defoc_clean = psf_from_wavefront(wf_defoc)

        if not np.all(np.isfinite(psf_focus_clean)) or not np.all(np.isfinite(psf_defoc_clean)):
            print(f"WARNING: Non-finite clean PSF at dz={dz:.1f}, v0={v0:.1f} - skipping")
            continue

        # OTF of clean PSFs 
        try:
            otf_focus_clean = otf_from_psf_numpy(psf_focus_clean)
            otf_amp_focus_clean[i, j] = find_otf_sidepeaks_1D(otf_focus_clean, kill_core_pix=9)
            
            otf_defoc_clean = otf_from_psf_numpy(psf_defoc_clean)
            otf_amp_defoc_clean[i, j] = find_otf_sidepeaks_1D(otf_defoc_clean, kill_core_pix=9)
        except:
            pass

        n_converged = 0
        trial_results = []
        trial_amplitudes = []
        
        for t in range(N_trials):
            psf0 = add_read_noise(psf_focus_clean, sigma_e, rng)
            psfd = add_read_noise(psf_defoc_clean, sigma_e, rng)

            psf0 = np.clip(psf0, 0, None)
            psfd = np.clip(psfd, 0, None)

            if not np.all(np.isfinite(psf0)) or not np.all(np.isfinite(psfd)):
                print(f"WARNING: Non-finite noisy PSF at dz={dz:.1f}, v0={v0:.1f}, trial={t}")
                continue

            # OTF of noisy PSFs (per trial) #NOT NECESSARY PER TRIAL
            try:
                otf_focus_noisy = otf_from_psf_numpy(psf0)
                otf_amp_focus_noisy[i, j, t] = find_otf_sidepeaks_1D(otf_focus_noisy, kill_core_pix=9)
                
                otf_defoc_noisy = otf_from_psf_numpy(psfd)
                otf_amp_defoc_noisy[i, j, t] = find_otf_sidepeaks_1D(otf_defoc_noisy, kill_core_pix=9)
            except:
                pass

            try:
                dz_um = dz_mm * 1e3
                mp = FocusDiversePhaseRetrieval([psf0, psfd], wavelength_um, [seal_parameters['image_dx']], [dz_um])

                for _ in range(100):
                    psf00 = mp.step()

                if psf00 is None or not np.any(np.isfinite(psf00)):
                    continue

                raw_pupil = np.angle(mft_rev(psf00, conf))
                
                if not np.any(np.isfinite(raw_pupil)):
                    continue
                    
                real_pupil = resize(raw_pupil, (256, 256)) * telescope_pupil.shaped #pupil phase

                '''
                Thoughts for the difference
                
                median_subtracted = real_pupil - np.median(real_pupil[np.array(telescope_pupil.shaped, dtype=bool)])

                difference_subtracted = (pupil_image.phase.shaped - med_subtracted)

                error_region = difference_subtracted[np.array(telescope_pupil.shaped, dtype=bool)]

                '''



                masked_phase = real_pupil[pupil_mask]
                

                if not np.any(np.isfinite(masked_phase)):
                    continue
                mask = np.array(telescope_pupil.shaped, dtype=bool)
                truth_masked = telescope_pupil.shaped[mask]
                phase_rms_rad = np.sqrt(np.nanmean((masked_phase-truth_masked)**2))

                
                if not np.isfinite(phase_rms_rad) or phase_rms_rad < 0:
                    continue
                    
                wfe_rms_nm = phase_rms_rad * lam_m * 1e9 / (2*np.pi)
                
                if np.isfinite(wfe_rms_nm) and wfe_rms_nm >= 0:
                    rms_trials_nm[i, j, t] = wfe_rms_nm
                    trial_results.append(wfe_rms_nm)
                    n_converged += 1
                    
                    # OTF of reconstructed PSF
                    try:
                        otf_rec = otf_from_psf_numpy(np.abs(psf00)**2)
                        recovered_amp = find_otf_sidepeaks_1D(otf_rec, kill_core_pix=9)
                        if np.isfinite(recovered_amp):
                            otf_amp_reconstructed[i, j, t] = recovered_amp
                            amp_trials[i, j, t] = recovered_amp
                            trial_amplitudes.append(recovered_amp)
                    except:
                        pass
                    
            except Exception as e:
                print(f"FDPR error at dz={dz:.1f}, v0={v0:.1f}, trial={t}: {e}")
                continue

        convergence_rate[i, j] = n_converged / N_trials
        
        if len(trial_results) > 0:
            rms_mean_nm[i, j] = np.mean(trial_results)
            rms_std_nm[i, j] = np.std(trial_results)
        if len(trial_amplitudes) > 0:
            amp_mean[i, j] = np.mean(trial_amplitudes)
            amp_std[i, j] = np.std(trial_amplitudes)
        # else: leave as NaN (initialized value)

    # Progress with timing info
    elapsed = time.time() - t0
    rate = (i + 1) / elapsed if elapsed > 0 else 0
    remaining = (Nd - i - 1) / rate if rate > 0 else 0
    print(f"Done dz index {i+1}/{Nd} ({dz:.1f} mm) - "
          f"elapsed: {elapsed/60:.1f} min, remaining: {remaining/60:.1f} min")

elapsed = time.time() - t0
print(f"MC finished in {elapsed/60:.2f} min")

# Print summary statistics
print(f"\n{'='*60}")
print("MONTE CARLO TRIALS SUMMARY")
print(f"{'='*60}")
print(f"Grid shape: {Nd} dz x {Nv} v0")
print(f"Trials per point: {N_trials}")
print(f"Total trial slots: {rms_trials_nm.size}")
valid_count = np.sum(np.isfinite(rms_trials_nm))
print(f"Valid trials: {valid_count} ({100*valid_count/rms_trials_nm.size:.1f}%)")

valid_per_point = np.sum(np.isfinite(rms_trials_nm), axis=2)
print(f"\nConvergence by grid point:")
print(f"  All {N_trials} trials converged: {np.sum(valid_per_point == N_trials)} points")
print(f"  Some trials converged: {np.sum((valid_per_point > 0) & (valid_per_point < N_trials))} points")
print(f"  No trials converged: {np.sum(valid_per_point == 0)} points")

valid_means = rms_mean_nm[np.isfinite(rms_mean_nm)]
if len(valid_means) > 0:
    print(f"\nRMS WFE statistics across grid:")
    print(f"  Mean of means: {np.mean(valid_means):.2f} nm")
    print(f"  Std of means: {np.std(valid_means):.2f} nm")
    print(f"  Range: {np.min(valid_means):.2f} - {np.max(valid_means):.2f} nm")
print(f"{'='*60}\n")

save_label="OTF_heatmap_amps.npz"
np.savez(save_label,
         dzs_mc=dzs_mc,
         v0s_mc=v0s_mc,
         rms_trials_nm=rms_trials_nm,
         otf_amp_focus_clean=otf_amp_focus_clean,
         otf_amp_defoc_clean=otf_amp_defoc_clean,
         otf_amp_focus_noisy=otf_amp_focus_noisy,
         otf_amp_defoc_noisy=otf_amp_defoc_noisy,
         otf_amp_reconstructed=otf_amp_reconstructed,
         amp_trials=amp_trials,
         amp_mean=amp_mean,
         amp_std=amp_std,
         rms_mean_nm=rms_mean_nm,
         rms_std_nm=rms_std_nm,
         convergence_rate=convergence_rate,
         N_trials=N_trials,
         sigma_e=sigma_e,
         seed=seed,
         m_waves=m_waves)
print(f"Saved: {save_label}")

'''
IF WE WANT TO LOAD PREVIOUS
'''
load_heatmap=True
if load_heatmap:
            loaded_heatmap = np.load("OTF_heatmap_data.npz")
            H = loaded_heatmap["H"]
            fixed_dz_heatmap= loaded_heatmap["fixed_dz_heatmap"]
            v0_heatmap = loaded_heatmap["v0_heatmap"]
            # otf heatmap
            otf = np.load("OTF_heatmap_data.npz")
            H = otf["H"]
            fixed_dz_heatmap = otf["fixed_dz_heatmap"]
            v0_heatmap = otf["v0_heatmap"]
            # orginal otf heatmap
            extent_otf = [v0_heatmap.min(), v0_heatmap.max(),
                        fixed_dz_heatmap.min(), fixed_dz_heatmap.max()]
            plt.figure(figsize=(8,6))
            plt.imshow(H, origin='lower', aspect='auto', extent=extent_otf, cmap='viridis')
            plt.colorbar(label="OTF side-peak amplitude")
            plt.xlabel("v0 [cycles/ap]")
            plt.ylabel("dz [mm]")
            plt.title("OTF heatmap")
            plt.tight_layout()
            plt.show()

'''

THIS IS FOR PLOTTING 
'''


# monte carlo heatmap
mc = np.load("MC_RMS_heatmap_CORRECTED_dz.npz")
dzs_mc = mc["dzs_mc"]
v0s_mc = mc["v0s_mc"]
rms_mean_nm = mc["rms_mean_nm"]
rms_std_nm  = mc["rms_std_nm"]
rms_trials_nm = mc["rms_trials_nm"]
convergence_rate = mc["convergence_rate"] if "convergence_rate" in mc else None
dzs = mc["dzs_mc"]
v0s = mc["v0s_mc"]
mean = mc["rms_mean_nm"]


print("OTF H finite fraction:", np.isfinite(H).mean())
print("MC mean finite fraction:", np.isfinite(rms_mean_nm).mean())
print("MC std finite fraction:", np.isfinite(rms_std_nm).mean())

# Check if we have enough valid data to plot
if np.sum(np.isfinite(rms_mean_nm)) == 0:
    print("WARNING: rms_mean_nm contains all NaNs - no valid MC data to plot!")
if np.sum(np.isfinite(rms_std_nm)) == 0:
    print("WARNING: rms_std_nm contains all NaNs - no valid MC std data to plot!")



# monte carlo mean
extent_mc = [v0s_mc.min(), v0s_mc.max(),
             dzs_mc.min(), dzs_mc.max()]

# OTF amplitude heatmap
#Trying to recreate previous heatmap
plt.figure(figsize=(8,6))
plt.imshow(amp_mean, origin='lower', aspect='auto', extent=extent_mc, cmap='viridis')
plt.colorbar(label="OTF side-peak amplitude")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("Recovered OTF Amplitude Heatmap")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
finite_vals = rms_mean_nm[np.isfinite(rms_mean_nm)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
    plt.imshow(rms_mean_nm, origin='lower', aspect='auto', extent=extent_mc,
               cmap='magma_r', vmin=vmin, vmax=vmax)
    plt.colorbar(label="FDPR WFE RMS mean [nm]")
else:
    plt.imshow(rms_mean_nm, origin='lower', aspect='auto', extent=extent_mc,
               cmap='magma_r')
    plt.colorbar(label="FDPR WFE RMS mean [nm] (ALL NaN)")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("MC mean RMS heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
finite_vals = rms_trials_nm[np.isfinite(rms_trials_nm)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
    plt.imshow(rms_trials_nm, origin='lower', aspect='auto', extent=extent_mc,
               cmap='magma_r', vmin=vmin, vmax=vmax)
    plt.colorbar(label="FDPR WFE RMS mean [nm]")
else:
    plt.imshow(rms_trials_nm, origin='lower', aspect='auto', extent=extent_mc,
               cmap='magma_r')
    plt.colorbar(label="FDPR WFE RMS mean [nm] (ALL NaN)")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("MC trials RMS heatmap")
plt.tight_layout()
plt.show()

#monte carlo std deviation
plt.figure(figsize=(8,6))
finite = np.isfinite(rms_std_nm)
if np.any(finite):
    vmin, vmax = np.nanpercentile(rms_std_nm[finite], [5, 95])
    plt.imshow(rms_std_nm, origin='lower', aspect='auto', extent=extent_mc,
               cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label="FDPR WFE RMS std [nm]")
else:
    plt.imshow(rms_std_nm, origin='lower', aspect='auto', extent=extent_mc,
               cmap='viridis')
    plt.colorbar(label="FDPR WFE RMS std [nm] (ALL NaN)")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("MC RMS variability heatmap")
plt.tight_layout()
plt.show()
'''
# convergence rate heatmap
if convergence_rate is not None:
    plt.figure(figsize=(8,6))
    plt.imshow(convergence_rate, origin='lower', aspect='auto', extent=extent_mc,
               cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label="FDPR convergence rate")
    plt.xlabel("v0 [cycles/ap]")
    plt.ylabel("dz [mm]")
    plt.title("FDPR Convergence Rate")
    plt.tight_layout()
    plt.show()
'''
#trial distribution with histogram
i, j = 5, 5
vals = rms_trials_nm[i, j, :]
valid_vals = vals[np.isfinite(vals)]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#trial values
axes[0].plot(vals, 'o-')
axes[0].set_xlabel("trial index")
axes[0].set_ylabel("WFE RMS [nm]")
axes[0].set_title(f"Trials at dz={dzs_mc[i]:.1f} mm, v0={v0s_mc[j]:.2f}")
axes[0].grid(True, alpha=0.3)

#histogram of valid values
if len(valid_vals) > 1:
    axes[1].hist(valid_vals, bins='auto', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(valid_vals), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(valid_vals):.1f} nm')
    axes[1].axvline(np.median(valid_vals), color='g', linestyle=':', 
                    label=f'Median: {np.median(valid_vals):.1f} nm')
    axes[1].set_xlabel("WFE RMS [nm]")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Distribution (n={len(valid_vals)}, std={np.std(valid_vals):.1f} nm)")
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, f"Only {len(valid_vals)} valid trial(s)", 
                 ha='center', va='center', transform=axes[1].transAxes)

plt.tight_layout()
plt.show()

# Print trial statistics
print(f"\nTrial statistics at dz={dzs_mc[i]:.1f} mm, v0={v0s_mc[j]:.2f}:")
print(f"  Total trials: {len(vals)}")
print(f"  Valid trials: {len(valid_vals)} ({100*len(valid_vals)/len(vals):.0f}%)")
if len(valid_vals) > 0:
    print(f"  Mean RMS: {np.mean(valid_vals):.2f} nm")
    print(f"  Std RMS: {np.std(valid_vals):.2f} nm")
    print(f"  Min/Max: {np.min(valid_vals):.2f} / {np.max(valid_vals):.2f} nm")


'''
ADDITIONAL rms_trials_nm VISUALIZATIONS
'''

# Heatmap of a single trial 
trial_idx = 0
plt.figure(figsize=(8,6))
trial_data = rms_trials_nm[:, :, trial_idx]
finite_vals = trial_data[np.isfinite(trial_data)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
else:
    vmin, vmax = 0, 1
plt.imshow(trial_data, origin='lower', aspect='auto', extent=extent_mc,
           cmap='magma_r', vmin=vmin, vmax=vmax)
plt.colorbar(label=f"WFE RMS [nm] - Trial {trial_idx}")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title(f"Single Trial #{trial_idx} RMS Heatmap")
plt.tight_layout()
plt.show()


#PICK A CERTAIN DZ

dz_target = 50.0  # mm
i = int(np.argmin(np.abs(dzs - dz_target)))

plt.figure(figsize=(9,5))
plt.plot(v0s, mean[i, :], "o-")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("Mean WFE RMS [nm]")
plt.title(f"Slice at dz={dzs[i]:.1f} mm")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# All trials at a fixed defocus (slice across v0)
dz_idx = min(10, len(dzs_mc) - 1)  # pick a defocus index
plt.figure(figsize=(10, 5))
for t in range(rms_trials_nm.shape[2]):
    trial_slice = rms_trials_nm[dz_idx, :, t]
    plt.plot(v0s_mc, trial_slice, 'o-', alpha=0.6, label=f'Trial {t}')
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("WFE RMS [nm]")
plt.title(f"All trials at dz={dzs_mc[dz_idx]:.1f} mm")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# All trials at a fixed spatial frequency (slice across dz)
v0_idx = min(10, len(v0s_mc) - 1)  # pick a v0 index
plt.figure(figsize=(10, 5))
for t in range(rms_trials_nm.shape[2]):
    trial_slice = rms_trials_nm[:, v0_idx, t]
    plt.plot(dzs_mc, trial_slice, 'o-', alpha=0.6, label=f'Trial {t}')
plt.xlabel("dz [mm]")
plt.ylabel("WFE RMS [nm]")
plt.title(f"All trials at v0={v0s_mc[v0_idx]:.2f} cycles/ap")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# max - min across trials
trial_range = np.nanmax(rms_trials_nm, axis=2) - np.nanmin(rms_trials_nm, axis=2)
plt.figure(figsize=(8,6))
finite_vals = trial_range[np.isfinite(trial_range)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
else:
    vmin, vmax = 0, 1
plt.imshow(trial_range, origin='lower', aspect='auto', extent=extent_mc,
           cmap='plasma', vmin=vmin, vmax=vmax)
plt.colorbar(label="WFE RMS range (max-min) [nm]")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("Trial-to-Trial Variability (Max - Min across trials)")
plt.tight_layout()
plt.show()

'''amp_mean = mc["amp_mean"] if "amp_mean" in mc else None
if amp_mean is not None:
    plt.figure(figsize=(8,6))
    finite_amp = amp_mean[np.isfinite(amp_mean)]
    if len(finite_amp) > 0:
        vmin, vmax = np.nanpercentile(finite_amp, [5, 95])
        plt.imshow(amp_mean, origin='lower', aspect='auto', extent=extent_mc,
                   cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(label="Recovered sinusoid amplitude [waves]")
    else:
        plt.imshow(amp_mean, origin='lower', aspect='auto', extent=extent_mc,
                   cmap='viridis')
        plt.colorbar(label="Recovered sinusoid amplitude [waves] (ALL NaN)")
    plt.xlabel("v0 [cycles/ap]")
    plt.ylabel("dz [mm]")
    plt.title(f"Recovered Sinusoid Amplitude (injected: {m_waves} waves)")
    plt.tight_layout()
    plt.show()
    
    # Amplitude recovery ratio (recovered / injected)
    # Should be ~1.0 where FDPR works well
    plt.figure(figsize=(8,6))
    amp_ratio = amp_mean / m_waves
    finite_ratio = amp_ratio[np.isfinite(amp_ratio)]
    if len(finite_ratio) > 0:
        vmin, vmax = np.nanpercentile(finite_ratio, [5, 95])
        plt.imshow(amp_ratio, origin='lower', aspect='auto', extent=extent_mc,
                   cmap='RdBu_r', vmin=0, vmax=2)  # 0-2 range centered on 1
        plt.colorbar(label="Amplitude ratio (recovered/injected)")
    plt.xlabel("v0 [cycles/ap]")
    plt.ylabel("dz [mm]")
    plt.title("Sinusoid Recovery Ratio (1.0 = perfect recovery)")
    plt.tight_layout()
    plt.show()
    '''