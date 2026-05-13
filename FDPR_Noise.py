#FDPR Noise
# ------------------------------
# Monte-Carlo FDPR noise study
# ------------------------------

# --- knobs you can tune ---
N_trials            = 50                 # Monte-Carlo iterations
read_sigma_e        = 11.0               # Gaussian read noise [e-/px rms]
flux_peak_e         = 2.5e5              # peak electrons in the focused PSF (sets SNR)
re_normalize_for_fdpr = False            # set True only if your FDPR expects normalized images
seed0               = 12345              # base seed for reproducibility

# If your FDPR expects two images (e.g., 0 and +Δz), set this True.
# If it expects more, extend the stack accordingly.
use_focus_and_defocus = True

# ------------------------------
# Utilities
# ------------------------------

def piston_removed_rms_nm(phase_rad, pupil_bool, wav_m):
    """RMS WFE (nm) inside pupil after removing piston."""
    ph = np.array(phase_rad, copy=True)
    m  = pupil_bool
    # subtract piston
    piston = np.mean(ph[m])
    ph[m] = ph[m] - piston
    # convert to meters and then to nm
    wfe_m  = phase_to_m(ph, wav_m)
    wfe_nm = wfe_m * 1e9
    return np.sqrt(np.mean((wfe_nm[m])**2))

def to_electrons(psf_norm, peak_e):
    """Scale a peak-normalized PSF (max=1) to electrons."""
    return peak_e * (psf_norm / psf_norm.max() if psf_norm.max() > 0 else psf_norm)

def add_gaussian_read_noise(img_e, sigma_e, rng):
    """Add zero-mean Gaussian read noise in electrons/pixel."""
    noise = rng.normal(loc=0.0, scale=sigma_e, size=img_e.shape)
    return img_e + noise

def maybe_normalize_for_fdpr(img):
    """Optional per-frame normalization if your FDPR expects normalized inputs."""
    if not re_normalize_for_fdpr:
        return img
    m = np.max(img)
    return img / m if m > 0 else img

# ---- Replace this stub with your actual FDPR call ----
def run_fdpr_stack(img_stack_list, conf):
    """
    img_stack_list: [I0, I_defocus, ...] as numpy arrays (float), ideally electrons or normalized intensity
    conf: your InstrumentConfiguration or similar
    RETURNS:
        phase_map_rad_on_pupil_grid (2D numpy array aligned with pupil_grid.shaped)
    """
    # Example skeleton (YOU should replace with your API)
    # fdpr = FocusDiversePhaseRetrieval(conf)
    # out  = fdpr.reconstruct_from_stack(img_stack_list, pupil_grid=pupil_grid, focal_grid=focal_grid)
    # phase_est_rad = out['phase']  # shaped like pupil grid
    # return np.asarray(phase_est_rad)
    raise NotImplementedError("Plug your FDPR API here.")

# Pupil mask (boolean) once
pupil_mask = np.array(telescope_pupil.shaped > 0, dtype=bool)

# Allocate result containers (Nd x Nv)
Nd, Nv = len(fixed_dz_heatmap), len(v0_heatmap)
rms_mean_nm = np.zeros((Nd, Nv)) * np.nan
rms_std_nm  = np.zeros((Nd, Nv)) * np.nan

# ------------------------------
# Monte-Carlo loops
# ------------------------------

for i, dz in enumerate(fixed_dz_heatmap):
    # Build defocus phase once per row (on pupil grid, radians)
    phi_def = calculate_defocus_phase(seal_parameters, float(dz))

    for j, v0 in enumerate(v0_heatmap):
        # Skip out-of-band columns if you want (optional)
        # (We just run everything; FDPR robustness will reflect in RMS.)

        # Build sinusoid (waves) and convert to radians
        phi_sine_waves = make_sinusoidal_phase_waves(pupil_grid, D_m, v0, m_waves)
        phi_sine_rad   = 2*np.pi*phi_sine_waves

        # Make CLEAN PSFs (peak-normalized already by psf_from_wavefront)
        wf_focus = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad)), lam_m)
        psf_focus_clean = psf_from_wavefront(wf_focus)                 # normalized

        wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
        psf_defoc_clean = psf_from_wavefront(wf_defoc)                 # normalized

        # Monte Carlo container for this (dz, v0)
        rms_trials = []

        for t in range(N_trials):
            rng = np.random.default_rng(seed0 + t)

            # Scale to electrons and add read noise
            psf0_e  = to_electrons(psf_focus_clean,  flux_peak_e)
            psfd_e  = to_electrons(psf_defoc_clean, flux_peak_e)

            psf0_noisy = add_gaussian_read_noise(psf0_e, read_sigma_e, rng)
            psfd_noisy = add_gaussian_read_noise(psfd_e, read_sigma_e, rng)

            # Optional re-normalization if your FDPR expects normalized intensities
            I0   = maybe_normalize_for_fdpr(psf0_noisy)
            Idz  = maybe_normalize_for_fdpr(psfd_noisy)

            # Run FDPR
            try:
                if use_focus_and_defocus:
                    phase_est_rad = run_fdpr_stack([I0, Idz], conf)
                else:
                    phase_est_rad = run_fdpr_stack([Idz], conf)  # e.g., if your alg uses only defocus
            except NotImplementedError:
                # Remove this when you wire in your actual FDPR call
                # For now, set NaN so code runs without crashing.
                phase_est_rad = np.full(pupil_grid.shaped.shape, np.nan)

            # Compute masked RMS in nm
            rms_nm = piston_removed_rms_nm(phase_est_rad, pupil_mask, lam_m)
            rms_trials.append(rms_nm)

        # Save trial stats
        rms_trials = np.array(rms_trials, dtype=float)
        rms_mean_nm[i, j] = np.nanmean(rms_trials)
        rms_std_nm[i, j]  = np.nanstd(rms_trials)

# ------------------------------
# Plots: Mean RMS and Std RMS
# ------------------------------
plt.figure(figsize=(8,6))
extent = [v0_heatmap.min(), v0_heatmap.max(),
          fixed_dz_heatmap.min(), fixed_dz_heatmap.max()]
im = plt.imshow(rms_mean_nm, aspect='auto', origin='lower', extent=extent,
                cmap='magma_r')
plt.colorbar(im, label="FDPR phase RMS (nm) — mean over trials")
plt.xlabel(r"Spatial frequency $v_0$ [cycles/aperture]")
plt.ylabel(r"Defocus $\Delta z$ [mm]")
plt.title(f"FDPR Monte-Carlo mean RMS (N={N_trials}, σ_read={read_sigma_e:.0f} e⁻/px)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
im2 = plt.imshow(rms_std_nm, aspect='auto', origin='lower', extent=extent,
                 cmap='viridis')
plt.colorbar(im2, label="FDPR phase RMS (nm) — std over trials")
plt.xlabel(r"Spatial frequency $v_0$ [cycles/aperture]")
plt.ylabel(r"Defocus $\Delta z$ [mm]")
plt.title(f"FDPR Monte-Carlo RMS variability (N={N_trials})")
plt.tight_layout()
plt.show()

# ------------------------------
# (Optional) Overlay your ridge (Δz_max vs v0) on the mean-RMS heatmap
# ------------------------------
# If you already computed points (v0_samples, dz_at_max) from your OTF heatmap:
#   plt.figure(...)
#   ... imshow(rms_mean_nm) ...
#   plt.scatter(v0_samples, dz_at_max, c='w', s=35, edgecolors='k', label='OTF ridge')
#   plt.legend(); plt.show()
