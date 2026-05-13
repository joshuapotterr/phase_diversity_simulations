"""
fdpr_mc_pm_sinamp.py
====================
Monte Carlo sweep over (dz, sinusoid_amplitude) at a fixed low spatial
frequency.  Uses ±dz diversity (3 PSFs per trial).

Motivation: the v0 sweep at 0.5 waves amplitude showed the algorithm
fails for sinusoids because the phase is too large (π rad peak).
This script instead fixes v0 and sweeps amplitude from 0.05→0.4 waves
to find the algorithm's dynamic range for sinusoidal aberrations.

Grid axes:
  Axis 0 — dz        : defocus distance [mm]  (±dz pair per trial)
  Axis 1 — amplitude : sinusoid amplitude [waves]

Unit convention (identical to fdpr_injectable.py):
  HCIPy lengths : meters
  FDPR lib      : micrometers  (wavelength, image_dx, defocus)
  Config values : millimeters  (defocus distances)
  Phase         : radians
  Reported error: nm
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
from skimage.transform import resize
from image_sharpening import FocusDiversePhaseRetrieval, mft_rev, InstrumentConfiguration

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# --- Aberration type ---
ABERRATION_TYPE = "sinusoid"

# Fixed spatial frequency — low enough for the algorithm to converge
SINUSOID_V0_FIXED = 2.0          # cycles/aperture (fixed)

# --- Grid ---
# Axis 0: defocus distances [mm] — ±dz pair used per trial
DZ_GRID_MM = np.linspace(5, 100, 10)

# Axis 1: sinusoid amplitude [waves]  — swept to find dynamic range
SINUSOID_AMP_GRID = np.linspace(0.05, 0.4, 10)

# --- Monte Carlo ---
N_TRIALS   = 5
N_ITER     = 200
IF_NOISE   = True
SIGMA_E    = 11          # read noise [e⁻/pixel]
SEED       = 12345

# --- Output ---
SAVE_LABEL = f"fdpr_mc_pm_sinamp_v{SINUSOID_V0_FIXED:.0f}.npz"

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

NUM_PHOTONS = 1e6
RNG = np.random.default_rng(SEED)

seal_parameters = {
    'image_dx':              2.0071,
    'focal_length_meters':   500e-3,
    'wavelength_meter':      650e-9,
    'pupil_size':            10.12e-3,
    'pupil_pixel_dimension': 256,
    'q':                     16,
    'Num_airycircles':       16,
}

seal_param_config = {
    'image_dx':   seal_parameters['image_dx'],
    'efl':        seal_parameters['focal_length_meters'] * 1e3,   # mm
    'wavelength': seal_parameters['wavelength_meter']    * 1e6,   # µm
    'pupil_size': seal_parameters['pupil_size']          * 1e3,   # mm
}

WAVELENGTH_UM = seal_param_config['wavelength']
IMAGE_DX_UM   = seal_param_config['image_dx']

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ELEMENTS
# ─────────────────────────────────────────────────────────────────────────────

conf = InstrumentConfiguration(seal_param_config)

pupil_grid = make_pupil_grid(
    seal_parameters['pupil_pixel_dimension'],
    seal_parameters['pupil_size']
)
focal_grid = make_focal_grid(
    q=seal_parameters['q'],
    num_airy=seal_parameters['Num_airycircles'],
    pupil_diameter=seal_parameters['pupil_size'],
    focal_length=seal_parameters['focal_length_meters'],
    reference_wavelength=seal_parameters['wavelength_meter']
)

aperture        = make_circular_aperture(seal_parameters['pupil_size'])
telescope_pupil = aperture(pupil_grid)
pupil_mask      = np.array(telescope_pupil.shaped, dtype=bool)

prop_p2f = FraunhoferPropagator(
    pupil_grid, focal_grid,
    seal_parameters['focal_length_meters']
)

zernike_modes    = make_zernike_basis(
    num_modes=256,
    D=seal_parameters['pupil_size'],
    grid=pupil_grid
)
defocus_template = zernike_modes[3]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def delta_to_p(delta_m, f_m, D_m):
    """Mechanical defocus distance [m] → P2V OPD [m]."""
    return -delta_m / (8.0 * (f_m / D_m) ** 2)


def calculate_defocus_phase(defocus_mm):
    """Zernike defocus phase [rad] for a given mechanical defocus [mm]."""
    template_s   = defocus_template.shaped
    template_p2v = template_s[pupil_mask].max() - template_s[pupil_mask].min()
    unit_defocus  = defocus_template / template_p2v

    dz_m          = defocus_mm * 1e-3
    defocus_p2v_m = delta_to_p(
        delta_m=dz_m,
        f_m=seal_parameters['focal_length_meters'],
        D_m=seal_parameters['pupil_size'],
    )
    phase_p2v_rad = defocus_p2v_m * (2.0 * np.pi / seal_parameters['wavelength_meter'])
    return unit_defocus * phase_p2v_rad


def make_sinusoidal_phase(cycles_per_aperture, amplitude_waves):
    """Single-frequency sinusoid along x [rad]."""
    x = pupil_grid.x
    D = seal_parameters['pupil_size']
    return Field(
        amplitude_waves * np.sin(2.0 * np.pi * cycles_per_aperture * (x / D)) * 2.0 * np.pi,
        pupil_grid
    )


def make_zernike_phase(mode_index, amplitude_waves):
    """Zernike phase [rad] normalised to a given P2V amplitude in waves."""
    mode   = zernike_modes[mode_index]
    mode_s = mode.shaped
    p2v    = mode_s[pupil_mask].max() - mode_s[pupil_mask].min()
    return (mode / p2v) * (amplitude_waves * 2.0 * np.pi)


def psf_from_wavefront(wf):
    """Focal-plane PSF [photons/pixel] using .power with total_power set."""
    return np.asarray(prop_p2f(wf).power.shaped, dtype=float)


def make_wavefront(phase_rad):
    """Pupil-plane wavefront normalised to NUM_PHOTONS total power."""
    wf = Wavefront(
        telescope_pupil * np.exp(1j * phase_rad),
        seal_parameters['wavelength_meter']
    )
    wf.total_power = NUM_PHOTONS
    return wf


def add_read_noise(psf):
    """Gaussian read noise [e⁻/px], clipped to ≥ 0."""
    return np.clip(psf + RNG.normal(scale=SIGMA_E, size=psf.shape), 0.0, None)


def run_fdpr(psf_focused, psf_defoc_pos, psf_defoc_neg, dz_mm):
    """
    Run FocusDiversePhaseRetrieval with ±dz diversity to break sign ambiguity.

    PSF list: [focused, +dz, -dz]
    dx list : [IMAGE_DX_UM, IMAGE_DX_UM]
    dz list : [+dz_um, -dz_um]

    Returns the reconstructed pupil phase [rad], masked to the pupil.
    """
    psf0   = add_read_noise(psf_focused)    if IF_NOISE else psf_focused.copy()
    psfd_p = add_read_noise(psf_defoc_pos)  if IF_NOISE else psf_defoc_pos.copy()
    psfd_n = add_read_noise(psf_defoc_neg)  if IF_NOISE else psf_defoc_neg.copy()

    if not (np.all(np.isfinite(psf0)) and
            np.all(np.isfinite(psfd_p)) and
            np.all(np.isfinite(psfd_n))):
        return None

    mp = FocusDiversePhaseRetrieval(
        [psf0, psfd_p, psfd_n],
        WAVELENGTH_UM,
        [IMAGE_DX_UM, IMAGE_DX_UM],
        [dz_mm * 1e3, -dz_mm * 1e3],   # ±dz in µm
    )
    for _ in range(N_ITER):
        psf_rec = mp.step()

    raw   = np.angle(mft_rev(psf_rec, conf))
    recon = resize(raw, (256, 256), preserve_range=True) * telescope_pupil.shaped
    return recon


def rms_residual_nm(recon, truth_shaped):
    """
    RMS of (truth − median-subtracted reconstruction) inside the pupil [nm].
    Returns np.nan on failure.
    """
    if recon is None or not np.any(np.isfinite(recon[pupil_mask])):
        return np.nan

    rec_ms = recon[pupil_mask] - np.median(recon[pupil_mask])

    truth_resize  = resize(truth_shaped, (256, 256), preserve_range=True)
    truth_ms      = truth_resize[pupil_mask] - np.median(truth_resize[pupil_mask])

    residual = truth_ms - rec_ms
    rms_rad  = np.sqrt(np.nanmean(residual ** 2))
    return rms_rad * seal_parameters['wavelength_meter'] / (2.0 * np.pi) * 1e9

# ─────────────────────────────────────────────────────────────────────────────
# GRID SETUP
# ─────────────────────────────────────────────────────────────────────────────

param_grid  = SINUSOID_AMP_GRID
param_label = f"Amplitude [waves]  (sinusoid v₀={SINUSOID_V0_FIXED} cyc/ap)"
param_name  = "amplitude_waves"

Ndz  = len(DZ_GRID_MM)
Np   = len(param_grid)

rms_trials = np.full((Ndz, Np, N_TRIALS), np.nan)
rms_mean   = np.full((Ndz, Np), np.nan)
rms_std    = np.full((Ndz, Np), np.nan)
conv_rate  = np.zeros((Ndz, Np))

print(f"\n{'='*60}")
print(f"FDPR Monte Carlo (±dz diversity)  —  sinusoid amplitude sweep")
print(f"  Grid  : {Ndz} dz × {Np} amplitude = {Ndz*Np} points")
print(f"  Trials: {N_TRIALS}   Iterations: {N_ITER}   Noise: {IF_NOISE}")
print(f"  Diversity: ±dz  (3 PSFs per trial)")
print(f"  Fixed v0    : {SINUSOID_V0_FIXED} cyc/aperture")
print(f"  dz range    : {DZ_GRID_MM[0]:.1f} – {DZ_GRID_MM[-1]:.1f} mm")
print(f"  amp range   : {param_grid[0]:.3f} – {param_grid[-1]:.3f} waves")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO LOOP
# ─────────────────────────────────────────────────────────────────────────────

t0 = time.time()

for i, dz_mm in enumerate(DZ_GRID_MM):

    phi_def = calculate_defocus_phase(dz_mm)

    for j, param in enumerate(param_grid):

        # Build aberration phase for this grid point
        phi_ab = make_sinusoidal_phase(SINUSOID_V0_FIXED, float(param))

        truth_shaped = phi_ab.shaped * telescope_pupil.shaped

        # Clean PSFs — noise added per-trial inside run_fdpr
        psf_focused   = psf_from_wavefront(make_wavefront(phi_ab))
        psf_defoc_pos = psf_from_wavefront(make_wavefront(phi_ab + phi_def))
        psf_defoc_neg = psf_from_wavefront(make_wavefront(phi_ab - phi_def))

        if not (np.all(np.isfinite(psf_focused)) and
                np.all(np.isfinite(psf_defoc_pos)) and
                np.all(np.isfinite(psf_defoc_neg))):
            print(f"  Non-finite PSF at dz={dz_mm:.1f}, param={param:.2f} — skipping")
            continue

        trial_rms = []
        for t in range(N_TRIALS):
            recon = run_fdpr(psf_focused, psf_defoc_pos, psf_defoc_neg, dz_mm)
            r     = rms_residual_nm(recon, truth_shaped)
            if np.isfinite(r):
                rms_trials[i, j, t] = r
                trial_rms.append(r)

        n_conv = len(trial_rms)
        conv_rate[i, j] = n_conv / N_TRIALS

        if n_conv > 0:
            rms_mean[i, j] = np.mean(trial_rms)
            rms_std[i, j]  = np.std(trial_rms)

    elapsed   = time.time() - t0
    rate      = (i + 1) / elapsed if elapsed > 0 else 1
    remaining = (Ndz - i - 1) / rate
    print(f"  dz {i+1:2d}/{Ndz}  ({dz_mm:6.1f} mm)  "
          f"elapsed {elapsed/60:.1f} min  remaining {remaining/60:.1f} min")

print(f"\nFinished in {(time.time()-t0)/60:.2f} min")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

valid = rms_mean[np.isfinite(rms_mean)]
print(f"\n{'─'*40}")
print(f"Valid grid points: {len(valid)} / {Ndz*Np}")
if len(valid):
    print(f"Residual RMS  mean: {np.mean(valid):.2f} nm")
    print(f"              std : {np.std(valid):.2f} nm")
    print(f"              range: {np.min(valid):.2f} – {np.max(valid):.2f} nm")
print(f"{'─'*40}\n")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

np.savez(
    SAVE_LABEL,
    dz_grid_mm      = DZ_GRID_MM,
    param_grid      = param_grid,
    param_name      = param_name,
    rms_trials      = rms_trials,
    rms_mean        = rms_mean,
    rms_std         = rms_std,
    conv_rate       = conv_rate,
    N_trials        = N_TRIALS,
    N_iter          = N_ITER,
    if_noise        = IF_NOISE,
    sigma_e         = SIGMA_E,
    aberration_type = ABERRATION_TYPE,
    sinusoid_v0     = SINUSOID_V0_FIXED,
)
print(f"Saved: {SAVE_LABEL}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

extent = [param_grid[0], param_grid[-1], DZ_GRID_MM[0], DZ_GRID_MM[-1]]

def heatmap(data, title, cbar_label, cmap='magma_r', vmin=None, vmax=None):
    finite = data[np.isfinite(data)]
    if vmin is None and len(finite):
        vmin, vmax = np.nanpercentile(finite, [5, 95])
    plt.figure(figsize=(7, 5))
    plt.imshow(data, origin='lower', aspect='auto', extent=extent,
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=cbar_label)
    plt.xlabel(param_label)
    plt.ylabel("dz [mm]")
    plt.title(title)
    plt.tight_layout()
    plt.show()

heatmap(rms_mean,  "Mean Residual RMS  (truth − recon)  [±dz diversity]", "RMS [nm]")
heatmap(rms_std,   "Std of Residual RMS across trials",                    "std [nm]")
heatmap(conv_rate, "Convergence Rate",
        "fraction", cmap='RdYlGn', vmin=0, vmax=1)

# Row slices at min / median / max dz
fig, ax = plt.subplots(figsize=(7, 4))
for idx in [0, Ndz // 2, Ndz - 1]:
    row = rms_mean[idx]
    if np.any(np.isfinite(row)):
        ax.plot(param_grid, row, label=f"dz = {DZ_GRID_MM[idx]:.1f} mm")
ax.set_xlabel(param_label)
ax.set_ylabel("Mean residual RMS [nm]")
ax.set_title(f"RMS vs {param_name}  at selected dz values  [±dz diversity]")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
