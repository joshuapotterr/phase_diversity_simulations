"""
fdpr_injectable.py
==================
Based on FDPRno_phase_error.py, extended to support:
  - Injected phase aberrations (Zernike or sinusoidal)
  - Read noise
  - Configurable number of defocus positions

Unit convention (enforced throughout):
  - Lengths passed to HCIPy:      meters
  - Lengths passed to FDPR lib:   micrometers  (wavelength, image_dx, defocus)
  - Defocus config values:        millimeters  (human-readable; converted at use-site)
  - Phase:                        radians
  - Reported errors:              nm

PSF generation uses .power.shaped with wf.total_power = num_photons so that
the pixel values are in photons and the amplitude constraint in FDPR is
physically meaningful.
"""

import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
from skimage.transform import resize
from image_sharpening import FocusDiversePhaseRetrieval, mft_rev, InstrumentConfiguration

#  
# CONFIGURATION FLAGS
#  
#
# To replicate FDPRno_phase_error.py exactly, use:
#   INJECT_ABERRATION = False
#   IF_NOISE          = False
#   DEFOCUS_MM_LIST   = np.array([-12.0, -8.0, -4.0, 4.0, 8.0, 12.0])
#   N_ITER            = 200
#
#  

#   Aberration injection  
INJECT_ABERRATION = True        # False → noise-floor / baseline run
ABERRATION_TYPE   = "zernike"   # "zernike"  or  "sinusoid"

# Zernike option: index into make_zernike_basis (0-based; 3=defocus, 6=coma …)
ZERNIKE_MODE_INDEX    = 6       # 3 for def, 6 for coma, todo: sin waves for grid, low/mid/high zernikes
ZERNIKE_AMPLITUDE_WV  = 0.5     # peak amplitude in waves

# Sinusoid option: single-frequency along x
SINUSOID_CYCLES_PER_AP = 5.0    # cycles per aperture diameter
SINUSOID_AMPLITUDE_WV  = 0.5    # peak amplitude in waves

#   Noise  
IF_NOISE  = True
SIGMA_E   = 11          # read noise  [e⁻/pixel]
SEED      = 42

#   Defocus positions (mm) — the focused image is always added automatically  
# Include both positive and negative values to break degeneracies
DEFOCUS_MM_LIST = np.array([-12.0, -8.0, -4.0, 4.0, 8.0, 12.0])

#   FDPR solver  
N_ITER = 200

#  
# SYSTEM PARAMETERS
#  

NUM_PHOTONS = 1e6
RNG = np.random.default_rng(SEED)

seal_parameters = {
    'image_dx':             2.0071,     # pixel scale  [µm]  — used by FDPR
    'focal_length_meters':  500e-3,     # [m]
    'wavelength_meter':     650e-9,     # [m]
    'pupil_size':           10.12e-3,   # entrance pupil diameter  [m]
    'pupil_pixel_dimension': 256,
    'q':                    16,
    'Num_airycircles':      16,
}

# InstrumentConfiguration expects mixed units as the original code did:
#   efl in mm, wavelength in µm, pupil_size in mm, image_dx in µm
seal_param_config = {
    'image_dx':   seal_parameters['image_dx'],                          # µm
    'efl':        seal_parameters['focal_length_meters'] * 1e3,         # mm
    'wavelength': seal_parameters['wavelength_meter']    * 1e6,         # µm
    'pupil_size': seal_parameters['pupil_size']          * 1e3,         # mm
}

WAVELENGTH_UM = seal_param_config['wavelength']     # shorthand for FDPR calls
IMAGE_DX_UM   = seal_param_config['image_dx']       # shorthand for FDPR calls

#  
# BUILD SIMULATION ELEMENTS
#  

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
    pupil_grid,
    focal_grid,
    seal_parameters['focal_length_meters']
)

zernike_modes   = make_zernike_basis(
    num_modes=256,
    D=seal_parameters['pupil_size'],
    grid=pupil_grid
)
defocus_template = zernike_modes[3]   # Z4 — used to build defocus phase

#  
# HELPER FUNCTIONS
#  

def delta_to_p(delta_m, f_m, D_m):
    """Convert mechanical defocus distance [m] to P2V OPD error [m]."""
    return -delta_m / (8.0 * (f_m / D_m) ** 2)


def calculate_defocus_phase(defocus_mm):
    """
    Build the defocus phase screen [rad] on pupil_grid for a given
    mechanical defocus distance.

    Matches FDPRno_phase_error.py and fdpr_vs_AS_save.py exactly:
        dz_mm → dz_m → OPD_m (via delta_to_p) → phase_rad (× 2π/λ)

    The same dz_mm value converted to µm is what gets passed to
    FocusDiversePhaseRetrieval, which uses it as a physical angular-spectrum
    propagation distance.  Both models represent the same defocus.

    Parameters
       -
    defocus_mm : float
        Defocus distance [mm].  Positive = camera moved away from focus.

    Returns
      -
    defocus_phase : hcipy.Field
        Phase in radians on pupil_grid.
    """
    mask = pupil_mask
    template_s   = defocus_template.shaped
    template_p2v = template_s[mask].max() - template_s[mask].min()
    unit_defocus  = defocus_template / template_p2v    # normalised to 1 rad P2V

    dz_m          = defocus_mm * 1e-3                  # mm → m
    defocus_p2v_m = delta_to_p(                        # OPD in meters
        delta_m=dz_m,
        f_m=seal_parameters['focal_length_meters'],
        D_m=seal_parameters['pupil_size'],
    )
    # OPD [m] → phase [rad]:  φ = OPD × 2π / λ
    phase_p2v_rad = defocus_p2v_m * (2.0 * np.pi / seal_parameters['wavelength_meter'])
    defocus_phase = unit_defocus * phase_p2v_rad
    return defocus_phase


def make_sinusoidal_phase(cycles_per_aperture, amplitude_waves):
    """
    Single-frequency sinusoid along x [rad] on pupil_grid.

    Parameters
       -
    cycles_per_aperture : float
    amplitude_waves     : float  — peak amplitude in waves

    Returns
      -
    hcipy.Field  [rad]
    """
    x = pupil_grid.x
    D = seal_parameters['pupil_size']
    phase_waves = amplitude_waves * np.sin(2.0 * np.pi * cycles_per_aperture * (x / D))
    return Field(phase_waves * 2.0 * np.pi, pupil_grid)   # waves → rad


def psf_from_wavefront(wf):
    """
    Propagate wavefront to focal plane and return PSF [photons/pixel].

    Uses .power.shaped (not .intensity) so that setting wf.total_power
    correctly distributes photons across pixels.
    """
    return np.asarray(prop_p2f(wf).power.shaped, dtype=float)


def add_read_noise(psf, sigma_e):
    """Add Gaussian read noise [e⁻/pixel] and clip to ≥ 0."""
    noisy = psf + RNG.normal(scale=sigma_e, size=psf.shape)
    return np.clip(noisy, 0.0, None)


def make_wavefront(phase_rad):
    """
    Build a Wavefront on the pupil plane with the given phase [rad],
    normalised to NUM_PHOTONS total power.
    """
    wf = Wavefront(
        telescope_pupil * np.exp(1j * phase_rad),
        seal_parameters['wavelength_meter']
    )
    wf.total_power = NUM_PHOTONS
    return wf

#  
# BUILD ABERRATION PHASE
#  

if INJECT_ABERRATION:
    if ABERRATION_TYPE == "zernike":
        mode   = zernike_modes[ZERNIKE_MODE_INDEX]
        mask   = pupil_mask
        mode_s = mode.shaped
        p2v    = mode_s[mask].max() - mode_s[mask].min()
        # Normalise to unit P2V, then scale to desired amplitude
        phi_aberration = (mode / p2v) * (ZERNIKE_AMPLITUDE_WV * 2.0 * np.pi)
        aberration_label = (f"Zernike mode {ZERNIKE_MODE_INDEX}, "
                            f"A={ZERNIKE_AMPLITUDE_WV} waves")

    elif ABERRATION_TYPE == "sinusoid":
        phi_aberration = make_sinusoidal_phase(
            SINUSOID_CYCLES_PER_AP, SINUSOID_AMPLITUDE_WV
        )
        aberration_label = (f"Sinusoid {SINUSOID_CYCLES_PER_AP} cyc/ap, "
                            f"A={SINUSOID_AMPLITUDE_WV} waves")

    else:
        raise ValueError(f"Unknown ABERRATION_TYPE: {ABERRATION_TYPE!r}")

    print(f"Injected aberration: {aberration_label}")

    #    Plot injected phase immediately so it's visible before FDPR runs   
    phi_ab_shaped  = phi_aberration.shaped * telescope_pupil.shaped
    phi_ab_masked  = phi_ab_shaped[pupil_mask]
    ab_rms_nm      = np.sqrt(np.mean(phi_ab_masked**2)) * seal_parameters['wavelength_meter'] / (2*np.pi) * 1e9
    ab_p2v_nm      = (phi_ab_masked.max() - phi_ab_masked.min()) * seal_parameters['wavelength_meter'] / (2*np.pi) * 1e9

    # Focused PSF of the aberrated pupil (clean, for reference)
    wf_ab_preview = make_wavefront(phi_aberration)
    psf_ab_preview = psf_from_wavefront(wf_ab_preview)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(phi_ab_shaped, cmap='RdBu_r')
    axes[0].set_title(f'Injected aberration [rad]\n{aberration_label}\nRMS={ab_rms_nm:.1f} nm  P2V={ab_p2v_nm:.1f} nm')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(np.log10(psf_ab_preview / psf_ab_preview.max() + 1e-10), vmin=-5, cmap='inferno')
    axes[1].set_title('Focused PSF of injected aberration\n(clean, log₁₀ normalised)')
    axes[1].axis('off')

    fig.suptitle('Injected Aberration — Ground Truth', fontsize=13)
    plt.tight_layout()
    plt.show()

else:
    # Flat phase — noise-floor baseline (replicates FDPRno_phase_error.py)
    phi_aberration = Field(np.zeros(pupil_grid.size), pupil_grid)
    aberration_label = "None (flat phase)"
    print("No aberration injected — baseline / noise floor run.")

#  
# BUILD PSF LIST
#  
# psf_list[0]  = focused (aberration only, no added defocus)
# psf_list[1:] = one per entry in DEFOCUS_MM_LIST
#
# FDPR API:  FocusDiversePhaseRetrieval(psf_list, λ_µm, dx_list, defocus_list)
#   len(psf_list)    = N + 1
#   len(dx_list)     = N      [µm]
#   len(defocus_list)= N      [µm]

print("\nBuilding PSFs ...")

# Focused PSF (aberration, no defocus)
wf_focused    = make_wavefront(phi_aberration)
psf_focused   = psf_from_wavefront(wf_focused)

psf_list     = [psf_focused]
dx_list      = []
defocus_um_list = []

for dz_mm in DEFOCUS_MM_LIST:
    phi_def = calculate_defocus_phase(dz_mm)
    wf_def  = make_wavefront(phi_aberration + phi_def)
    psf_def = psf_from_wavefront(wf_def)

    psf_list.append(psf_def)
    dx_list.append(IMAGE_DX_UM)           # µm — same pixel scale for all positions
    defocus_um_list.append(dz_mm * 1e3)   # mm → µm

print(f"  PSF list length : {len(psf_list)}  (1 focused + {len(DEFOCUS_MM_LIST)} defocused)")
print(f"  PSF shape       : {psf_focused.shape}")
print(f"  Focused PSF sum : {psf_focused.sum():.3e}  (≈ {NUM_PHOTONS:.0e} photons)")
print(f"  Defocus [mm]    : {list(DEFOCUS_MM_LIST)}")

#  
# ADD NOISE (optional)
#  

if IF_NOISE:
    print(f"\nAdding read noise (σ = {SIGMA_E} e⁻/px) ...")
    psf_list_noisy = [add_read_noise(p, SIGMA_E) for p in psf_list]
else:
    psf_list_noisy = [p.copy() for p in psf_list]

# Sanity check — all PSFs finite
for k, p in enumerate(psf_list_noisy):
    if not np.all(np.isfinite(p)):
        raise RuntimeError(f"Non-finite values in psf_list_noisy[{k}]")

#  
# RUN FDPR
#  

print(f"\nRunning FocusDiversePhaseRetrieval ({N_ITER} iterations) ...")

mp = FocusDiversePhaseRetrieval(
    psf_list_noisy,
    WAVELENGTH_UM,      # µm
    dx_list,            # µm  (one per defocused image)
    defocus_um_list,    # µm  (one per defocused image)
)

for it in range(N_ITER):
    psf_rec = mp.step()
    if (it + 1) % 50 == 0:
        costs = [cf[-1] for cf in mp.cost_functions if len(cf) > 0]
        print(f"  iter {it+1:4d}  costs: {[f'{c:.3e}' for c in costs]}")

print("Done.")

#  
# PUPIL READOUT
#  
# mft_rev does a matrix Fourier transform from the FDPR focal-plane output
# back to the pupil plane.  Resize to match the 256×256 pupil grid.

raw_pupil_phase = np.angle(mft_rev(psf_rec, conf))
pupil_phase_rec = (
    resize(raw_pupil_phase, (256, 256), preserve_range=True)
    * telescope_pupil.shaped
)

#  
# ERROR METRICS
#  

rad_to_nm = seal_parameters['wavelength_meter'] / (2.0 * np.pi) * 1e9

# Piston removal via median subtraction (inside pupil)
rec_masked  = pupil_phase_rec[pupil_mask].copy()
rec_masked -= np.median(rec_masked)

if INJECT_ABERRATION:
    truth_shaped = phi_aberration.shaped
    truth_resize = resize(truth_shaped, (256, 256), preserve_range=True)
    truth_masked = truth_resize[pupil_mask]
    truth_masked_ms = truth_masked - np.median(truth_masked)

    residual       = truth_masked_ms - rec_masked
    rms_residual   = np.sqrt(np.nanmean(residual ** 2))
    rms_residual_nm = rms_residual * rad_to_nm

    rms_truth      = np.sqrt(np.nanmean(truth_masked_ms ** 2))
    rms_truth_nm   = rms_truth * rad_to_nm
else:
    # Baseline run: truth = zero, so residual = reconstruction itself
    residual        = rec_masked
    rms_residual_nm = np.sqrt(np.nanmean(residual ** 2)) * rad_to_nm
    truth_resize    = np.zeros((256, 256))
    rms_truth_nm    = 0.0

rms_total      = np.sqrt(np.nanmean(rec_masked ** 2))
rms_total_nm   = rms_total * rad_to_nm
p2v_rec_nm     = (rec_masked.max() - rec_masked.min()) * rad_to_nm

print(f"\n{'─'*40}")
print(f"Aberration  : {aberration_label}")
print(f"Noise       : {'ON  σ=' + str(SIGMA_E) + ' e⁻/px' if IF_NOISE else 'OFF'}")
print(f"Defocus pts : {len(DEFOCUS_MM_LIST)}  ({list(DEFOCUS_MM_LIST)} mm)")
if INJECT_ABERRATION:
    print(f"Truth RMS   : {rms_truth_nm:.2f} nm")
print(f"Recon RMS   : {rms_total_nm:.2f} nm")
print(f"Recon P2V   : {p2v_rec_nm:.2f} nm")
print(f"Residual RMS: {rms_residual_nm:.2f} nm")
print(f"{'─'*40}\n")

#  
# PLOTS
#  

#    1. Cost function convergence   
plt.figure(figsize=(7, 4))
for i, cf in enumerate(mp.cost_functions):
    if len(cf) > 0:
        dz = DEFOCUS_MM_LIST[i] if i < len(DEFOCUS_MM_LIST) else i
        plt.semilogy(cf, label=f'dz = {dz:.1f} mm')
plt.xlabel('Iteration')
plt.ylabel('MSE cost')
plt.title('FDPR Cost Function Convergence')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#    2. PSF gallery (clean)     
n_psf  = len(psf_list)
fig, axes = plt.subplots(1, n_psf, figsize=(3 * n_psf, 3.5))
if n_psf == 1:
    axes = [axes]
labels = ['Focused (dz=0)'] + [f'dz={dz:.1f} mm' for dz in DEFOCUS_MM_LIST]
for ax, psf, lbl in zip(axes, psf_list, labels):
    ax.imshow(np.log10(psf / psf.max() + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title(lbl, fontsize=9)
    ax.axis('off')
fig.suptitle('Clean PSFs (log₁₀, normalised)', fontsize=12)
plt.tight_layout()
plt.show()

#    3. Reconstruction panel       ─
rec_med_sub = resize(
    pupil_phase_rec - np.median(pupil_phase_rec[pupil_mask]),
    (256, 256), preserve_range=True
)
diff_image = truth_resize - rec_med_sub

if INJECT_ABERRATION:
    # Shared colour scale for truth / reconstruction so they're directly comparable
    phase_vmax = np.max(np.abs(truth_resize[pupil_mask]))
    phase_vmin = -phase_vmax

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(np.log10(np.abs(psf_rec) + 1e-10), cmap='inferno', vmin=-5)
    axes[0].set_title('FDPR output PSF (log₁₀)')
    axes[0].axis('off')

    im1 = axes[1].imshow(truth_resize * telescope_pupil.shaped,
                         cmap='RdBu_r', vmin=phase_vmin, vmax=phase_vmax)
    axes[1].set_title(f'Truth [rad]\n{aberration_label}\nRMS={rms_truth_nm:.1f} nm')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(rec_med_sub * telescope_pupil.shaped,
                         cmap='RdBu_r', vmin=phase_vmin, vmax=phase_vmax)
    axes[2].set_title(f'Reconstruction [rad]\nRMS={rms_total_nm:.1f} nm')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(diff_image * telescope_pupil.shaped, cmap='RdBu_r')
    axes[3].set_title(f'Residual (truth − recon)\nRMS={rms_residual_nm:.1f} nm')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

else:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(np.log10(np.abs(psf_rec) + 1e-10), cmap='inferno', vmin=-5)
    axes[0].set_title('FDPR output PSF (log₁₀)')
    axes[0].axis('off')

    im1 = axes[1].imshow(rec_med_sub * telescope_pupil.shaped, cmap='RdBu_r')
    axes[1].set_title(f'Reconstruction [rad]\n(baseline — truth = 0)\nRMS={rms_total_nm:.1f} nm')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

fig.suptitle('FDPR Phase Retrieval Results', fontsize=13)
plt.tight_layout()
plt.show()
