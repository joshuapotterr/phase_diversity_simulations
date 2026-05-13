# OTF-based analysis of sinusoidal aberrations
# Refined version with subtle improvements

import numpy as np
from hcipy import *
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

verbose = True
save_label = "OTF_heatmap_data_new_block.npz"
dzs_otf = np.linspace(5, 250, 80)      # mm
v0s_otf = np.linspace(0.5, 80, 150)    # cycles/aperture


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def mm_to_m(x_mm: float) -> float:
    """Convert millimeters to meters."""
    return x_mm * 1e-3


def delta_to_p(delta: float, f: float, D: float) -> float:
    """
    Convert mechanical defocus distance to peak-to-valley wavefront error.
    
    Parameters
    ----------
    delta : float
        Mechanical defocus distance (same units as f and D)
    f : float
        Focal length
    D : float
        Pupil diameter
    
    Returns
    -------
    float
        Peak-to-valley wavefront error (same units as delta)
    """
    return -delta / (8 * (f / D) ** 2)


def make_sinusoidal_phase_waves(pupil_grid, pupil_diameter_m: float, 
                                 cycles_per_aperture: float, m_waves: float) -> Field:
    """
    Create single-frequency sinusoidal phase aberration along x-axis.
    
    Parameters
    ----------
    pupil_grid : Grid
        HCIPy pupil grid
    pupil_diameter_m : float
        Pupil diameter in meters
    cycles_per_aperture : float
        Spatial frequency in cycles per aperture diameter
    m_waves : float
        Peak amplitude in waves
    
    Returns
    -------
    Field
        Phase aberration in waves
    """
    x = pupil_grid.x
    phase_waves = m_waves * np.sin(2 * np.pi * cycles_per_aperture * (x / pupil_diameter_m))
    return Field(phase_waves, pupil_grid)


def psf_from_wavefront(wf, propagator) -> np.ndarray:
    """Propagate wavefront to focal plane and return PSF intensity."""
    return np.asarray(propagator(wf).intensity.shaped)


def calculate_defocus_phase(seal_parameters: dict, defocus_distance_mm: float, 
                            telescope_pupil, defocus_template) -> Field:
    """
    Calculate defocus phase from mechanical defocus distance.
    
    Parameters
    ----------
    seal_parameters : dict
        Optical system parameters
    defocus_distance_mm : float
        Defocus distance in millimeters
    telescope_pupil : Field
        Pupil aperture function
    defocus_template : Field
        Unit defocus Zernike mode
    
    Returns
    -------
    Field
        Defocus phase in radians
    """
    # Create boolean mask once (avoid repeated creation)
    mask = np.asarray(telescope_pupil.shaped, dtype=bool)
    defocus_template_s = defocus_template.shaped
    
    # Normalize to unit P-V
    template_p2v = defocus_template_s[mask].ptp()  # More concise than max - min
    unit_defocus = defocus_template / template_p2v
    
    # Convert mm to wavefront error
    dz_m = mm_to_m(defocus_distance_mm)
    defocus_p2v = delta_to_p(
        delta=dz_m,
        f=seal_parameters['focal_length_meters'],
        D=seal_parameters['pupil_size']
    )
    
    # Convert to phase (radians)
    phase_p2v = defocus_p2v * (2 * np.pi / seal_parameters['wavelength_meter'])
    return unit_defocus * phase_p2v


def otf_from_psf(psf: np.ndarray) -> np.ndarray:
    """
    Compute OTF magnitude from PSF via FFT.
    
    The OTF is the Fourier transform of the PSF, normalized so that
    OTF(0,0) = 1 for a perfect system.
    """
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    return np.abs(otf)


def find_otf_sidepeaks_1D(otf: np.ndarray, kill_core_pix: int = 9, 
                          subpixel: bool = True) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    """
    Find the two symmetric side peaks in the OTF from a 1D sinusoidal aberration.
    
    Searches along the central horizontal row (y=0), ignoring the DC core.
    
    Parameters
    ----------
    otf : np.ndarray
        2D OTF magnitude array
    kill_core_pix : int
        Half-width of DC region to ignore
    subpixel : bool
        Enable quadratic subpixel refinement
    
    Returns
    -------
    offset : float
        Average pixel distance of side-peaks from DC
    amplitude : float
        Average peak amplitude
    offs_lr : tuple
        (left offset, right offset) - negative for left
    amps_lr : tuple
        (left amplitude, right amplitude)
    """
    row = otf[otf.shape[0] // 2, :].copy()
    center = len(row) // 2
    
    # Zero out DC core
    row[center - kill_core_pix : center + kill_core_pix + 1] = 0.0
    
    # Split into left and right halves
    left, right = row[:center], row[center + 1:]
    
    # Find peak indices
    idx_left = int(np.argmax(left))
    idx_right = int(np.argmax(right))
    amp_left = float(left[idx_left])
    amp_right = float(right[idx_right])
    
    def refine_peak(idx: int, arr: np.ndarray) -> float:
        """Quadratic subpixel refinement around peak."""
        if not subpixel or idx <= 0 or idx >= len(arr) - 1:
            return float(idx)
        y0, y1, y2 = arr[idx - 1], arr[idx], arr[idx + 1]
        denom = y0 - 2 * y1 + y2
        if denom == 0:
            return float(idx)
        return idx + 0.5 * (y0 - y2) / denom
    
    # Compute refined positions
    pos_left = refine_peak(idx_left, left)
    pos_right = refine_peak(idx_right, right) + center + 1
    
    # Convert to offsets from center
    off_left = pos_left - center   # Negative
    off_right = pos_right - center  # Positive
    
    # Average magnitude
    offset = 0.5 * (abs(off_left) + off_right)
    amplitude = 0.5 * (amp_left + amp_right)
    
    return offset, amplitude, (off_left, off_right), (amp_left, amp_right)


def plot_otf_analysis(psf: np.ndarray, otf: np.ndarray, v0: float, dz_mm: float,
                      title_prefix: str = "") -> Tuple[float, float]:
    """
    Plot PSF and its OTF with side-peak analysis.
    
    Returns
    -------
    offset : float
        Side-peak offset in pixels
    amplitude : float
        Side-peak amplitude
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # PSF (log scale)
    ax = axes[0]
    psf_log = np.log10(psf + 1e-10)
    im = ax.imshow(psf_log, vmin=-5, cmap='inferno')
    ax.set_title(f'{title_prefix}PSF (log10)\ndz={dz_mm:.1f}mm, v0={v0:.2f}')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.axis('off')
    
    # OTF magnitude
    ax = axes[1]
    im = ax.imshow(otf, cmap='viridis')
    ax.set_title(f'{title_prefix}OTF Magnitude')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.axis('off')
    
    # OTF central row with peak markers
    ax = axes[2]
    row = otf[otf.shape[0] // 2, :]
    ax.plot(row, 'b-', linewidth=0.8)
    ax.set_xlabel('Pixel')
    ax.set_ylabel('OTF Amplitude')
    ax.set_title('OTF Central Row')
    ax.grid(True, alpha=0.3)
    
    # Mark side peaks
    offset, amp, offs_lr, amps_lr = find_otf_sidepeaks_1D(otf, kill_core_pix=9)
    center = len(row) // 2
    ax.axvline(center + offs_lr[0], color='r', linestyle='--', linewidth=1.5,
               label=f'Left: {amps_lr[0]:.0f}')
    ax.axvline(center + offs_lr[1], color='g', linestyle='--', linewidth=1.5,
               label=f'Right: {amps_lr[1]:.0f}')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return offset, amp


# =============================================================================
# OPTICAL SYSTEM PARAMETERS
# =============================================================================

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

# Build simulation elements
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

# Zernike basis for defocus
zernike_modes = make_zernike_basis(
    num_modes=256,
    D=seal_parameters['pupil_size'],
    grid=pupil_grid
)
defocus_template = zernike_modes[3]


# =============================================================================
# OTF GRID SETUP
# =============================================================================

m_waves = 0.10  # Peak amplitude of sinusoid in waves
D_m = seal_parameters['pupil_size']
lam_m = seal_parameters['wavelength_meter']

Nd, Nv = len(dzs_otf), len(v0s_otf)

print(f"\n{'=' * 50}")
print("OTF Analysis Configuration:")
print(f"  Grid size: {Nd} x {Nv} = {Nd * Nv} points")
print(f"  dz range: {dzs_otf.min():.1f} - {dzs_otf.max():.1f} mm")
print(f"  v0 range: {v0s_otf.min():.1f} - {v0s_otf.max():.1f} cycles/ap")
print(f"  m_waves: {m_waves}")
print(f"{'=' * 50}\n")


# =============================================================================
# STORAGE ARRAYS
# =============================================================================

# OTF amplitudes
otf_amp_focus = np.full((Nd, Nv), np.nan)
otf_amp_defoc = np.full((Nd, Nv), np.nan)

# OTF peak offsets (in pixels)
otf_offset_focus = np.full((Nd, Nv), np.nan)
otf_offset_defoc = np.full((Nd, Nv), np.nan)


# =============================================================================
# PRECOMPUTE INVARIANTS (performance improvement)
# =============================================================================

# Precompute all defocus phases (saves repeated calculation)
print("Precomputing defocus phases...")
defocus_phases = {}
for dz in dzs_otf:
    defocus_phases[dz] = calculate_defocus_phase(
        seal_parameters, float(dz), telescope_pupil, defocus_template
    )

# Precompute all sinusoidal aberrations
print("Precomputing sinusoidal aberrations...")
sine_phases = {}
for v0 in v0s_otf:
    if v0 == 0:
        continue
    phi_waves = make_sinusoidal_phase_waves(pupil_grid, D_m, float(v0), m_waves)
    sine_phases[v0] = 2 * np.pi * phi_waves  # Convert to radians


# =============================================================================
# OTF CALCULATION LOOP
# =============================================================================

print("Computing OTF heatmap...")

# Precompute pupil array for efficiency
pupil_array = np.asarray(telescope_pupil)

for i, dz in enumerate(dzs_otf):
    dz_mm = float(dz)
    phi_def = defocus_phases[dz]
    
    for j, v0 in enumerate(v0s_otf):
        if v0 == 0:
            continue
        
        phi_sine_rad = sine_phases[v0]
        
        # Focused PSF (no defocus)
        wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), lam_m)
        psf_focus = psf_from_wavefront(wf_focus, prop_p2f)
        
        # Defocused PSF
        wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
        psf_defoc = psf_from_wavefront(wf_defoc, prop_p2f)
        
        # Compute OTFs and find peaks
        try:
            otf_focus = otf_from_psf(psf_focus)
            offset, amp, _, _ = find_otf_sidepeaks_1D(otf_focus, kill_core_pix=9)
            otf_amp_focus[i, j] = amp
            otf_offset_focus[i, j] = offset
            
            otf_defoc = otf_from_psf(psf_defoc)
            offset, amp, _, _ = find_otf_sidepeaks_1D(otf_defoc, kill_core_pix=9)
            otf_amp_defoc[i, j] = amp
            otf_offset_defoc[i, j] = offset
            
            # Verbose plotting for selected points
            if verbose and i % 20 == 0 and j % 30 == 0:
                plot_otf_analysis(psf_defoc, otf_defoc, v0, dz_mm, "Defocused ")
        
        except Exception as e:
            print(f"OTF error at dz={dz:.1f}, v0={v0:.1f}: {e}")
            continue
    
    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1}/{Nd} defocus values completed")

print("OTF calculation complete.")


# =============================================================================
# SAVE RESULTS
# =============================================================================

np.savez(
    save_label,
    # Primary data
    dzs_otf=dzs_otf,
    v0s_otf=v0s_otf,
    otf_amp_focus=otf_amp_focus,
    otf_amp_defoc=otf_amp_defoc,
    otf_offset_focus=otf_offset_focus,
    otf_offset_defoc=otf_offset_defoc,
    # Backwards compatibility
    H=otf_amp_defoc,
    fixed_dz_heatmap=dzs_otf,
    v0_heatmap=v0s_otf,
    # Metadata
    m_waves=m_waves,
    wavelength_m=lam_m,
    pupil_diameter_m=D_m
)
print(f"Saved: {save_label}")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_heatmap(data: np.ndarray, extent: list, title: str, 
                 cmap: str = 'viridis', vmin: Optional[float] = None,
                 vmax: Optional[float] = None, cbar_label: str = "") -> None:
    """Helper function for consistent heatmap plotting."""
    plt.figure(figsize=(10, 8))
    
    finite_vals = data[np.isfinite(data)]
    if len(finite_vals) > 0 and vmin is None:
        vmin, vmax = np.nanpercentile(finite_vals, [2, 98])
    
    plt.imshow(data, origin='lower', aspect='auto', extent=extent,
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=cbar_label)
    plt.xlabel("v0 [cycles/ap]")
    plt.ylabel("dz [mm]")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Define common extent
extent_otf = [v0s_otf.min(), v0s_otf.max(), dzs_otf.min(), dzs_otf.max()]

# Defocused OTF amplitude heatmap
plot_heatmap(
    otf_amp_defoc, extent_otf, 
    "OTF Heatmap (Defocused PSF)",
    cbar_label="OTF side-peak amplitude"
)

# Focused OTF amplitude heatmap
plot_heatmap(
    otf_amp_focus, extent_otf,
    "OTF Heatmap (Focused PSF)",
    cbar_label="OTF side-peak amplitude"
)

# Ratio: defocused / focused
plt.figure(figsize=(10, 8))
ratio = otf_amp_defoc / otf_amp_focus
plt.imshow(ratio, origin='lower', aspect='auto', extent=extent_otf,
           cmap='RdBu_r', vmin=0, vmax=2)
plt.colorbar(label="OTF ratio (defoc/focus)")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("OTF Transfer Ratio (Defocused / Focused)")
plt.tight_layout()
plt.show()

# Slice plots at fixed defocus values
dz_slices = [25, 50, 100, 150, 200]
plt.figure(figsize=(10, 6))
for dz_target in dz_slices:
    idx = np.argmin(np.abs(dzs_otf - dz_target))
    plt.plot(v0s_otf, otf_amp_defoc[idx, :], label=f'dz={dzs_otf[idx]:.0f} mm')
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("OTF side-peak amplitude")
plt.title("OTF Amplitude vs Spatial Frequency (at fixed defocus)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Slice plots at fixed spatial frequencies
v0_slices = [5, 10, 20, 40, 60]
plt.figure(figsize=(10, 6))
for v0_target in v0_slices:
    idx = np.argmin(np.abs(v0s_otf - v0_target))
    plt.plot(dzs_otf, otf_amp_defoc[:, idx], label=f'v0={v0s_otf[idx]:.1f} cyc/ap')
plt.xlabel("dz [mm]")
plt.ylabel("OTF side-peak amplitude")
plt.title("OTF Amplitude vs Defocus (at fixed spatial frequency)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nOTF analysis complete!")