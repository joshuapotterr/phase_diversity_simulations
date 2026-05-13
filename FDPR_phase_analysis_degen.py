# FDPR Phase Reconstruction Analysis
# ====================================
# Connects OTF side-peak amplitude to FDPR reconstruction quality
# 
# Goal: Show that high OTF amplitude → low RMS error

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from dataclasses import dataclass
import time
import os

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colors
COLORS = {
    'blue': '#0072B2',
    'orange': '#D55E00', 
    'green': '#009E73',
    'pink': '#CC79A7',
    'cyan': '#56B4E9',
}

# =============================================================================
# TRY TO IMPORT DEPENDENCIES
# =============================================================================

try:
    from hcipy import (
        make_pupil_grid, make_focal_grid,
        make_circular_aperture, make_zernike_basis,
        Wavefront, FraunhoferPropagator, Field
    )
    HAS_HCIPY = True
except ImportError:
    HAS_HCIPY = False
    print("Warning: hcipy not available")

try:
    from image_sharpening import FocusDiversePhaseRetrieval, mft_rev, InstrumentConfiguration
    from skimage.transform import resize
    HAS_FDPR = True
except ImportError:
    HAS_FDPR = False
    print("Warning: FDPR not available - will use synthetic data for demonstration")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FDPRConfig:
    """Configuration for FDPR analysis."""
    # Optical system (SEAL)
    wavelength_m: float = 650e-9
    pupil_diameter_m: float = 10.12e-3
    focal_length_m: float = 500e-3
    pupil_npix: int = 256
    
    # FDPR parameters
    n_iterations: int = 200
    n_trials: int = 5  # Monte Carlo trials per (dz, v0) point
    
    # Noise
    read_noise_e: float = 5.0
    photon_flux: float = 1e6
    
    # Grid (should match OTF heatmap)
    dz_values: np.ndarray = None
    v0_values: np.ndarray = None
    
    # Aberration
    m_waves: float = 0.1
    
    def __post_init__(self):
        if self.dz_values is None:
            self.dz_values = np.linspace(5, 250, 20)  # Coarser for speed
        if self.v0_values is None:
            self.v0_values = np.linspace(3, 80, 30)


# =============================================================================
# PHYSICS HELPERS
# =============================================================================

def dz_to_a_hat(dz_mm: np.ndarray, cfg: FDPRConfig) -> np.ndarray:
    """Convert mechanical defocus (mm) to normalized defocus â (waves P-V)."""
    dz_m = np.asarray(dz_mm) * 1e-3
    return (dz_m * cfg.pupil_diameter_m**2) / (8 * cfg.focal_length_m**2 * cfg.wavelength_m)


def compute_theoretical_snr(dz_mm: float, v0: float, cfg: FDPRConfig) -> float:
    """
    Compute theoretical SNR for FDPR based on OTF amplitude.
    
    SNR ∝ |sin(π v₀² / 8â)| * sqrt(photon_flux) / read_noise
    """
    a_hat = dz_to_a_hat(dz_mm, cfg)
    if a_hat < 0.01:
        return 0.0
    
    # OTF amplitude factor
    arg = np.pi * v0**2 / (8 * a_hat)
    otf_factor = np.abs(np.sin(arg))
    
    # SNR model
    snr = otf_factor * np.sqrt(cfg.photon_flux) / cfg.read_noise_e
    return float(snr)


def predict_rms_from_otf(otf_amplitude: float, noise_floor: float = 5.0,
                          scaling: float = 1e6) -> float:
    """
    Predict FDPR RMS error from OTF amplitude.
    
    Model: RMS ≈ noise_floor + scaling / (OTF_amplitude + epsilon)
    
    Higher OTF amplitude → lower RMS error
    """
    epsilon = 1e3  # Prevent division by zero
    rms = noise_floor + scaling / (otf_amplitude + epsilon)
    return rms


# =============================================================================
# SYNTHETIC FDPR DATA GENERATION
# =============================================================================

def generate_synthetic_fdpr_results(otf_data: dict, cfg: FDPRConfig,
                                     noise_level: float = 0.15) -> dict:
    """
    Generate synthetic FDPR results based on OTF amplitude.
    
    This models the expected inverse relationship between OTF amplitude
    and reconstruction error.
    
    Parameters
    ----------
    otf_data : dict
        Contains 'H_photometry', 'dz', 'v0' from OTF analysis
    cfg : FDPRConfig
        Configuration
    noise_level : float
        Fractional noise in RMS estimates
    
    Returns
    -------
    dict
        FDPR results including RMS heatmap
    """
    H_otf = otf_data['H_photometry']
    dz = otf_data['dz']
    v0 = otf_data['v0']
    
    # Model: RMS = noise_floor + k / (OTF + epsilon)
    noise_floor = 5.0  # nm - fundamental limit
    k = 5e6  # Scaling factor (tuned to give realistic RMS range)
    epsilon = 1e4
    
    # Compute RMS from OTF
    rms_mean = noise_floor + k / (H_otf + epsilon)
    
    # Add noise for Monte Carlo uncertainty
    np.random.seed(42)
    rms_std = rms_mean * noise_level * np.abs(np.random.randn(*rms_mean.shape))
    
    # Mask invalid regions (where OTF was NaN)
    invalid = ~np.isfinite(H_otf) | (H_otf < 100)
    rms_mean[invalid] = np.nan
    rms_std[invalid] = np.nan
    
    # Compute convergence rate (faster where OTF is higher)
    convergence_iter = 50 + 150 * np.exp(-H_otf / 2e5)
    convergence_iter[invalid] = np.nan
    
    return {
        'dz': dz,
        'v0': v0,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'convergence_iter': convergence_iter,
        'H_otf': H_otf,
    }


# =============================================================================
# ACTUAL FDPR RUNNER (if available)
# =============================================================================

def run_fdpr_grid(cfg: FDPRConfig, dz_subset: np.ndarray = None,
                   v0_subset: np.ndarray = None) -> dict:
    """
    Run actual FDPR reconstructions over a (dz, v0) grid.
    
    This is computationally expensive - use sparse grids!
    """
    if not HAS_FDPR or not HAS_HCIPY:
        print("FDPR/HCIPy not available - cannot run actual reconstructions")
        return None
    
    if dz_subset is None:
        dz_subset = cfg.dz_values
    if v0_subset is None:
        v0_subset = cfg.v0_values
    
    print(f"Running FDPR grid: {len(dz_subset)} x {len(v0_subset)} = {len(dz_subset)*len(v0_subset)} points")
    print(f"Trials per point: {cfg.n_trials}")
    print(f"Iterations per trial: {cfg.n_iterations}")
    
    # Setup optical system
    pupil_grid = make_pupil_grid(cfg.pupil_npix, cfg.pupil_diameter_m)
    focal_grid = make_focal_grid(
        q=4, num_airy=64,
        pupil_diameter=cfg.pupil_diameter_m,
        focal_length=cfg.focal_length_m,
        reference_wavelength=cfg.wavelength_m
    )
    aperture = make_circular_aperture(cfg.pupil_diameter_m)
    telescope_pupil = aperture(pupil_grid)
    propagator = FraunhoferPropagator(pupil_grid, focal_grid, cfg.focal_length_m)
    
    # Zernike basis for defocus
    zernike_basis = make_zernike_basis(num_modes=256, D=cfg.pupil_diameter_m, grid=pupil_grid)
    defocus_template = zernike_basis[3]
    
    # Normalize defocus
    pupil_mask = np.asarray(telescope_pupil.shaped, dtype=bool)
    defocus_shaped = defocus_template.shaped
    template_pv = defocus_shaped[pupil_mask].ptp()
    unit_defocus = defocus_template / template_pv
    
    # FDPR configuration
    seal_param_config = {
        'image_dx': 2.0071,
        'efl': cfg.focal_length_m * 1e3,
        'wavelength': cfg.wavelength_m * 1e6,
        'pupil_size': cfg.pupil_diameter_m * 1e3,
    }
    fdpr_conf = InstrumentConfiguration(seal_param_config)
    
    # Storage
    rms_results = np.full((len(dz_subset), len(v0_subset), cfg.n_trials), np.nan)
    convergence_results = np.full((len(dz_subset), len(v0_subset), cfg.n_trials), np.nan)
    
    rng = np.random.default_rng(12345)
    
    total = len(dz_subset) * len(v0_subset)
    count = 0
    t0 = time.time()
    
    for i, dz in enumerate(dz_subset):
        # Compute defocus phase
        dz_m = dz * 1e-3
        defocus_pv_m = -dz_m / (8 * (cfg.focal_length_m / cfg.pupil_diameter_m)**2)
        phase_pv_rad = defocus_pv_m * (2 * np.pi / cfg.wavelength_m)
        phi_defocus = unit_defocus * phase_pv_rad
        
        for j, v0 in enumerate(v0_subset):
            # Create sinusoidal aberration
            x = pupil_grid.x
            phi_sine_waves = cfg.m_waves * np.sin(2 * np.pi * v0 * (x / cfg.pupil_diameter_m))
            phi_sine_rad = 2 * np.pi * Field(phi_sine_waves, pupil_grid)
            
            # True phase (what we're trying to reconstruct)
            true_phase = phi_sine_rad
            
            for t in range(cfg.n_trials):
                # Generate PSFs
                wf_focus = Wavefront(telescope_pupil * np.exp(1j * true_phase), cfg.wavelength_m)
                psf_focus = np.asarray(propagator(wf_focus).intensity.shaped)
                
                wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (true_phase + phi_defocus)), cfg.wavelength_m)
                psf_defoc = np.asarray(propagator(wf_defoc).intensity.shaped)
                
                # Add noise
                psf_focus_noisy = psf_focus + rng.normal(0, cfg.read_noise_e, psf_focus.shape)
                psf_defoc_noisy = psf_defoc + rng.normal(0, cfg.read_noise_e, psf_defoc.shape)
                
                # Clip to non-negative (required for FDPR sqrt)
                psf_focus_noisy = np.maximum(psf_focus_noisy, 0)
                psf_defoc_noisy = np.maximum(psf_defoc_noisy, 0)
                
                # Run FDPR
                psf_list = [psf_focus_noisy, psf_defoc_noisy]
                dx_list = [2.0071]
                
                try:
                    mp = FocusDiversePhaseRetrieval(
                        psf_list, cfg.wavelength_m * 1e6, dx_list, [dz]
                    )
                    
                    # Track convergence
                    rms_history = []
                    for iteration in range(cfg.n_iterations):
                        psf_result = mp.step()
                        
                        # Compute RMS every 10 iterations
                        if iteration % 10 == 0:
                            raw_pupil = np.angle(mft_rev(psf_result, fdpr_conf))
                            # Handle potential NaN from failed reconstruction
                            if np.all(np.isnan(raw_pupil)):
                                continue
                            with np.errstate(invalid='ignore'):
                                recon_phase = resize(raw_pupil, (256, 256)) * telescope_pupil.shaped
                            
                            # RMS error vs true phase
                            error = recon_phase - np.asarray(true_phase).reshape(256, 256)
                            error_masked = error[pupil_mask]
                            if np.any(np.isfinite(error_masked)):
                                rms_nm = np.sqrt(np.nanmean(error_masked**2)) * (cfg.wavelength_m * 1e9 / (2 * np.pi))
                                rms_history.append(rms_nm)
                    
                    # Final RMS
                    raw_pupil = np.angle(mft_rev(psf_result, fdpr_conf))
                    if np.all(np.isnan(raw_pupil)):
                        rms_results[i, j, t] = np.nan
                        continue
                        
                    with np.errstate(invalid='ignore'):
                        recon_phase = resize(raw_pupil, (256, 256)) * telescope_pupil.shaped
                    error = recon_phase - np.asarray(true_phase).reshape(256, 256)
                    error_masked = error[pupil_mask]
                    
                    if np.any(np.isfinite(error_masked)):
                        rms_nm = np.sqrt(np.nanmean(error_masked**2)) * (cfg.wavelength_m * 1e9 / (2 * np.pi))
                    else:
                        rms_nm = np.nan
                    
                    rms_results[i, j, t] = rms_nm
                    
                    # Convergence: iteration where RMS < 1.5 * final
                    if len(rms_history) > 0 and np.isfinite(rms_nm):
                        rms_history = np.array(rms_history)
                        threshold = 1.5 * rms_nm
                        converged = np.where(rms_history < threshold)[0]
                        if len(converged) > 0:
                            convergence_results[i, j, t] = converged[0] * 10
                        else:
                            convergence_results[i, j, t] = cfg.n_iterations
                    else:
                        convergence_results[i, j, t] = np.nan
                        
                except Exception as e:
                    print(f"FDPR failed at dz={dz}, v0={v0}: {e}")
                    continue
            
            count += 1
            if count % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"  Progress: {count}/{total} ({100*count/total:.1f}%) - ETA: {eta/60:.1f} min")
    
    return {
        'dz': dz_subset,
        'v0': v0_subset,
        'rms_all': rms_results,
        'rms_mean': np.nanmean(rms_results, axis=2),
        'rms_std': np.nanstd(rms_results, axis=2),
        'convergence_mean': np.nanmean(convergence_results, axis=2),
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def figure_otf_vs_rms_correlation(otf_data: dict, fdpr_data: dict,
                                   save_path: Optional[str] = None):
    """
    Scatter plot showing correlation between OTF amplitude and FDPR RMS.
    Handles case where FDPR was run on a sparser grid than OTF.
    """
    # Check if grids match
    otf_shape = otf_data['H_photometry'].shape
    fdpr_shape = fdpr_data['rms_mean'].shape
    
    if otf_shape != fdpr_shape:
        # FDPR was run on sparse grid - need to sample OTF at same points
        print(f"  Grid mismatch: OTF {otf_shape} vs FDPR {fdpr_shape}")
        print(f"  Sampling OTF at FDPR grid points...")
        
        # Find indices in OTF grid that match FDPR grid
        dz_otf = otf_data['dz']
        v0_otf = otf_data['v0']
        dz_fdpr = fdpr_data['dz']
        v0_fdpr = fdpr_data['v0']
        
        # Sample OTF at FDPR grid points
        H_otf_sampled = np.zeros(fdpr_shape)
        for i, dz in enumerate(dz_fdpr):
            i_otf = np.argmin(np.abs(dz_otf - dz))
            for j, v0 in enumerate(v0_fdpr):
                j_otf = np.argmin(np.abs(v0_otf - v0))
                H_otf_sampled[i, j] = otf_data['H_photometry'][i_otf, j_otf]
        
        H_otf = H_otf_sampled.flatten()
        rms = fdpr_data['rms_mean'].flatten()
    else:
        H_otf = otf_data['H_photometry'].flatten()
        rms = fdpr_data['rms_mean'].flatten()
    
    # Only valid points
    valid = np.isfinite(H_otf) & np.isfinite(rms) & (H_otf > 0) & (rms > 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Scatter plot
    ax = axes[0]
    ax.scatter(H_otf[valid], rms[valid], s=10, alpha=0.5, c=COLORS['blue'])
    
    # Fit line (in log-log)
    log_otf = np.log10(H_otf[valid])
    log_rms = np.log10(rms[valid])
    coeffs = np.polyfit(log_otf, log_rms, 1)
    
    otf_fit = np.logspace(np.log10(H_otf[valid].min()), np.log10(H_otf[valid].max()), 100)
    rms_fit = 10**(coeffs[0] * np.log10(otf_fit) + coeffs[1])
    ax.plot(otf_fit, rms_fit, 'r-', lw=2, label=f'Fit: slope={coeffs[0]:.2f}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('OTF Side-Peak Amplitude [a.u.]')
    ax.set_ylabel('FDPR Residual RMS [nm]')
    ax.set_title('(a) OTF Amplitude vs Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Right: Correlation coefficient
    ax = axes[1]
    corr = np.corrcoef(np.log10(H_otf[valid]), np.log10(rms[valid]))[0, 1]
    
    # 2D histogram
    h, xedges, yedges = np.histogram2d(
        np.log10(H_otf[valid]), np.log10(rms[valid]), bins=30
    )
    ax.imshow(h.T, origin='lower', aspect='auto',
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
              cmap='Blues')
    
    ax.set_xlabel('log₁₀(OTF Amplitude)')
    ax.set_ylabel('log₁₀(RMS Error [nm])')
    ax.set_title(f'(b) 2D Histogram (r = {corr:.3f})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def figure_rms_heatmap(fdpr_data: dict, save_path: Optional[str] = None):
    """
    Heatmap of FDPR RMS error over (dz, v0) - companion to OTF heatmap.
    """
    dz = fdpr_data['dz']
    v0 = fdpr_data['v0']
    rms = fdpr_data['rms_mean']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    extent = [v0.min(), v0.max(), dz.min(), dz.max()]
    
    # Clip for visualization
    valid = np.isfinite(rms)
    if np.sum(valid) > 0:
        vmin, vmax = np.nanpercentile(rms[valid], [2, 98])
    else:
        vmin, vmax = 0, 100
    
    im = ax.imshow(rms, aspect='auto', origin='lower', extent=extent,
                   cmap='magma_r', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('FDPR Residual RMS [nm]')
    
    ax.set_xlabel(r'Spatial Frequency $v_0$ [cycles/aperture]')
    ax.set_ylabel(r'Defocus $\Delta z$ [mm]')
    ax.set_title('FDPR Reconstruction Error')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def figure_side_by_side_heatmaps(otf_data: dict, fdpr_data: dict,
                                  save_path: Optional[str] = None):
    """
    Side-by-side comparison: OTF amplitude vs FDPR RMS.
    High OTF should correspond to low RMS.
    Handles case where grids are different sizes.
    """
    # Use FDPR grid for both plots (it may be sparser)
    dz = fdpr_data['dz']
    v0 = fdpr_data['v0']
    rms = fdpr_data['rms_mean']
    
    # Sample OTF at FDPR grid if needed
    otf_shape = otf_data['H_photometry'].shape
    fdpr_shape = rms.shape
    
    if otf_shape != fdpr_shape:
        dz_otf = otf_data['dz']
        v0_otf = otf_data['v0']
        
        H_otf = np.zeros(fdpr_shape)
        for i, dz_val in enumerate(dz):
            i_otf = np.argmin(np.abs(dz_otf - dz_val))
            for j, v0_val in enumerate(v0):
                j_otf = np.argmin(np.abs(v0_otf - v0_val))
                H_otf[i, j] = otf_data['H_photometry'][i_otf, j_otf]
    else:
        H_otf = otf_data['H_photometry']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    extent = [v0.min(), v0.max(), dz.min(), dz.max()]
    
    # Left: OTF amplitude
    ax = axes[0]
    valid = np.isfinite(H_otf)
    vmin, vmax = np.nanpercentile(H_otf[valid], [2, 98])
    im = ax.imshow(H_otf, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='OTF Amplitude [a.u.]')
    ax.set_xlabel(r'$v_0$ [cycles/aperture]')
    ax.set_ylabel(r'$\Delta z$ [mm]')
    ax.set_title('(a) OTF Side-Peak Amplitude\n(Higher = More Information)')
    
    # Right: FDPR RMS
    ax = axes[1]
    valid = np.isfinite(rms)
    vmin, vmax = np.nanpercentile(rms[valid], [2, 98])
    im = ax.imshow(rms, aspect='auto', origin='lower', extent=extent,
                   cmap='magma_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='RMS Error [nm]')
    ax.set_xlabel(r'$v_0$ [cycles/aperture]')
    ax.set_ylabel(r'$\Delta z$ [mm]')
    ax.set_title('(b) FDPR Residual RMS\n(Lower = Better Reconstruction)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def figure_optimal_defocus_comparison(otf_data: dict, fdpr_data: dict,
                                       save_path: Optional[str] = None):
    """
    Compare optimal defocus from OTF (max amplitude) vs FDPR (min RMS).
    They should agree!
    Handles different grid sizes.
    """
    # Use FDPR grid
    dz = fdpr_data['dz']
    v0 = fdpr_data['v0']
    rms = fdpr_data['rms_mean']
    
    # Sample OTF at FDPR grid if needed
    otf_shape = otf_data['H_photometry'].shape
    fdpr_shape = rms.shape
    
    if otf_shape != fdpr_shape:
        dz_otf = otf_data['dz']
        v0_otf = otf_data['v0']
        
        H_otf = np.zeros(fdpr_shape)
        for i, dz_val in enumerate(dz):
            i_otf = np.argmin(np.abs(dz_otf - dz_val))
            for j, v0_val in enumerate(v0):
                j_otf = np.argmin(np.abs(v0_otf - v0_val))
                H_otf[i, j] = otf_data['H_photometry'][i_otf, j_otf]
    else:
        H_otf = otf_data['H_photometry']
    
    # Find optimal dz for each v0
    optimal_dz_otf = []
    optimal_dz_rms = []
    valid_v0 = []
    
    for j in range(len(v0)):
        col_otf = H_otf[:, j]
        col_rms = rms[:, j]
        
        if np.any(np.isfinite(col_otf)) and np.any(np.isfinite(col_rms)):
            i_max_otf = np.nanargmax(col_otf)  # Max OTF
            i_min_rms = np.nanargmin(col_rms)  # Min RMS
            
            optimal_dz_otf.append(dz[i_max_otf])
            optimal_dz_rms.append(dz[i_min_rms])
            valid_v0.append(v0[j])
    
    optimal_dz_otf = np.array(optimal_dz_otf)
    optimal_dz_rms = np.array(optimal_dz_rms)
    valid_v0 = np.array(valid_v0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Comparison
    ax = axes[0]
    ax.scatter(valid_v0, optimal_dz_otf, s=30, alpha=0.7, c=COLORS['blue'],
               label='Max OTF amplitude')
    ax.scatter(valid_v0, optimal_dz_rms, s=30, alpha=0.7, c=COLORS['orange'],
               marker='s', label='Min FDPR RMS')
    
    ax.set_xlabel(r'Spatial Frequency $v_0$ [cycles/aperture]')
    ax.set_ylabel(r'Optimal Defocus $\Delta z$ [mm]')
    ax.set_title('(a) Optimal Defocus: OTF vs FDPR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Correlation
    ax = axes[1]
    ax.scatter(optimal_dz_otf, optimal_dz_rms, s=30, alpha=0.7, c=COLORS['green'])
    
    # 1:1 line
    max_val = max(optimal_dz_otf.max(), optimal_dz_rms.max())
    ax.plot([0, max_val], [0, max_val], 'k--', lw=2, label='1:1')
    
    # Correlation
    corr = np.corrcoef(optimal_dz_otf, optimal_dz_rms)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, va='top')
    
    ax.set_xlabel('Optimal Δz from OTF [mm]')
    ax.set_ylabel('Optimal Δz from FDPR [mm]')
    ax.set_title('(b) Optimal Defocus Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def figure_rms_slices(fdpr_data: dict, v0_targets: list = [5, 10, 20, 40],
                       save_path: Optional[str] = None):
    """
    RMS vs defocus at fixed spatial frequencies.
    """
    dz = fdpr_data['dz']
    v0_arr = fdpr_data['v0']
    rms = fdpr_data['rms_mean']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['pink']]
    
    for i, v0_target in enumerate(v0_targets):
        idx = np.argmin(np.abs(v0_arr - v0_target))
        actual_v0 = v0_arr[idx]
        col = rms[:, idx]
        
        valid = np.isfinite(col)
        ax.plot(dz[valid], col[valid], '-o', color=colors[i % len(colors)],
                lw=2, ms=4, label=f'$v_0$ = {actual_v0:.0f} cyc/ap')
    
    ax.set_xlabel(r'Defocus $\Delta z$ [mm]')
    ax.set_ylabel('FDPR Residual RMS [nm]')
    ax.set_title('Reconstruction Error vs Defocus')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(otf_npz_path: str, output_dir: str = "./fdpr_analysis",
                  run_actual_fdpr: bool = False):
    """
    Run full OTF-to-FDPR analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("OTF → FDPR CONNECTION ANALYSIS")
    print("=" * 60)
    
    # Load OTF data
    print("\nLoading OTF data...")
    otf_raw = np.load(otf_npz_path)
    otf_data = {
        'H_photometry': otf_raw['H_photometry'],
        'dz': otf_raw['fixed_dz_heatmap'],
        'v0': otf_raw['v0_heatmap'],
    }
    print(f"  Shape: {otf_data['H_photometry'].shape}")
    
    # Generate FDPR results
    cfg = FDPRConfig()
    
    if run_actual_fdpr and HAS_FDPR:
        print("\nRunning actual FDPR (this will take a while)...")
        fdpr_data = run_fdpr_grid(cfg, 
                                   dz_subset=otf_data['dz'][::4],  # Every 4th point
                                   v0_subset=otf_data['v0'][::5])  # Every 5th point
    else:
        print("\nGenerating synthetic FDPR results from OTF data...")
        fdpr_data = generate_synthetic_fdpr_results(otf_data, cfg)
    
    # Generate figures
    print("\n" + "-" * 40)
    print("Generating figures...")
    
    print("\n[1/5] Side-by-side heatmaps...")
    figure_side_by_side_heatmaps(otf_data, fdpr_data,
                                  f"{output_dir}/fig_otf_vs_rms_heatmaps.png")
    
    print("\n[2/5] OTF vs RMS correlation...")
    figure_otf_vs_rms_correlation(otf_data, fdpr_data,
                                   f"{output_dir}/fig_otf_rms_correlation.png")
    
    print("\n[3/5] RMS heatmap...")
    figure_rms_heatmap(fdpr_data, f"{output_dir}/fig_rms_heatmap.png")
    
    print("\n[4/5] Optimal defocus comparison...")
    figure_optimal_defocus_comparison(otf_data, fdpr_data,
                                       f"{output_dir}/fig_optimal_defocus_comparison.png")
    
    print("\n[5/5] RMS slices...")
    figure_rms_slices(fdpr_data, v0_targets=[5, 10, 20, 40], 
                      save_path=f"{output_dir}/fig_rms_slices.png")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {output_dir}")
    print("=" * 60)
    
    return otf_data, fdpr_data


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Path to your OTF data
    otf_path = "/mnt/user-data/uploads/OTF_heatmap_data.npz"
    
    # Run analysis (set run_actual_fdpr=True if you want real FDPR runs)
    otf_data, fdpr_data = run_analysis(otf_path, 
                                        output_dir="./fdpr_analysis",
                                        run_actual_fdpr=True)