# Dense FDPR Grid Analysis
# =========================
# Runs FDPR on a dense grid to match OTF heatmap resolution
# WARNING: This will take a LONG time! Consider running overnight.
#
# Estimated time for 80x150 grid with 1 trial, 200 iterations:
#   ~12,000 FDPR runs × ~2-5 sec each = 7-17 hours
#
# Recommendations:
#   - Start with a medium grid (40x75) to test: ~2-4 hours
#   - Use fewer iterations (100 instead of 200) for speed
#   - Run overnight for full resolution

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from dataclasses import dataclass
import time
import os
import warnings

# Suppress warnings during FDPR
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# IMPORTS
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
    raise ImportError("hcipy is required for this script")

try:
    from image_sharpening import FocusDiversePhaseRetrieval, mft_rev, InstrumentConfiguration
    from skimage.transform import resize
    HAS_FDPR = True
except ImportError:
    HAS_FDPR = False
    raise ImportError("image_sharpening and skimage are required for this script")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DenseGridConfig:
    """Configuration for dense FDPR grid."""
    # Optical system (SEAL)
    wavelength_m: float = 650e-9
    pupil_diameter_m: float = 10.12e-3
    focal_length_m: float = 500e-3
    pupil_npix: int = 256
    
    # FDPR parameters
    n_iterations: int = 200  # Reduce to 100-150 for faster runs
    n_trials: int = 1        # Keep at 1 for speed, increase for statistics
    
    # Noise
    read_noise_e: float = 5.0
    
    # Aberration
    m_waves: float = 0.1
    
    # Grid density options
    # 'full': 80x150 (matches OTF exactly) - ~10+ hours
    # 'dense': 40x75 - ~3 hours  
    # 'medium': 20x40 - ~45 min
    # 'sparse': 10x20 - ~10 min (for testing)
    grid_density: str = 'dense'
    
    def get_grid_params(self):
        """Return (n_dz, n_v0) based on density setting."""
        params = {
            'full': (80, 150),
            'dense': (40, 75),
            'medium': (20, 40),
            'sparse': (10, 20),
        }
        return params.get(self.grid_density, (40, 75))


# =============================================================================
# SETUP OPTICAL SYSTEM
# =============================================================================

def setup_optics(cfg: DenseGridConfig):
    """Initialize optical system components."""
    print("Setting up optical system...")
    
    pupil_grid = make_pupil_grid(cfg.pupil_npix, cfg.pupil_diameter_m)
    focal_grid = make_focal_grid(
        q=16, num_airy=16,  # Match original notebook parameters
        pupil_diameter=cfg.pupil_diameter_m,
        focal_length=cfg.focal_length_m,
        reference_wavelength=cfg.wavelength_m
    )
    
    aperture = make_circular_aperture(cfg.pupil_diameter_m)
    telescope_pupil = aperture(pupil_grid)
    propagator = FraunhoferPropagator(pupil_grid, focal_grid, cfg.focal_length_m)
    
    # Zernike for defocus
    zernike_basis = make_zernike_basis(num_modes=256, D=cfg.pupil_diameter_m, grid=pupil_grid)
    defocus_template = zernike_basis[3]
    
    # Normalize defocus
    pupil_mask = np.asarray(telescope_pupil.shaped, dtype=bool)
    defocus_shaped = defocus_template.shaped
    template_pv = defocus_shaped[pupil_mask].ptp()
    unit_defocus = defocus_template / template_pv
    
    # FDPR config
    seal_param_config = {
        'image_dx': 2.0071,
        'efl': cfg.focal_length_m * 1e3,
        'wavelength': cfg.wavelength_m * 1e6,
        'pupil_size': cfg.pupil_diameter_m * 1e3,
    }
    fdpr_conf = InstrumentConfiguration(seal_param_config)
    
    return {
        'pupil_grid': pupil_grid,
        'focal_grid': focal_grid,
        'telescope_pupil': telescope_pupil,
        'propagator': propagator,
        'unit_defocus': unit_defocus,
        'pupil_mask': pupil_mask,
        'fdpr_conf': fdpr_conf,
    }


# =============================================================================
# SINGLE FDPR RUN
# =============================================================================

def run_single_fdpr(dz_mm: float, v0: float, cfg: DenseGridConfig, optics: dict,
                    rng: np.random.Generator) -> float:
    """
    Run a single FDPR reconstruction and return RMS error.
    
    Returns NaN if reconstruction fails.
    """
    # Compute defocus phase
    dz_m = dz_mm * 1e-3
    f_number = cfg.focal_length_m / cfg.pupil_diameter_m
    defocus_pv_m = -dz_m / (8 * f_number**2)
    phase_pv_rad = defocus_pv_m * (2 * np.pi / cfg.wavelength_m)
    phi_defocus = optics['unit_defocus'] * phase_pv_rad
    
    # Create sinusoidal aberration (true phase)
    x = optics['pupil_grid'].x
    phi_sine_waves = cfg.m_waves * np.sin(2 * np.pi * v0 * (x / cfg.pupil_diameter_m))
    phi_sine_rad = 2 * np.pi * Field(phi_sine_waves, optics['pupil_grid'])
    true_phase = phi_sine_rad
    
    # Generate PSFs
    telescope_pupil = optics['telescope_pupil']
    propagator = optics['propagator']
    
    wf_focus = Wavefront(telescope_pupil * np.exp(1j * true_phase), cfg.wavelength_m)
    psf_focus = np.asarray(propagator(wf_focus).intensity.shaped)
    
    wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (true_phase + phi_defocus)), cfg.wavelength_m)
    psf_defoc = np.asarray(propagator(wf_defoc).intensity.shaped)
    
    # Add noise and clip
    psf_focus_noisy = psf_focus + rng.normal(0, cfg.read_noise_e, psf_focus.shape)
    psf_defoc_noisy = psf_defoc + rng.normal(0, cfg.read_noise_e, psf_defoc.shape)
    psf_focus_noisy = np.maximum(psf_focus_noisy, 0)
    psf_defoc_noisy = np.maximum(psf_defoc_noisy, 0)
    
    # Run FDPR
    try:
        psf_list = [psf_focus_noisy, psf_defoc_noisy]
        dx_list = [2.0071]
        
        # IMPORTANT: FDPR expects distance in microns
        dz_um = dz_mm * 1e3  # mm → μm
        
        mp = FocusDiversePhaseRetrieval(
            psf_list, cfg.wavelength_m * 1e6, dx_list, [dz_um]
        )
        
        for _ in range(cfg.n_iterations):
            psf_result = mp.step()
        
        # Extract reconstructed phase
        raw_pupil = np.angle(mft_rev(psf_result, optics['fdpr_conf']))
        
        if np.all(np.isnan(raw_pupil)):
            return np.nan
        
        with np.errstate(invalid='ignore'):
            recon_phase = resize(raw_pupil, (256, 256)) * telescope_pupil.shaped
        
        # Compute RMS error
        true_phase_arr = np.asarray(true_phase).reshape(256, 256)
        error = recon_phase - true_phase_arr
        error_masked = error[optics['pupil_mask']]
        
        if not np.any(np.isfinite(error_masked)):
            return np.nan
        
        rms_rad = np.sqrt(np.nanmean(error_masked**2))
        rms_nm = rms_rad * (cfg.wavelength_m * 1e9 / (2 * np.pi))
        
        return rms_nm
        
    except Exception as e:
        return np.nan


# =============================================================================
# MAIN GRID RUNNER
# =============================================================================

def run_dense_fdpr_grid(cfg: DenseGridConfig, otf_data: dict,
                         save_intermediate: bool = True,
                         output_dir: str = "./fdpr_dense") -> dict:
    """
    Run FDPR on a dense grid matching OTF data resolution.
    
    Parameters
    ----------
    cfg : DenseGridConfig
        Configuration
    otf_data : dict
        OTF data with 'dz', 'v0', 'H_photometry' keys
    save_intermediate : bool
        Save results every 10 rows (for crash recovery)
    output_dir : str
        Output directory
    
    Returns
    -------
    dict
        FDPR results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get grid size
    n_dz, n_v0 = cfg.get_grid_params()
    
    # Create grids that span the same range as OTF
    dz_min, dz_max = otf_data['dz'].min(), otf_data['dz'].max()
    v0_min, v0_max = otf_data['v0'].min(), otf_data['v0'].max()
    
    dz_values = np.linspace(dz_min, dz_max, n_dz)
    v0_values = np.linspace(v0_min, v0_max, n_v0)
    
    print("=" * 60)
    print("DENSE FDPR GRID ANALYSIS")
    print("=" * 60)
    print(f"Grid density: {cfg.grid_density}")
    print(f"Grid size: {n_dz} × {n_v0} = {n_dz * n_v0} points")
    print(f"Δz range: {dz_min:.1f} - {dz_max:.1f} mm")
    print(f"v₀ range: {v0_min:.1f} - {v0_max:.1f} cycles/aperture")
    print(f"Iterations per point: {cfg.n_iterations}")
    print(f"Trials per point: {cfg.n_trials}")
    print()
    
    # Estimate time
    est_time_per_point = 3  # seconds (rough estimate)
    total_points = n_dz * n_v0 * cfg.n_trials
    est_total_time = total_points * est_time_per_point
    print(f"Estimated time: {est_total_time/3600:.1f} hours")
    print("=" * 60)
    print()
    
    # Setup optics
    optics = setup_optics(cfg)
    
    # Storage
    rms_results = np.full((n_dz, n_v0, cfg.n_trials), np.nan)
    
    # Random generator
    rng = np.random.default_rng(12345)
    
    # Main loop
    t_start = time.time()
    completed = 0
    total = n_dz * n_v0 * cfg.n_trials
    
    for i, dz in enumerate(dz_values):
        row_start = time.time()
        
        for j, v0 in enumerate(v0_values):
            for t in range(cfg.n_trials):
                rms = run_single_fdpr(dz, v0, cfg, optics, rng)
                rms_results[i, j, t] = rms
                completed += 1
        
        # Progress report
        row_time = time.time() - row_start
        elapsed = time.time() - t_start
        rate = completed / elapsed
        remaining = (total - completed) / rate if rate > 0 else 0
        
        valid_this_row = np.sum(np.isfinite(rms_results[i, :, :]))
        total_this_row = n_v0 * cfg.n_trials
        
        print(f"Row {i+1}/{n_dz} (Δz={dz:.1f}mm): "
              f"{valid_this_row}/{total_this_row} valid, "
              f"{row_time:.1f}s, "
              f"ETA: {remaining/60:.1f} min")
        
        # Save intermediate results
        if save_intermediate and (i + 1) % 10 == 0:
            np.savez(
                f"{output_dir}/fdpr_intermediate_row{i+1}.npz",
                dz_values=dz_values,
                v0_values=v0_values,
                rms_results=rms_results,
                completed_rows=i+1,
            )
            print(f"  → Saved intermediate results")
    
    # Compute statistics
    rms_mean = np.nanmean(rms_results, axis=2)
    rms_std = np.nanstd(rms_results, axis=2)
    
    # Final save
    results = {
        'dz': dz_values,
        'v0': v0_values,
        'rms_all': rms_results,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'config': {
            'grid_density': cfg.grid_density,
            'n_iterations': cfg.n_iterations,
            'n_trials': cfg.n_trials,
            'read_noise_e': cfg.read_noise_e,
            'm_waves': cfg.m_waves,
        }
    }
    
    np.savez(f"{output_dir}/fdpr_dense_results.npz", **results)
    print(f"\nSaved final results to {output_dir}/fdpr_dense_results.npz")
    
    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/3600:.2f} hours")
    
    return results


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_correlation(otf_data: dict, fdpr_data: dict) -> dict:
    """
    Compute correlation between OTF amplitude and FDPR RMS.
    
    Returns correlation coefficient and other statistics.
    """
    # Sample OTF at FDPR grid points
    dz_otf = otf_data['dz']
    v0_otf = otf_data['v0']
    dz_fdpr = fdpr_data['dz']
    v0_fdpr = fdpr_data['v0']
    
    H_otf_sampled = np.zeros_like(fdpr_data['rms_mean'])
    for i, dz in enumerate(dz_fdpr):
        i_otf = np.argmin(np.abs(dz_otf - dz))
        for j, v0 in enumerate(v0_fdpr):
            j_otf = np.argmin(np.abs(v0_otf - v0))
            H_otf_sampled[i, j] = otf_data['H_photometry'][i_otf, j_otf]
    
    # Flatten and filter valid points
    otf_flat = H_otf_sampled.flatten()
    rms_flat = fdpr_data['rms_mean'].flatten()
    
    valid = np.isfinite(otf_flat) & np.isfinite(rms_flat) & (otf_flat > 0) & (rms_flat > 0)
    
    otf_valid = otf_flat[valid]
    rms_valid = rms_flat[valid]
    
    # Pearson correlation (linear)
    pearson_r = np.corrcoef(otf_valid, rms_valid)[0, 1]
    
    # Spearman correlation (rank-based, better for nonlinear relationships)
    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(otf_valid, rms_valid)
    
    # Log-log correlation (for power-law relationship)
    log_otf = np.log10(otf_valid)
    log_rms = np.log10(rms_valid)
    log_pearson_r = np.corrcoef(log_otf, log_rms)[0, 1]
    
    # Power-law fit: log(RMS) = a * log(OTF) + b
    coeffs = np.polyfit(log_otf, log_rms, 1)
    slope, intercept = coeffs
    
    print("=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)
    print(f"Valid points: {np.sum(valid)} / {len(valid)}")
    print(f"\nPearson r (linear): {pearson_r:.4f}")
    print(f"Spearman r (rank): {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"Pearson r (log-log): {log_pearson_r:.4f}")
    print(f"\nPower-law fit: RMS ∝ OTF^{slope:.3f}")
    print(f"  log₁₀(RMS) = {slope:.3f} × log₁₀(OTF) + {intercept:.3f}")
    print("=" * 50)
    
    return {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'log_pearson_r': log_pearson_r,
        'power_law_slope': slope,
        'power_law_intercept': intercept,
        'n_valid': np.sum(valid),
        'otf_valid': otf_valid,
        'rms_valid': rms_valid,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_side_by_side(otf_data: dict, fdpr_data: dict, corr_stats: dict,
                       save_path: Optional[str] = None):
    """
    Create publication-quality side-by-side heatmaps with correlation info.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample OTF at FDPR grid
    dz_otf = otf_data['dz']
    v0_otf = otf_data['v0']
    dz_fdpr = fdpr_data['dz']
    v0_fdpr = fdpr_data['v0']
    
    H_otf_sampled = np.zeros_like(fdpr_data['rms_mean'])
    for i, dz in enumerate(dz_fdpr):
        i_otf = np.argmin(np.abs(dz_otf - dz))
        for j, v0 in enumerate(v0_fdpr):
            j_otf = np.argmin(np.abs(v0_otf - v0))
            H_otf_sampled[i, j] = otf_data['H_photometry'][i_otf, j_otf]
    
    rms = fdpr_data['rms_mean']
    extent = [v0_fdpr.min(), v0_fdpr.max(), dz_fdpr.min(), dz_fdpr.max()]
    
    # Left: OTF
    ax = axes[0]
    valid = np.isfinite(H_otf_sampled)
    vmin, vmax = np.nanpercentile(H_otf_sampled[valid], [2, 98])
    im = ax.imshow(H_otf_sampled, aspect='auto', origin='lower', extent=extent,
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
    
    # Add correlation info
    fig.text(0.5, 0.02, 
             f"Spearman r = {corr_stats['spearman_r']:.3f}, "
             f"Power-law: RMS ∝ OTF$^{{{corr_stats['power_law_slope']:.2f}}}$",
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_correlation_scatter(corr_stats: dict, save_path: Optional[str] = None):
    """
    Scatter plot showing OTF vs RMS correlation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    otf = corr_stats['otf_valid']
    rms = corr_stats['rms_valid']
    
    # Left: Log-log scatter with fit
    ax = axes[0]
    ax.scatter(otf, rms, s=10, alpha=0.5, c='#0072B2')
    
    # Power-law fit line
    otf_fit = np.logspace(np.log10(otf.min()), np.log10(otf.max()), 100)
    rms_fit = 10**(corr_stats['power_law_slope'] * np.log10(otf_fit) + 
                   corr_stats['power_law_intercept'])
    ax.plot(otf_fit, rms_fit, 'r-', lw=2, 
            label=f"Fit: RMS ∝ OTF$^{{{corr_stats['power_law_slope']:.2f}}}$")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('OTF Side-Peak Amplitude [a.u.]')
    ax.set_ylabel('FDPR Residual RMS [nm]')
    ax.set_title(f"(a) OTF vs RMS (log-log)\nr = {corr_stats['log_pearson_r']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Right: Rank plot (for Spearman)
    ax = axes[1]
    otf_rank = np.argsort(np.argsort(otf))
    rms_rank = np.argsort(np.argsort(rms))
    ax.scatter(otf_rank, rms_rank, s=10, alpha=0.5, c='#D55E00')
    ax.plot([0, len(otf)], [len(rms), 0], 'k--', lw=1, alpha=0.5, label='Perfect negative correlation')
    ax.set_xlabel('OTF Rank (low to high)')
    ax.set_ylabel('RMS Rank (low to high)')
    ax.set_title(f"(b) Rank Correlation\nSpearman r = {corr_stats['spearman_r']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main(otf_path: str, output_dir: str = "./fdpr_dense",
         grid_density: str = 'dense', n_iterations: int = 200):
    """
    Main entry point.
    
    Parameters
    ----------
    otf_path : str
        Path to OTF_heatmap_data.npz
    output_dir : str
        Output directory
    grid_density : str
        'full' (80x150), 'dense' (40x75), 'medium' (20x40), 'sparse' (10x20)
    n_iterations : int
        FDPR iterations per point
    """
    # Load OTF data
    print("Loading OTF data...")
    otf_raw = np.load(otf_path)
    otf_data = {
        'H_photometry': otf_raw['H_photometry'],
        'dz': otf_raw['fixed_dz_heatmap'],
        'v0': otf_raw['v0_heatmap'],
    }
    
    # Configure
    cfg = DenseGridConfig(
        grid_density=grid_density,
        n_iterations=n_iterations,
    )
    
    # Run FDPR grid
    fdpr_data = run_dense_fdpr_grid(cfg, otf_data, output_dir=output_dir)
    
    # Compute correlation
    corr_stats = compute_correlation(otf_data, fdpr_data)
    
    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    plot_side_by_side(otf_data, fdpr_data, corr_stats,
                      f"{output_dir}/fig_otf_vs_rms_dense.png")
    plot_correlation_scatter(corr_stats, 
                             f"{output_dir}/fig_correlation.png")
    
    # Save correlation stats
    np.savez(f"{output_dir}/correlation_stats.npz", **corr_stats)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return otf_data, fdpr_data, corr_stats


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    otf_path = "/Users/joshuapotter/Documents/SEAL/FDPRNotebooks/OTF_heatmap_data.npz"
    output_dir = "./fdpr_dense_results"
    
    # Grid density options:
    # 'sparse' - 10x20 grid, ~10 min (for testing)
    # 'medium' - 20x40 grid, ~45 min
    # 'dense'  - 40x75 grid, ~3 hours
    # 'full'   - 80x150 grid, ~10+ hours (matches OTF exactly)
    
    grid_density = 'full'  # Start with this, upgrade to 'full' overnight
    n_iterations = 150      # Reduce from 200 for speed
    
    otf_data, fdpr_data, corr_stats = main(
        otf_path, 
        output_dir=output_dir,
        grid_density=grid_density,
        n_iterations=n_iterations
    )