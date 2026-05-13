# High-Resolution OTF Analysis for Dean & Bowers Theory Comparison
# ================================================================
# Generates OTF data with much finer defocus sampling to match theory curve
#
# The issue: Your current 80 defocus points aren't enough to capture
# the sharp oscillations in |sin(πv₀²/8â)|
#
# Solution: Increase defocus sampling to 200-400 points

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from dataclasses import dataclass
import time

# HCIPy imports
from hcipy import (
    make_pupil_grid, make_focal_grid, make_uniform_grid,
    make_circular_aperture, make_zernike_basis,
    Wavefront, FraunhoferPropagator, Field
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass 
class HighResConfig:
    """Configuration for high-resolution OTF analysis."""
    # Optical system (SEAL)
    wavelength_m: float = 650e-9
    pupil_diameter_m: float = 10.12e-3
    focal_length_m: float = 500e-3
    pupil_npix: int = 256
    q: int = 4
    num_airy: int = 64
    
    # Aberration
    m_waves: float = 0.1
    
    # HIGH RESOLUTION defocus grid
    dz_min_mm: float = 5.0
    dz_max_mm: float = 250.0
    n_dz: int = 300  # Increase from 80 to 300 for smooth curve
    
    # Spatial frequencies to analyze (for single-v0 plots)
    v0_single: float = 10.0  # The v0 used in your theory comparison plot
    
    # Full heatmap (optional, takes longer)
    v0_min: float = 3.0
    v0_max: float = 80.0
    n_v0: int = 150
    
    # Photometry aperture
    photometry_aperture_radius: float = 3.0
    
    @property
    def f_number(self):
        return self.focal_length_m / self.pupil_diameter_m


# =============================================================================
# PHYSICS
# =============================================================================

def dz_to_a_hat(dz_mm: float, cfg: HighResConfig) -> float:
    """Convert mechanical defocus (mm) to normalized defocus â (waves P-V)."""
    dz_m = dz_mm * 1e-3
    return (dz_m * cfg.pupil_diameter_m**2) / (8 * cfg.focal_length_m**2 * cfg.wavelength_m)


def a_hat_to_dz(a_hat: float, cfg: HighResConfig) -> float:
    """Convert â (waves P-V) to mechanical defocus (mm)."""
    dz_m = 8 * a_hat * cfg.focal_length_m**2 * cfg.wavelength_m / cfg.pupil_diameter_m**2
    return dz_m * 1e3


def dean_bowers_theory(a_hat: np.ndarray, v0: float) -> np.ndarray:
    """Dean & Bowers: OTF amplitude ~ |sin(πv₀²/8â)|"""
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = np.pi * v0**2 / (8 * a_hat)
        result = np.abs(np.sin(arg))
        result[a_hat < 0.01] = 0  # Avoid singularity at â=0
    return result


# =============================================================================
# OPTICAL SYSTEM SETUP
# =============================================================================

def setup_optics(cfg: HighResConfig) -> dict:
    """Initialize optical system."""
    pupil_grid = make_pupil_grid(cfg.pupil_npix, cfg.pupil_diameter_m)
    focal_grid = make_focal_grid(
        q=cfg.q, num_airy=cfg.num_airy,
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
    
    pupil_mask = np.asarray(telescope_pupil.shaped, dtype=bool)
    defocus_shaped = defocus_template.shaped
    template_pv = defocus_shaped[pupil_mask].ptp()
    unit_defocus = defocus_template / template_pv
    
    # Pixel scales for photometry
    dtheta = (2 * cfg.num_airy * (cfg.wavelength_m / cfg.pupil_diameter_m)) / focal_grid.shape[0]
    delta_x = 2 * cfg.num_airy * (cfg.wavelength_m / cfg.pupil_diameter_m) * cfg.focal_length_m
    dx = delta_x / focal_grid.shape[0]
    delta_k = (2 * np.pi) / dx
    dk = (2 * np.pi) / delta_x
    
    return {
        'pupil_grid': pupil_grid,
        'focal_grid': focal_grid,
        'telescope_pupil': telescope_pupil,
        'propagator': propagator,
        'unit_defocus': unit_defocus,
        'pupil_mask': pupil_mask,
        'delta_x': delta_x,
        'dk': dk,
        'delta_k': delta_k,
    }


# =============================================================================
# OTF MEASUREMENT
# =============================================================================

def compute_psf_otf(phi_aberration: Field, phi_defocus: Field, 
                    cfg: HighResConfig, optics: dict):
    """Compute PSF and OTF for given aberration and defocus."""
    total_phase = phi_aberration + phi_defocus
    wf = Wavefront(
        optics['telescope_pupil'] * np.exp(1j * total_phase),
        cfg.wavelength_m
    )
    psf = np.asarray(optics['propagator'](wf).intensity.shaped)
    otf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf))))
    return psf, otf


def measure_photometry(otf: np.ndarray, px_pred: float, cfg: HighResConfig, 
                       optics: dict) -> float:
    """Measure OTF amplitude at predicted peak location using aperture photometry."""
    dk = optics['dk']
    delta_k = optics['delta_k']
    
    otf_grid = make_uniform_grid([otf.shape[0], otf.shape[1]], delta_k)
    aperture_radius = cfg.photometry_aperture_radius * dk
    
    # Positive side
    peak_aperture = make_circular_aperture(aperture_radius, center=[px_pred * dk, 0])
    mask = peak_aperture(otf_grid) > 0
    
    if not np.any(mask.shaped):
        return np.nan
    
    amp_pos = float(otf[mask.shaped].max())
    
    # Negative side
    peak_aperture_neg = make_circular_aperture(aperture_radius, center=[-px_pred * dk, 0])
    mask_neg = peak_aperture_neg(otf_grid) > 0
    
    if np.any(mask_neg.shaped):
        amp_neg = float(otf[mask_neg.shaped].max())
        return 0.5 * (amp_pos + amp_neg)
    
    return amp_pos


def predict_peak_position(dz_mm: float, v0: float, cfg: HighResConfig, 
                          optics: dict) -> float:
    """Predict OTF side-peak position in pixels."""
    d_proj = dz_mm * 1e-3 / cfg.f_number
    p_pred = d_proj / v0
    px_pred = optics['delta_x'] / p_pred
    return px_pred


# =============================================================================
# HIGH-RES SINGLE-V0 SWEEP
# =============================================================================

def run_high_res_single_v0(cfg: HighResConfig, optics: dict, 
                            v0: float = None) -> dict:
    """
    Run high-resolution defocus sweep for a single spatial frequency.
    This is what you need for the theory comparison plot.
    """
    if v0 is None:
        v0 = cfg.v0_single
    
    print(f"\nHigh-resolution sweep for v₀ = {v0} cyc/ap")
    print(f"Defocus points: {cfg.n_dz}")
    print(f"Range: {cfg.dz_min_mm} - {cfg.dz_max_mm} mm")
    
    dz_values = np.linspace(cfg.dz_min_mm, cfg.dz_max_mm, cfg.n_dz)
    
    # Precompute sinusoidal aberration
    x = optics['pupil_grid'].x
    phi_sine_waves = cfg.m_waves * np.sin(2 * np.pi * v0 * (x / cfg.pupil_diameter_m))
    phi_sine_rad = 2 * np.pi * Field(phi_sine_waves, optics['pupil_grid'])
    
    # Storage
    otf_amplitudes = np.zeros(cfg.n_dz)
    a_hat_values = np.zeros(cfg.n_dz)
    
    t0 = time.time()
    
    for i, dz in enumerate(dz_values):
        # Compute defocus phase
        dz_m = dz * 1e-3
        defocus_pv_m = -dz_m / (8 * cfg.f_number**2)
        phase_pv_rad = defocus_pv_m * (2 * np.pi / cfg.wavelength_m)
        phi_defocus = optics['unit_defocus'] * phase_pv_rad
        
        # Compute OTF
        _, otf = compute_psf_otf(phi_sine_rad, phi_defocus, cfg, optics)
        
        # Measure at predicted location
        px_pred = predict_peak_position(dz, v0, cfg, optics)
        amp = measure_photometry(otf, px_pred, cfg, optics)
        
        otf_amplitudes[i] = amp
        a_hat_values[i] = dz_to_a_hat(dz, cfg)
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{cfg.n_dz} done...")
    
    elapsed = time.time() - t0
    print(f"  Complete in {elapsed:.1f}s")
    
    return {
        'dz': dz_values,
        'a_hat': a_hat_values,
        'otf_amplitude': otf_amplitudes,
        'v0': v0,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_theory_comparison(data: dict, cfg: HighResConfig, 
                           save_path: Optional[str] = None):
    """
    Plot measured OTF vs Dean & Bowers theory - high resolution version.
    """
    v0 = data['v0']
    a_hat = data['a_hat']
    dz = data['dz']
    measured = data['otf_amplitude']
    
    # Normalize measured data
    measured_norm = measured / np.nanmax(measured)
    
    # Theory curve (use even finer sampling for smooth line)
    a_hat_theory = np.linspace(0.1, a_hat.max(), 1000)
    theory = dean_bowers_theory(a_hat_theory, v0)
    
    # Convert theory a_hat to dz for right panel
    dz_theory = np.array([a_hat_to_dz(a, cfg) for a in a_hat_theory])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: vs â
    ax = axes[0]
    ax.plot(a_hat, measured_norm, 'o-', color='#0072B2', lw=1.5, ms=3, 
            alpha=0.8, label='Measured (normalized)')
    ax.plot(a_hat_theory, theory, 'r-', lw=2, 
            label=r'Theory: $|\sin(\pi v_0^2 / 8\hat{a})|$')
    
    ax.set_xlabel(r'Normalized Defocus $\hat{a}$ [waves P-V]')
    ax.set_ylabel('OTF Amplitude (normalized)')
    ax.set_title(f'(a) Comparison to Theory ($v_0$ = {v0:.0f} cyc/ap)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, a_hat.max())
    ax.set_ylim(0, 1.1)
    
    # Right: vs Δz
    ax = axes[1]
    ax.plot(dz, measured_norm, 'o-', color='#0072B2', lw=1.5, ms=3,
            alpha=0.8, label='Measured')
    ax.plot(dz_theory, theory, 'r-', lw=2, label='Theory')
    
    # Mark theoretical maxima (n=1,2,3,...)
    for n in range(1, 8):
        a_max = v0**2 / (4 * (2*n - 1))
        dz_max = a_hat_to_dz(a_max, cfg)
        if dz_max <= dz.max() and dz_max >= dz.min():
            ax.axvline(dz_max, color='gray', linestyle=':', alpha=0.5)
            ax.text(dz_max, 1.05, f'n={n}', ha='center', fontsize=9, color='gray')
    
    ax.set_xlabel(r'Defocus $\Delta z$ [mm]')
    ax.set_ylabel('OTF Amplitude (normalized)')
    ax.set_title(f'(b) Measured vs Theory ($v_0$ = {v0:.0f} cyc/ap)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, dz.max())
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_multiple_v0(cfg: HighResConfig, optics: dict,
                     v0_list: list = [5, 10, 20, 40],
                     save_path: Optional[str] = None):
    """
    Plot theory comparison for multiple spatial frequencies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    
    for idx, v0 in enumerate(v0_list):
        print(f"\nProcessing v₀ = {v0}...")
        
        # Run sweep (use fewer points for speed)
        cfg_temp = HighResConfig(n_dz=200)
        data = run_high_res_single_v0(cfg_temp, optics, v0=v0)
        
        ax = axes[idx]
        
        # Measured
        measured_norm = data['otf_amplitude'] / np.nanmax(data['otf_amplitude'])
        ax.plot(data['dz'], measured_norm, 'o-', color=colors[idx], 
                lw=1, ms=2, alpha=0.7, label='Measured')
        
        # Theory
        a_hat_theory = np.linspace(0.1, data['a_hat'].max(), 500)
        theory = dean_bowers_theory(a_hat_theory, v0)
        dz_theory = np.array([a_hat_to_dz(a, cfg) for a in a_hat_theory])
        ax.plot(dz_theory, theory, 'r-', lw=2, label='Theory')
        
        ax.set_xlabel(r'$\Delta z$ [mm]')
        ax.set_ylabel('OTF Amplitude (norm)')
        ax.set_title(f'$v_0$ = {v0} cyc/ap')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, data['dz'].max())
        ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# FULL HEATMAP (OPTIONAL)
# =============================================================================

def run_high_res_heatmap(cfg: HighResConfig, optics: dict) -> dict:
    """
    Run high-resolution heatmap over full (dz, v0) grid.
    This takes longer but gives smoother results.
    """
    print("\n" + "=" * 60)
    print("HIGH-RESOLUTION HEATMAP")
    print("=" * 60)
    print(f"Grid: {cfg.n_dz} × {cfg.n_v0} = {cfg.n_dz * cfg.n_v0} points")
    
    dz_values = np.linspace(cfg.dz_min_mm, cfg.dz_max_mm, cfg.n_dz)
    v0_values = np.linspace(cfg.v0_min, cfg.v0_max, cfg.n_v0)
    
    H = np.full((cfg.n_dz, cfg.n_v0), np.nan)
    
    # Precompute defocus phases
    print("Precomputing defocus phases...")
    defocus_phases = {}
    for dz in dz_values:
        dz_m = dz * 1e-3
        defocus_pv_m = -dz_m / (8 * cfg.f_number**2)
        phase_pv_rad = defocus_pv_m * (2 * np.pi / cfg.wavelength_m)
        defocus_phases[dz] = optics['unit_defocus'] * phase_pv_rad
    
    # Precompute sinusoidal phases
    print("Precomputing sinusoidal phases...")
    sine_phases = {}
    for v0 in v0_values:
        x = optics['pupil_grid'].x
        phi_sine_waves = cfg.m_waves * np.sin(2 * np.pi * v0 * (x / cfg.pupil_diameter_m))
        sine_phases[v0] = 2 * np.pi * Field(phi_sine_waves, optics['pupil_grid'])
    
    # Main loop
    t0 = time.time()
    for i, dz in enumerate(dz_values):
        phi_def = defocus_phases[dz]
        
        for j, v0 in enumerate(v0_values):
            phi_sine = sine_phases[v0]
            
            _, otf = compute_psf_otf(phi_sine, phi_def, cfg, optics)
            
            px_pred = predict_peak_position(dz, v0, cfg, optics)
            H[i, j] = measure_photometry(otf, px_pred, cfg, optics)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (cfg.n_dz - i - 1)
            print(f"  Row {i+1}/{cfg.n_dz}, ETA: {eta/60:.1f} min")
    
    print(f"  Complete in {(time.time()-t0)/60:.1f} min")
    
    return {
        'dz': dz_values,
        'v0': v0_values,
        'H': H,
        'H_photometry': H,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("HIGH-RESOLUTION OTF ANALYSIS")
    print("=" * 60)
    
    # Configuration - ADJUST THESE AS NEEDED
    cfg = HighResConfig(
        n_dz=300,           # High resolution for smooth curve
        v0_single=10.0,     # The frequency for your theory plot
    )
    
    # Setup optics
    print("\nSetting up optical system...")
    optics = setup_optics(cfg)
    
    # Run high-res sweep for v0=10
    data_v0_10 = run_high_res_single_v0(cfg, optics, v0=10.0)
    
    # Save data
    np.savez(
        "OTF_highres_v0_10.npz",
        **data_v0_10,
        m_waves=cfg.m_waves,
    )
    print("Saved: OTF_highres_v0_10.npz")
    
    # Plot
    plot_theory_comparison(data_v0_10, cfg, save_path="fig_theory_comparison_highres.png")
    
    # Optional: run for multiple v0 values
    # plot_multiple_v0(cfg, optics, v0_list=[5, 10, 20, 40], 
    #                  save_path="fig_theory_multi_v0.png")
    
    # Optional: run full high-res heatmap (takes ~30-60 min)
    # heatmap_data = run_high_res_heatmap(cfg, optics)
    # np.savez("OTF_heatmap_highres.npz", **heatmap_data)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return data_v0_10


if __name__ == "__main__":
    data = main()