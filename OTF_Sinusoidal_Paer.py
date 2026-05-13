# FDPR Sinusoidal Aberration Analysis (Fixed Version)
# =====================================================
# Studies sinusoidal aberrations + OTF behavior under defocus
# Includes Dean & Bowers theory validation and FDPR Monte Carlo
#
# Fixes applied:
# - Increased minimum kill_core_pix from 5 to 20
# - Added peak position validation against predicted location
# - Improved photometry as primary measurement method
# - Better handling of grid boundary cases

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional, List, NamedTuple
from dataclasses import dataclass, field
import time
import warnings

# HCIPy imports
from hcipy import (
    make_pupil_grid, make_focal_grid, make_uniform_grid,
    make_circular_aperture, make_zernike_basis,
    Wavefront, FraunhoferPropagator, Field
)

# Optional imports (with graceful fallback)
try:
    from astropy.io import fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False
    print("Warning: astropy.io.fits not available, FITS output disabled")

try:
    from skimage.transform import resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: skimage not available, FDPR resizing disabled")

try:
    from image_sharpening import FocusDiversePhaseRetrieval, mft_rev, InstrumentConfiguration
    HAS_FDPR = True
except ImportError:
    HAS_FDPR = False
    print("Warning: image_sharpening not available, FDPR disabled")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """Central configuration for the simulation."""
    # Optical system
    wavelength_m: float = 650e-9
    pupil_diameter_m: float = 10.12e-3
    focal_length_m: float = 500e-3
    pupil_npix: int = 256
    q: int = 4
    num_airy: int = 64
    
    # Sinusoidal aberration
    cycles_per_aperture: float = 10.0
    m_waves: float = 0.10  # Peak amplitude in waves
    
    # Single snapshot demo
    single_defocus_mm: float = 100.0
    
    # Defocus sweep
    dz_min_mm: float = 0.5
    dz_max_mm: float = 200.0
    n_defocus: int = 200
    sweep_mode: str = 'linear'  # 'linear', 'log', or 'geometric'
    
    # Heatmap grid
    heatmap_dz_min: float = 5.0
    heatmap_dz_max: float = 250.0
    heatmap_n_dz: int = 80
    heatmap_v0_min: float = 3.0
    heatmap_v0_max: float = 80.0
    heatmap_n_v0: int = 150
    
    # Peak detection parameters (FIXED VALUES)
    min_kill_core_pix: int = 20          # Increased from 5 to avoid DC contamination
    kill_core_scale: float = 0.25        # Scale factor for adaptive kill window
    peak_position_tolerance: float = 0.3  # Max allowed deviation from predicted (30%)
    photometry_aperture_radius: float = 3.0  # Aperture radius in dk units
    
    # FDPR Monte Carlo
    n_trials: int = 1
    read_noise_e: float = 11.0
    fdpr_iterations: int = 200
    
    # Visualization
    log_psf_vmin: float = -5.0
    
    # Derived quantities (computed in __post_init__)
    f_number: float = field(init=False)
    otf_grid_half_size: int = field(init=False)
    
    def __post_init__(self):
        self.f_number = self.focal_length_m / self.pupil_diameter_m
        # OTF grid is same size as focal grid: q * pupil_npix
        self.otf_grid_half_size = (self.q * self.pupil_npix) // 2


# Analysis flags
RUN_SINUSOID = True
RUN_DEAN_BOWERS = True
RUN_FIXED_HEATMAP = True
RUN_SAMPLE_V0 = True
RUN_FDPR_MONTE_CARLO = False  # Disabled by default (slow)
SAVE_FITS = False
VERBOSE = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PeakMeasurement(NamedTuple):
    """Result of OTF side-peak measurement."""
    amplitude: float           # Measured peak amplitude
    offset_px: float           # Measured offset from DC in pixels
    predicted_px: float        # Predicted offset from theory
    is_valid: bool             # Whether measurement passed validation
    method: str                # 'detection' or 'photometry'
    rejection_reason: str      # Empty if valid, otherwise explains why rejected


# =============================================================================
# OPTICAL SYSTEM SETUP
# =============================================================================

def setup_optical_system(cfg: SimulationConfig) -> dict:
    """
    Initialize all optical simulation elements.
    
    Returns dictionary with grids, apertures, propagator, and Zernike basis.
    """
    # Grids
    pupil_grid = make_pupil_grid(cfg.pupil_npix, cfg.pupil_diameter_m)
    focal_grid = make_focal_grid(
        q=cfg.q,
        num_airy=cfg.num_airy,
        pupil_diameter=cfg.pupil_diameter_m,
        focal_length=cfg.focal_length_m,
        reference_wavelength=cfg.wavelength_m
    )
    
    # Apertures
    aperture = make_circular_aperture(cfg.pupil_diameter_m)
    telescope_pupil = aperture(pupil_grid)
    
    # Propagator
    propagator = FraunhoferPropagator(pupil_grid, focal_grid, cfg.focal_length_m)
    
    # Zernike basis (for defocus)
    zernike_basis = make_zernike_basis(
        num_modes=256,
        D=cfg.pupil_diameter_m,
        grid=pupil_grid
    )
    defocus_template = zernike_basis[3]
    
    # Precompute unit defocus (normalized to 1 wave P-V)
    pupil_mask = np.asarray(telescope_pupil.shaped, dtype=bool)
    defocus_shaped = defocus_template.shaped
    template_pv = defocus_shaped[pupil_mask].ptp()
    unit_defocus = defocus_template / template_pv
    
    # Pixel scales
    dtheta = (2 * cfg.num_airy * (cfg.wavelength_m / cfg.pupil_diameter_m)) / focal_grid.shape[0]
    u_hat_per_px = (cfg.wavelength_m / cfg.pupil_diameter_m) / (focal_grid.shape[1] * dtheta)
    delta_x = 2 * cfg.num_airy * (cfg.wavelength_m / cfg.pupil_diameter_m) * cfg.focal_length_m
    dx = delta_x / focal_grid.shape[0]
    delta_k = (2 * np.pi) / dx
    dk = (2 * np.pi) / delta_x
    
    return {
        'pupil_grid': pupil_grid,
        'focal_grid': focal_grid,
        'telescope_pupil': telescope_pupil,
        'pupil_mask': pupil_mask,
        'propagator': propagator,
        'zernike_basis': zernike_basis,
        'defocus_template': defocus_template,
        'unit_defocus': unit_defocus,
        'dtheta': dtheta,
        'u_hat_per_px': u_hat_per_px,
        'delta_x': delta_x,
        'dx': dx,
        'delta_k': delta_k,
        'dk': dk,
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def mm_to_m(x_mm: float) -> float:
    """Convert millimeters to meters."""
    return x_mm * 1e-3


def delta_to_p(delta: float, f: float, D: float) -> float:
    """Convert mechanical defocus to P-V wavefront error."""
    return -delta / (8 * (f / D) ** 2)


def p_to_delta(P: float, f: float, D: float) -> float:
    """Convert P-V wavefront error to mechanical defocus."""
    return 8 * P * (f / D) ** 2


def defocus_a_hat(defocus_mm: float, f_m: float, D_m: float, lam_m: float) -> float:
    """
    Compute normalized defocus (Dean & Bowers â parameter).
    
    Returns â in waves P-V, proportional to mechanical defocus.
    """
    delta_m = abs(defocus_mm) * 1e-3
    return (delta_m * D_m**2) / (8 * f_m**2 * lam_m)


def dz_mm_from_a_hat(a_hat: float, f_m: float, D_m: float, lam_m: float) -> float:
    """Convert â back to mechanical defocus in mm."""
    dz_m = 8.0 * a_hat * f_m**2 * lam_m / D_m**2
    return dz_m * 1e3


def coc_pixels(defocus_mm: float, cfg: SimulationConfig, optics: dict) -> float:
    """
    Compute circle of confusion in pixels for given defocus.
    
    Used to set adaptive DC kill window size.
    """
    c_m = abs(defocus_mm) * 1e-3 / cfg.f_number  # CoC diameter in meters
    return c_m / (cfg.focal_length_m * optics['dtheta'])


def predict_peak_position_px(dz_mm: float, v0: float, cfg: SimulationConfig, 
                              optics: dict) -> float:
    """
    Predict OTF side-peak position in pixels from theory.
    
    Parameters
    ----------
    dz_mm : float
        Defocus in millimeters
    v0 : float
        Spatial frequency in cycles/aperture
    cfg : SimulationConfig
        Configuration
    optics : dict
        Optical system
    
    Returns
    -------
    float
        Predicted peak position in pixels from center
    """
    d_proj = dz_mm * 1e-3 / cfg.f_number  # Projected diameter at image plane
    p_pred = d_proj / v0                   # Period of fringes
    px_pred = optics['delta_x'] / p_pred   # Peak position in pixels
    return px_pred


def is_peak_within_grid(px_pred: float, cfg: SimulationConfig, margin: float = 0.9) -> bool:
    """Check if predicted peak position is within OTF grid bounds."""
    max_offset = cfg.otf_grid_half_size * margin  # Leave some margin
    return px_pred < max_offset


# =============================================================================
# PHASE FUNCTIONS
# =============================================================================

def calculate_defocus_phase(defocus_mm: float, cfg: SimulationConfig, 
                            optics: dict, verbose: bool = False) -> Field:
    """
    Calculate defocus phase from mechanical distance.
    
    Parameters
    ----------
    defocus_mm : float
        Defocus distance in millimeters
    cfg : SimulationConfig
        Simulation configuration
    optics : dict
        Optical system elements from setup_optical_system()
    verbose : bool
        Print debug information
    
    Returns
    -------
    Field
        Defocus phase in radians on pupil grid
    """
    delta_m = defocus_mm * 1e-3
    defocus_pv_m = delta_to_p(delta_m, cfg.focal_length_m, cfg.pupil_diameter_m)
    phase_pv_rad = defocus_pv_m * (2 * np.pi / cfg.wavelength_m)
    
    if verbose:
        print(f"  Defocus P-V: {defocus_pv_m:.3e} m")
        print(f"  Phase P-V: {phase_pv_rad:.2f} rad")
    
    return optics['unit_defocus'] * phase_pv_rad


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
        Spatial frequency (cycles across diameter)
    m_waves : float
        Peak amplitude in waves
    
    Returns
    -------
    Field
        Phase in waves on pupil grid
    """
    x = pupil_grid.x
    phase_waves = m_waves * np.sin(2 * np.pi * cycles_per_aperture * (x / pupil_diameter_m))
    return Field(phase_waves, pupil_grid)


# =============================================================================
# PSF / OTF FUNCTIONS
# =============================================================================

def compute_psf(aberration_rad: Field, optics: dict, cfg: SimulationConfig,
                defocus_rad: Optional[Field] = None) -> np.ndarray:
    """
    Compute PSF from aberration phase.
    
    Parameters
    ----------
    aberration_rad : Field
        Aberration phase in radians
    optics : dict
        Optical system elements
    cfg : SimulationConfig
        Configuration
    defocus_rad : Field, optional
        Additional defocus phase
    
    Returns
    -------
    np.ndarray
        PSF intensity array
    """
    total_phase = aberration_rad
    if defocus_rad is not None:
        total_phase = aberration_rad + defocus_rad
    
    wf = Wavefront(
        optics['telescope_pupil'] * np.exp(1j * total_phase),
        cfg.wavelength_m
    )
    psf = optics['propagator'](wf).intensity.shaped
    return np.asarray(psf)


def compute_otf(psf: np.ndarray) -> np.ndarray:
    """Compute OTF magnitude from PSF via FFT."""
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    return np.abs(otf)


def find_otf_sidepeaks_1D(otf: np.ndarray, kill_core_pix: int = 20,
                          subpixel: bool = True) -> Tuple[float, float, Tuple, Tuple]:
    """
    Find symmetric side peaks in OTF from 1D sinusoidal aberration.
    
    Parameters
    ----------
    otf : np.ndarray
        2D OTF magnitude array
    kill_core_pix : int
        Half-width of DC region to zero out (default increased to 20)
    subpixel : bool
        Use quadratic refinement for subpixel accuracy
    
    Returns
    -------
    offset : float
        Average pixel distance from DC
    amplitude : float
        Average peak amplitude
    offs_lr : tuple
        (left_offset, right_offset)
    amps_lr : tuple
        (left_amplitude, right_amplitude)
    """
    row = otf[otf.shape[0] // 2, :].copy()
    center = len(row) // 2
    
    # Zero out DC core (FIXED: larger default window)
    row[center - kill_core_pix : center + kill_core_pix + 1] = 0.0
    
    # Split and find peaks
    left, right = row[:center], row[center + 1:]
    idx_l, idx_r = int(np.argmax(left)), int(np.argmax(right))
    amp_l, amp_r = float(left[idx_l]), float(right[idx_r])
    
    def refine_peak(idx: int, arr: np.ndarray) -> float:
        """Quadratic subpixel refinement."""
        if not subpixel or idx <= 0 or idx >= len(arr) - 1:
            return float(idx)
        y0, y1, y2 = arr[idx - 1], arr[idx], arr[idx + 1]
        denom = y0 - 2 * y1 + y2
        if denom == 0:
            return float(idx)
        return idx + 0.5 * (y0 - y2) / denom
    
    pos_l = refine_peak(idx_l, left)
    pos_r = refine_peak(idx_r, right) + center + 1
    
    off_l = pos_l - center  # Negative
    off_r = pos_r - center  # Positive
    
    offset = 0.5 * (abs(off_l) + off_r)
    amplitude = 0.5 * (amp_l + amp_r)
    
    return offset, amplitude, (off_l, off_r), (amp_l, amp_r)


def measure_peak_with_validation(otf: np.ndarray, dz_mm: float, v0: float,
                                  cfg: SimulationConfig, optics: dict) -> PeakMeasurement:
    """
    Measure OTF side-peak with validation against predicted position.
    
    This is the PRIMARY measurement function that combines detection
    with validation to reject spurious peaks.
    
    Parameters
    ----------
    otf : np.ndarray
        2D OTF magnitude array
    dz_mm : float
        Defocus in mm
    v0 : float
        Spatial frequency in cycles/aperture
    cfg : SimulationConfig
        Configuration
    optics : dict
        Optical system
    
    Returns
    -------
    PeakMeasurement
        Named tuple with amplitude, offsets, validity, and diagnostics
    """
    # Predict peak position from theory
    px_pred = predict_peak_position_px(dz_mm, v0, cfg, optics)
    
    # Check if peak should be within grid
    if not is_peak_within_grid(px_pred, cfg):
        return PeakMeasurement(
            amplitude=np.nan,
            offset_px=np.nan,
            predicted_px=px_pred,
            is_valid=False,
            method='none',
            rejection_reason=f'Peak outside grid (predicted {px_pred:.1f}px > {cfg.otf_grid_half_size}px)'
        )
    
    # Compute adaptive kill window (FIXED: higher minimum)
    coc = coc_pixels(dz_mm, cfg, optics)
    kill = max(cfg.min_kill_core_pix, int(cfg.kill_core_scale * coc))
    
    # Find peaks
    offset_meas, amp_meas, offs_lr, amps_lr = find_otf_sidepeaks_1D(otf, kill_core_pix=kill)
    
    # Validate: measured position should match predicted within tolerance
    if px_pred > 0:
        position_error = abs(offset_meas - px_pred) / px_pred
    else:
        position_error = np.inf
    
    if position_error > cfg.peak_position_tolerance:
        return PeakMeasurement(
            amplitude=np.nan,
            offset_px=offset_meas,
            predicted_px=px_pred,
            is_valid=False,
            method='detection',
            rejection_reason=f'Position mismatch: measured {offset_meas:.1f}px vs predicted {px_pred:.1f}px ({100*position_error:.0f}% error)'
        )
    
    # Check for asymmetry (suggests contamination from DC or noise)
    amp_asymmetry = abs(amps_lr[0] - amps_lr[1]) / max(amps_lr[0], amps_lr[1], 1e-10)
    if amp_asymmetry > 0.5:  # More than 50% difference
        return PeakMeasurement(
            amplitude=np.nan,
            offset_px=offset_meas,
            predicted_px=px_pred,
            is_valid=False,
            method='detection',
            rejection_reason=f'Asymmetric peaks: L={amps_lr[0]:.0f}, R={amps_lr[1]:.0f} ({100*amp_asymmetry:.0f}% diff)'
        )
    
    return PeakMeasurement(
        amplitude=amp_meas,
        offset_px=offset_meas,
        predicted_px=px_pred,
        is_valid=True,
        method='detection',
        rejection_reason=''
    )


def measure_peak_photometry(otf: np.ndarray, dz_mm: float, v0: float,
                            cfg: SimulationConfig, optics: dict) -> PeakMeasurement:
    """
    Measure OTF peak amplitude using aperture photometry at predicted location.
    
    This method is more robust than peak detection because it measures
    at the theoretically predicted location rather than searching for maxima.
    
    Parameters
    ----------
    otf : np.ndarray
        2D OTF array
    dz_mm : float
        Defocus in mm
    v0 : float
        Spatial frequency in cycles/aperture
    cfg : SimulationConfig
        Configuration
    optics : dict
        Optical system
    
    Returns
    -------
    PeakMeasurement
        Named tuple with measurement results
    """
    # Predict peak position
    px_pred = predict_peak_position_px(dz_mm, v0, cfg, optics)
    
    # Check grid bounds
    if not is_peak_within_grid(px_pred, cfg):
        return PeakMeasurement(
            amplitude=np.nan,
            offset_px=np.nan,
            predicted_px=px_pred,
            is_valid=False,
            method='photometry',
            rejection_reason=f'Peak outside grid (predicted {px_pred:.1f}px)'
        )
    
    # Create aperture at predicted location
    dk = optics['dk']
    delta_k = optics['delta_k']
    
    otf_grid = make_uniform_grid([otf.shape[0], otf.shape[1]], delta_k)
    aperture_radius = cfg.photometry_aperture_radius * dk
    
    # Measure at positive peak location
    peak_aperture = make_circular_aperture(aperture_radius, center=[px_pred * dk, 0])
    mask = peak_aperture(otf_grid) > 0
    
    if not np.any(mask.shaped):
        return PeakMeasurement(
            amplitude=np.nan,
            offset_px=np.nan,
            predicted_px=px_pred,
            is_valid=False,
            method='photometry',
            rejection_reason='Empty aperture mask'
        )
    
    amp_positive = float(otf[mask.shaped].max())
    
    # Also measure at negative peak for symmetry check
    peak_aperture_neg = make_circular_aperture(aperture_radius, center=[-px_pred * dk, 0])
    mask_neg = peak_aperture_neg(otf_grid) > 0
    
    if np.any(mask_neg.shaped):
        amp_negative = float(otf[mask_neg.shaped].max())
        amplitude = 0.5 * (amp_positive + amp_negative)
        
        # Check symmetry
        amp_asymmetry = abs(amp_positive - amp_negative) / max(amp_positive, amp_negative, 1e-10)
        if amp_asymmetry > 0.5:
            return PeakMeasurement(
                amplitude=amplitude,  # Still return value for diagnostics
                offset_px=px_pred,
                predicted_px=px_pred,
                is_valid=False,
                method='photometry',
                rejection_reason=f'Asymmetric: +{amp_positive:.0f}, -{amp_negative:.0f}'
            )
    else:
        amplitude = amp_positive
    
    return PeakMeasurement(
        amplitude=amplitude,
        offset_px=px_pred,
        predicted_px=px_pred,
        is_valid=True,
        method='photometry',
        rejection_reason=''
    )


# =============================================================================
# NOISE MODEL
# =============================================================================

def add_read_noise(image: np.ndarray, sigma_e: float, 
                   rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian read noise to image."""
    noise = rng.normal(scale=sigma_e, size=image.shape)
    return image + noise


# =============================================================================
# PROGRESS UTILITIES
# =============================================================================

class ProgressReporter:
    """Simple progress reporter for long loops."""
    
    def __init__(self, total: int, desc: str = "", report_every: int = 10):
        self.total = total
        self.desc = desc
        self.report_every = report_every
        self.start_time = time.time()
        self.current = 0
    
    def update(self, n: int = 1):
        self.current += n
        if self.current % self.report_every == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            print(f"  {self.desc}: {self.current}/{self.total} "
                  f"({100*self.current/self.total:.1f}%) "
                  f"[{elapsed:.1f}s elapsed, ~{eta:.1f}s remaining]")
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"  {self.desc}: Complete in {elapsed:.1f}s")


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def run_defocus_sweep(cfg: SimulationConfig, optics: dict) -> dict:
    """
    Run defocus sweep analysis for fixed sinusoidal aberration.
    
    Returns dictionary with PSFs, OTFs, and measured quantities.
    """
    print("\n" + "=" * 60)
    print("DEFOCUS SWEEP ANALYSIS")
    print("=" * 60)
    
    # Generate defocus values
    if cfg.sweep_mode == 'linear':
        sweep_dz = np.linspace(cfg.dz_min_mm, cfg.dz_max_mm, cfg.n_defocus)
    elif cfg.sweep_mode == 'log':
        sweep_dz = np.logspace(np.log10(cfg.dz_min_mm), np.log10(cfg.dz_max_mm), cfg.n_defocus)
    else:  # geometric
        sweep_dz = np.geomspace(cfg.dz_min_mm, cfg.dz_max_mm, cfg.n_defocus)
    
    print(f"  Spatial frequency: {cfg.cycles_per_aperture} cycles/aperture")
    print(f"  Amplitude: {cfg.m_waves} waves")
    print(f"  Defocus range: {sweep_dz.min():.1f} - {sweep_dz.max():.1f} mm ({cfg.sweep_mode})")
    
    # Precompute sinusoidal aberration (constant across sweep)
    phi_sine_waves = make_sinusoidal_phase_waves(
        optics['pupil_grid'], cfg.pupil_diameter_m,
        cfg.cycles_per_aperture, cfg.m_waves
    )
    phi_sine_rad = 2 * np.pi * phi_sine_waves
    
    # Storage
    focal_shape = optics['focal_grid'].shape
    psf_stack = np.empty((len(sweep_dz), focal_shape[0], focal_shape[1]))
    otf_stack = np.empty_like(psf_stack)
    
    # Use validated measurements
    amplitudes_detection = []
    amplitudes_photometry = []
    offsets_measured = []
    offsets_predicted = []
    valid_detection = []
    valid_photometry = []
    a_hats = []
    
    # Progress
    progress = ProgressReporter(len(sweep_dz), "Defocus sweep", report_every=20)
    
    for i, dz_mm in enumerate(sweep_dz):
        # Compute defocus phase
        phi_def = calculate_defocus_phase(dz_mm, cfg, optics, verbose=False)
        
        # Compute PSF and OTF
        psf = compute_psf(phi_sine_rad, optics, cfg, defocus_rad=phi_def)
        otf = compute_otf(psf)
        
        psf_stack[i] = psf
        otf_stack[i] = otf
        
        # Measure with validation
        meas_det = measure_peak_with_validation(otf, dz_mm, cfg.cycles_per_aperture, cfg, optics)
        meas_photo = measure_peak_photometry(otf, dz_mm, cfg.cycles_per_aperture, cfg, optics)
        
        amplitudes_detection.append(meas_det.amplitude if meas_det.is_valid else np.nan)
        amplitudes_photometry.append(meas_photo.amplitude if meas_photo.is_valid else np.nan)
        offsets_measured.append(meas_det.offset_px)
        offsets_predicted.append(meas_det.predicted_px)
        valid_detection.append(meas_det.is_valid)
        valid_photometry.append(meas_photo.is_valid)
        
        # Dean & Bowers â parameter
        a_hat = defocus_a_hat(dz_mm, cfg.focal_length_m, cfg.pupil_diameter_m, cfg.wavelength_m)
        a_hats.append(a_hat)
        
        progress.update()
    
    progress.finish()
    
    # Report validation statistics
    n_valid_det = sum(valid_detection)
    n_valid_photo = sum(valid_photometry)
    print(f"  Validation: {n_valid_det}/{len(sweep_dz)} detection, {n_valid_photo}/{len(sweep_dz)} photometry")
    
    # Also compute focused PSF (no defocus)
    wf_focus = Wavefront(
        optics['telescope_pupil'] * np.exp(1j * phi_sine_rad),
        cfg.wavelength_m
    )
    psf_focus = np.asarray(optics['propagator'](wf_focus).intensity.shaped)
    
    return {
        'sweep_dz': sweep_dz,
        'psf_stack': psf_stack,
        'otf_stack': otf_stack,
        'amplitudes_detection': np.array(amplitudes_detection),
        'amplitudes_photometry': np.array(amplitudes_photometry),
        'offsets_measured': np.array(offsets_measured),
        'offsets_predicted': np.array(offsets_predicted),
        'valid_detection': np.array(valid_detection),
        'valid_photometry': np.array(valid_photometry),
        'a_hats': np.array(a_hats),
        'psf_focus': psf_focus,
        'phi_sine_rad': phi_sine_rad,
    }


def run_heatmap_analysis(cfg: SimulationConfig, optics: dict) -> dict:
    """
    Compute OTF side-peak amplitude heatmap over (defocus, spatial frequency).
    
    Returns dictionary with heatmap data.
    """
    print("\n" + "=" * 60)
    print("HEATMAP ANALYSIS")
    print("=" * 60)
    
    dz_values = np.linspace(cfg.heatmap_dz_min, cfg.heatmap_dz_max, cfg.heatmap_n_dz)
    v0_values = np.linspace(cfg.heatmap_v0_min, cfg.heatmap_v0_max, cfg.heatmap_n_v0)
    
    print(f"  Grid: {len(dz_values)} x {len(v0_values)} = {len(dz_values)*len(v0_values)} points")
    print(f"  Defocus: {dz_values.min():.1f} - {dz_values.max():.1f} mm")
    print(f"  Frequency: {v0_values.min():.1f} - {v0_values.max():.1f} cycles/ap")
    print(f"  Min kill_core_pix: {cfg.min_kill_core_pix}")
    print(f"  Position tolerance: {100*cfg.peak_position_tolerance:.0f}%")
    
    # Storage
    H_detection = np.full((len(dz_values), len(v0_values)), np.nan)
    H_photometry = np.full((len(dz_values), len(v0_values)), np.nan)
    H_valid_detection = np.zeros((len(dz_values), len(v0_values)), dtype=bool)
    H_valid_photometry = np.zeros((len(dz_values), len(v0_values)), dtype=bool)
    
    # Precompute defocus phases
    print("  Precomputing defocus phases...")
    defocus_phases = {
        dz: calculate_defocus_phase(dz, cfg, optics, verbose=False)
        for dz in dz_values
    }
    
    # Precompute sinusoidal phases
    print("  Precomputing sinusoidal phases...")
    sine_phases = {
        v0: 2 * np.pi * make_sinusoidal_phase_waves(
            optics['pupil_grid'], cfg.pupil_diameter_m, v0, cfg.m_waves
        )
        for v0 in v0_values
    }
    
    # Track rejection reasons for diagnostics
    rejection_counts = {
        'outside_grid': 0,
        'position_mismatch': 0,
        'asymmetric': 0,
        'other': 0,
    }
    
    # Main loop
    progress = ProgressReporter(len(dz_values), "Heatmap rows", report_every=10)
    
    for i, dz in enumerate(dz_values):
        phi_def = defocus_phases[dz]
        
        for j, v0 in enumerate(v0_values):
            phi_sine_rad = sine_phases[v0]
            
            # Compute PSF and OTF
            psf = compute_psf(phi_sine_rad, optics, cfg, defocus_rad=phi_def)
            otf = compute_otf(psf)
            
            # Method 1: Validated peak detection
            meas_det = measure_peak_with_validation(otf, dz, v0, cfg, optics)
            H_detection[i, j] = meas_det.amplitude
            H_valid_detection[i, j] = meas_det.is_valid
            
            # Method 2: Photometry (primary method)
            meas_photo = measure_peak_photometry(otf, dz, v0, cfg, optics)
            H_photometry[i, j] = meas_photo.amplitude
            H_valid_photometry[i, j] = meas_photo.is_valid
            
            # Track rejections
            if not meas_photo.is_valid:
                reason = meas_photo.rejection_reason.lower()
                if 'outside' in reason:
                    rejection_counts['outside_grid'] += 1
                elif 'position' in reason or 'mismatch' in reason:
                    rejection_counts['position_mismatch'] += 1
                elif 'asymmetric' in reason:
                    rejection_counts['asymmetric'] += 1
                else:
                    rejection_counts['other'] += 1
        
        progress.update()
    
    progress.finish()
    
    # Report statistics
    n_total = len(dz_values) * len(v0_values)
    n_valid_det = np.sum(H_valid_detection)
    n_valid_photo = np.sum(H_valid_photometry)
    print(f"  Valid measurements:")
    print(f"    Detection: {n_valid_det}/{n_total} ({100*n_valid_det/n_total:.1f}%)")
    print(f"    Photometry: {n_valid_photo}/{n_total} ({100*n_valid_photo/n_total:.1f}%)")
    print(f"  Photometry rejections:")
    for reason, count in rejection_counts.items():
        if count > 0:
            print(f"    {reason}: {count}")
    
    return {
        'dz_values': dz_values,
        'v0_values': v0_values,
        'H_detection': H_detection,
        'H_photometry': H_photometry,
        'H_valid_detection': H_valid_detection,
        'H_valid_photometry': H_valid_photometry,
        # For backwards compatibility
        'H': H_photometry,  # Use photometry as primary
    }


def run_fdpr_monte_carlo(cfg: SimulationConfig, optics: dict,
                         dz_values: np.ndarray, v0_values: np.ndarray) -> dict:
    """
    Run FDPR Monte Carlo analysis over (defocus, frequency) grid.
    
    Parameters
    ----------
    cfg : SimulationConfig
        Configuration
    optics : dict
        Optical system
    dz_values : np.ndarray
        Defocus values in mm
    v0_values : np.ndarray
        Spatial frequencies in cycles/aperture
    
    Returns
    -------
    dict
        RMS statistics over trials
    """
    if not HAS_FDPR:
        print("FDPR not available, skipping Monte Carlo analysis")
        return None
    
    if not HAS_SKIMAGE:
        print("skimage not available for resizing, skipping FDPR")
        return None
    
    print("\n" + "=" * 60)
    print("FDPR MONTE CARLO ANALYSIS")
    print("=" * 60)
    print(f"  Trials per point: {cfg.n_trials}")
    print(f"  Read noise: {cfg.read_noise_e} e-/px")
    print(f"  FDPR iterations: {cfg.fdpr_iterations}")
    
    # Setup
    rng = np.random.default_rng(12345)
    Nd, Nv = len(dz_values), len(v0_values)
    rms_mean = np.full((Nd, Nv), np.nan)
    rms_std = np.full((Nd, Nv), np.nan)
    
    # FDPR configuration
    seal_param_config = {
        'image_dx': 2.0071,
        'efl': cfg.focal_length_m * 1e3,
        'wavelength': cfg.wavelength_m * 1e6,  # microns
        'pupil_size': cfg.pupil_diameter_m * 1e3,
    }
    conf = InstrumentConfiguration(seal_param_config)
    
    total_points = Nd * Nv
    progress = ProgressReporter(total_points, "FDPR Monte Carlo", report_every=50)
    
    for i, dz in enumerate(dz_values):
        phi_def = calculate_defocus_phase(dz, cfg, optics, verbose=False)
        
        for j, v0 in enumerate(v0_values):
            phi_sine_waves = make_sinusoidal_phase_waves(
                optics['pupil_grid'], cfg.pupil_diameter_m, v0, cfg.m_waves
            )
            phi_sine_rad = 2 * np.pi * phi_sine_waves
            
            # Clean PSFs
            psf_focus_clean = compute_psf(phi_sine_rad, optics, cfg)
            psf_defoc_clean = compute_psf(phi_sine_rad, optics, cfg, defocus_rad=phi_def)
            
            rms_trials = []
            
            for t in range(cfg.n_trials):
                # Add noise
                psf_focus_noisy = add_read_noise(psf_focus_clean, cfg.read_noise_e, rng)
                psf_defoc_noisy = add_read_noise(psf_defoc_clean, cfg.read_noise_e, rng)
                
                # Run FDPR
                psf_list = [psf_focus_noisy, psf_defoc_noisy]
                dx_list = [2.0071]
                
                mp = FocusDiversePhaseRetrieval(
                    psf_list, cfg.wavelength_m * 1e6, dx_list, [dz]
                )
                
                for _ in range(cfg.fdpr_iterations):
                    psf_result = mp.step()
                
                # Extract phase
                raw_pupil = np.angle(mft_rev(psf_result, conf))
                real_pupil = resize(raw_pupil, (256, 256)) * optics['telescope_pupil'].shaped
                
                # Compute RMS
                masked = real_pupil[optics['pupil_mask']]
                rms_nm = np.sqrt(np.mean(masked**2)) * (cfg.wavelength_m * 1e9 / (2 * np.pi))
                rms_trials.append(rms_nm)
            
            rms_mean[i, j] = np.nanmean(rms_trials)
            rms_std[i, j] = np.nanstd(rms_trials)
            
            progress.update()
    
    progress.finish()
    
    return {
        'dz_values': dz_values,
        'v0_values': v0_values,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_otf_rows_vs_defocus(sweep_results: dict, cfg: SimulationConfig, 
                              optics: dict, n_curves: int = 10):
    """Plot OTF central rows for selected defocus values."""
    sweep_dz = sweep_results['sweep_dz']
    otf_stack = sweep_results['otf_stack']
    
    # Select defocus values (log-spaced for visibility)
    dz_show = np.logspace(np.log10(sweep_dz.min()), np.log10(sweep_dz.max()), n_curves)
    
    norm = plt.Normalize(vmin=sweep_dz.min(), vmax=sweep_dz.max())
    cmap = cm.viridis
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for dz in dz_show:
        i = np.argmin(np.abs(sweep_dz - dz))
        otf = otf_stack[i]
        row = otf[otf.shape[0] // 2, :].copy()
        
        # Frequency axis
        k = np.arange(row.size) - row.size // 2
        vhat_axis = k * optics['u_hat_per_px']
        
        ax.plot(vhat_axis, row, lw=1.2, color=cmap(norm(sweep_dz[i])))
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Defocus Δz [mm]')
    
    ax.set_xlim(-0.35, 0.35)
    ax.set_xlabel('Normalized spatial frequency v̂')
    ax.set_ylabel('OTF magnitude')
    ax.set_title(f'OTF central row vs defocus (v₀={cfg.cycles_per_aperture} cyc/ap)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_heatmap(heatmap_results: dict, cfg: SimulationConfig, 
                 method: str = 'photometry', show_theory: bool = True):
    """
    Plot OTF amplitude heatmap.
    
    Parameters
    ----------
    method : str
        'photometry' or 'detection'
    show_theory : bool
        Overlay Dean & Bowers theoretical curve
    """
    dz = heatmap_results['dz_values']
    v0 = heatmap_results['v0_values']
    
    if method == 'photometry':
        H = heatmap_results['H_photometry']
        valid = heatmap_results['H_valid_photometry']
        title_suffix = "(photometry)"
    else:
        H = heatmap_results['H_detection']
        valid = heatmap_results['H_valid_detection']
        title_suffix = "(peak detection)"
    
    # Mask invalid measurements
    H_masked = np.where(valid, H, np.nan)
    
    plt.figure(figsize=(10, 8))
    extent = [v0.min(), v0.max(), dz.min(), dz.max()]
    
    # Use percentile clipping for color scale
    valid_vals = H_masked[np.isfinite(H_masked)]
    if len(valid_vals) > 0:
        vmin, vmax = np.nanpercentile(valid_vals, [2, 98])
    else:
        vmin, vmax = 0, 1
    
    plt.imshow(H_masked, aspect='auto', origin='lower', extent=extent,
               cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='OTF amplitude')
    
    # Overlay theory if requested
    if show_theory:
        v0_theory = np.linspace(v0.min(), v0.max(), 200)
        a_optimal = v0_theory**2 / 4  # Dean & Bowers
        dz_optimal = dz_mm_from_a_hat(a_optimal, cfg.focal_length_m, 
                                       cfg.pupil_diameter_m, cfg.wavelength_m)
        # Only plot within our dz range
        mask = dz_optimal <= dz.max()
        if np.any(mask):
            plt.plot(v0_theory[mask], dz_optimal[mask], 'r--', lw=2, 
                    label='Theory: â = v₀²/4')
            plt.legend(loc='upper left')
    
    plt.xlabel("Spatial frequency v₀ [cycles/aperture]")
    plt.ylabel("Defocus Δz [mm]")
    plt.title(f"OTF side-peak amplitude {title_suffix}")
    plt.tight_layout()
    plt.show()


def plot_heatmap_comparison(heatmap_results: dict, cfg: SimulationConfig):
    """Plot both detection and photometry heatmaps side by side."""
    dz = heatmap_results['dz_values']
    v0 = heatmap_results['v0_values']
    extent = [v0.min(), v0.max(), dz.min(), dz.max()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, method in zip(axes, ['detection', 'photometry']):
        if method == 'photometry':
            H = heatmap_results['H_photometry']
            valid = heatmap_results['H_valid_photometry']
        else:
            H = heatmap_results['H_detection']
            valid = heatmap_results['H_valid_detection']
        
        H_masked = np.where(valid, H, np.nan)
        valid_vals = H_masked[np.isfinite(H_masked)]
        
        if len(valid_vals) > 0:
            vmin, vmax = np.nanpercentile(valid_vals, [2, 98])
        else:
            vmin, vmax = 0, 1
        
        im = ax.imshow(H_masked, aspect='auto', origin='lower', extent=extent,
                       cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='OTF amplitude')
        
        n_valid = np.sum(valid)
        n_total = valid.size
        ax.set_xlabel("v₀ [cycles/aperture]")
        ax.set_ylabel("Δz [mm]")
        ax.set_title(f"{method.capitalize()}\n({n_valid}/{n_total} valid, {100*n_valid/n_total:.0f}%)")
    
    plt.tight_layout()
    plt.show()


def plot_dean_bowers_theory(cfg: SimulationConfig):
    """Plot Dean & Bowers theoretical curve."""
    v0 = cfg.cycles_per_aperture
    
    # â sweep
    a_hat = np.linspace(0.001, 50, 800)
    
    # Theory: OTF side-peak amplitude ~ -sin(π v₀² / 8â)
    y = -np.sin(np.pi * v0**2 / (8.0 * a_hat))
    
    # Reversal extrema at â_n = v₀² / [4(2n-1)]
    n_vals = np.arange(1, 5)
    a_rev = v0**2 / (4.0 * (2 * n_vals - 1))
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_hat, y, lw=2.0)
    ax.set_xlim(2, 16)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('â (defocus in waves P-V)')
    ax.set_ylabel('OTF amplitude factor')
    ax.set_title(f'Dean & Bowers: Maximum diversity defocus (v₀={v0:.0f} cyc/ap)')
    
    # Mark first reversal
    a_max = v0**2 / 4.0
    ax.axvline(a_max, color='red', linestyle='--', label=f'â = v₀²/4 = {a_max:.1f}')
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Secondary axis: mechanical defocus
    secax = ax.secondary_xaxis('top', functions=(
        lambda a: dz_mm_from_a_hat(a, cfg.focal_length_m, cfg.pupil_diameter_m, cfg.wavelength_m),
        lambda dz: defocus_a_hat(dz, cfg.focal_length_m, cfg.pupil_diameter_m, cfg.wavelength_m)
    ))
    secax.set_xlabel('Defocus Δz [mm]')
    
    plt.tight_layout()
    plt.show()


def plot_psf_comparison(sweep_results: dict, cfg: SimulationConfig):
    """Plot focused vs defocused PSF comparison."""
    psf_focus = sweep_results['psf_focus']
    
    # Find PSF at single_defocus_mm
    sweep_dz = sweep_results['sweep_dz']
    idx = np.argmin(np.abs(sweep_dz - cfg.single_defocus_mm))
    psf_defoc = sweep_results['psf_stack'][idx]
    actual_dz = sweep_dz[idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(np.log10(psf_focus + 1e-12), vmin=cfg.log_psf_vmin, cmap='inferno')
    axes[0].set_title(f"Focused PSF\n(v₀={cfg.cycles_per_aperture} cyc/ap, m={cfg.m_waves} waves)")
    axes[0].axis('off')
    
    axes[1].imshow(np.log10(psf_defoc + 1e-12), vmin=cfg.log_psf_vmin, cmap='inferno')
    axes[1].set_title(f"Defocused PSF\n(Δz = {actual_dz:.1f} mm)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_amplitude_vs_defocus(sweep_results: dict):
    """Plot OTF side-peak amplitude vs defocus, comparing methods."""
    dz = sweep_results['sweep_dz']
    amp_det = sweep_results['amplitudes_detection']
    amp_photo = sweep_results['amplitudes_photometry']
    valid_det = sweep_results['valid_detection']
    valid_photo = sweep_results['valid_photometry']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Both methods
    ax = axes[0]
    ax.plot(dz[valid_det], amp_det[valid_det], 'b.-', lw=1, ms=3, 
            alpha=0.7, label='Detection (valid)')
    ax.plot(dz[valid_photo], amp_photo[valid_photo], 'r.-', lw=1, ms=3,
            alpha=0.7, label='Photometry (valid)')
    ax.set_xlabel("Defocus Δz [mm]")
    ax.set_ylabel("OTF side-peak amplitude")
    ax.set_title("Amplitude vs defocus (both methods)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Measured vs predicted offset
    ax = axes[1]
    off_meas = sweep_results['offsets_measured']
    off_pred = sweep_results['offsets_predicted']
    valid = np.isfinite(off_meas) & np.isfinite(off_pred)
    ax.scatter(off_pred[valid], off_meas[valid], s=10, alpha=0.5)
    ax.plot([0, off_pred[valid].max()], [0, off_pred[valid].max()], 'r--', label='1:1 line')
    ax.set_xlabel("Predicted offset [px]")
    ax.set_ylabel("Measured offset [px]")
    ax.set_title("Peak position: measured vs predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_optimal_defocus_histogram(heatmap_results: dict):
    """Plot histogram of optimal defocus per spatial frequency."""
    dz = heatmap_results['dz_values']
    v0 = heatmap_results['v0_values']
    H = heatmap_results['H_photometry']
    valid = heatmap_results['H_valid_photometry']
    
    H_masked = np.where(valid, H, np.nan)
    
    # Find optimal dz for each v0
    optimal_dz = []
    for j in range(H_masked.shape[1]):
        column = H_masked[:, j]
        if np.any(np.isfinite(column)):
            i_max = np.nanargmax(column)
            optimal_dz.append(dz[i_max])
        else:
            optimal_dz.append(np.nan)
    
    optimal_dz = np.array(optimal_dz)
    valid_opt = np.isfinite(optimal_dz)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(optimal_dz[valid_opt], bins=15, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel("Optimal defocus Δz [mm]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of optimal defocus values")
    axes[0].grid(True, alpha=0.3)
    
    # Optimal dz vs v0
    axes[1].scatter(v0[valid_opt], optimal_dz[valid_opt], s=15, alpha=0.7, label='Measured')
    
    # Theory overlay
    v0_theory = np.linspace(v0.min(), v0.max(), 200)
    a_optimal = v0_theory**2 / 4
    dz_optimal_theory = dz_mm_from_a_hat(
        a_optimal, 0.5, 10.12e-3, 650e-9  # Use default params
    )
    mask = dz_optimal_theory <= dz.max() * 1.1
    if np.any(mask):
        axes[1].plot(v0_theory[mask], dz_optimal_theory[mask], 'r-', lw=2, label='Theory')
    
    axes[1].set_xlabel("Spatial frequency v₀ [cycles/aperture]")
    axes[1].set_ylabel("Optimal defocus Δz [mm]")
    axes[1].set_title("Optimal defocus vs spatial frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("SINUSOIDAL ABERRATION + OTF ANALYSIS (FIXED VERSION)")
    print("=" * 70)

    # Configuration
    cfg = SimulationConfig()
    
    # Setup optical system
    print("\nInitializing optical system...")
    optics = setup_optical_system(cfg)
    print("  Done.")
    
    # Print key parameters
    print(f"\n  F/# = {cfg.f_number:.1f}")
    print(f"  OTF grid half-size = {cfg.otf_grid_half_size} px")
    print(f"  â range: {defocus_a_hat(cfg.heatmap_dz_min, cfg.focal_length_m, cfg.pupil_diameter_m, cfg.wavelength_m):.2f} - "
          f"{defocus_a_hat(cfg.heatmap_dz_max, cfg.focal_length_m, cfg.pupil_diameter_m, cfg.wavelength_m):.2f} waves")
    
    # Run analyses based on flags
    sweep_results = None
    heatmap_results = None
    
    if RUN_SINUSOID:
        sweep_results = run_defocus_sweep(cfg, optics)
        
        # Plots
        plot_psf_comparison(sweep_results, cfg)
        plot_amplitude_vs_defocus(sweep_results)
        plot_otf_rows_vs_defocus(sweep_results, cfg, optics)
    
    if RUN_DEAN_BOWERS:
        plot_dean_bowers_theory(cfg)
    
    if RUN_FIXED_HEATMAP:
        heatmap_results = run_heatmap_analysis(cfg, optics)
        
        # Save
        np.savez(
            "OTF_heatmap_data_fixed.npz",
            H=heatmap_results['H'],
            H_detection=heatmap_results['H_detection'],
            H_photometry=heatmap_results['H_photometry'],
            H_valid_detection=heatmap_results['H_valid_detection'],
            H_valid_photometry=heatmap_results['H_valid_photometry'],
            fixed_dz_heatmap=heatmap_results['dz_values'],
            v0_heatmap=heatmap_results['v0_values'],
            m_waves=cfg.m_waves,
            min_kill_core_pix=cfg.min_kill_core_pix,
            peak_position_tolerance=cfg.peak_position_tolerance,
        )
        print("Saved: OTF_heatmap_data_fixed.npz")
        
        # Plots
        plot_heatmap_comparison(heatmap_results, cfg)
        plot_heatmap(heatmap_results, cfg, method='photometry', show_theory=True)
    
    if RUN_SAMPLE_V0 and heatmap_results is not None:
        plot_optimal_defocus_histogram(heatmap_results)
    
    if RUN_FDPR_MONTE_CARLO and heatmap_results is not None:
        fdpr_results = run_fdpr_monte_carlo(
            cfg, optics,
            heatmap_results['dz_values'],
            heatmap_results['v0_values']
        )
        
        if fdpr_results is not None:
            # Plot FDPR results
            extent = [
                heatmap_results['v0_values'].min(),
                heatmap_results['v0_values'].max(),
                heatmap_results['dz_values'].min(),
                heatmap_results['dz_values'].max()
            ]
            
            plt.figure(figsize=(9, 7))
            plt.imshow(fdpr_results['rms_mean'], aspect='auto', origin='lower',
                       extent=extent, cmap='magma_r')
            plt.colorbar(label="FDPR phase RMS [nm]")
            plt.xlabel("Spatial frequency v₀ [cycles/aperture]")
            plt.ylabel("Defocus Δz [mm]")
            plt.title(f"FDPR Monte Carlo: Mean RMS (N={cfg.n_trials})")
            plt.tight_layout()
            plt.show()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)



if __name__ == "__main__":
    main()
    