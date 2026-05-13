#FDPR_sinusoidal
#Sinusoidal Aberration Scriot
# ===========================
# SINUSOIDAL ABERRATION + OTF STUDY (10 cyc/ap)
# ===========================
import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from skimage.restoration import unwrap_phase
from skimage.transform import resize
from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
from processing import phase_unwrap_2d
from astropy.io import fits
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

'''
Simulation Setup
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

'''
This dictionary is specifically for Instrument Configuration Class
'''
seal_param_config = {'image_dx': 2.0071, # 
               'efl': seal_parameters['focal_length_meters']*1e3, # SEAL effective focal length, mm
               'wavelength': 0.65, # SEAL center wavelength, microns
                'pupil_size': seal_parameters['pupil_size']*1e3, # Keck entrance pupil diameter
                    }
conf = InstrumentConfiguration(seal_param_config)
'''
Now We build the simulation elements
'''
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
wavefront = Wavefront(telescope_pupil,
                      seal_parameters['wavelength_meter']
                      )
prop_p2f = FraunhoferPropagator(pupil_grid,
                                focal_grid,
                                seal_parameters['focal_length_meters']
                                )
pupil_image = wavefront.copy()
focal_image = prop_p2f.forward(wavefront)
perfect_focal = focal_image.copy()
zernike_modes = make_zernike_basis(
         num_modes=256,
         D=seal_parameters['pupil_size'],
         grid=pupil_grid
    )
defocus_template = zernike_modes[3]
'''
Helper Functions
'''
# Convert phase into meters using the wavelength
def phase_to_m(phase, wv):
    return phase * wv / (2 * np.pi)

# Calculate defocus distance from peak-to-valley (P2V) error in meters
def p_to_delta(P, f, D):
    return 8 * P * (f/D)**2

# Convert defocus distance into phase error
def delta_to_p(delta, f, D):
    return -1 * delta / (8 * (f/D)**2)

#Find Defocus Phase from Distance
def calculate_defocus_phase(seal_parameters,
                            defocus_distance):
    mask = np.array(telescope_pupil.shaped, dtype=bool)
    defocus_template_s=defocus_template.shaped
    template_p2v=defocus_template_s[mask].max() - defocus_template_s[mask].min()
    unit_defocus = defocus_template / template_p2v # normalizes zernike defocus to 1 P2V unit amp
    delta_m = defocus_distance * 1e-3 # [m]
    defocus_p2v = delta_to_p(
                            delta = delta_m,
                            f = seal_parameters['focal_length_meters'],
                            D=seal_parameters['pupil_size']
                            )
    phase_p2v = defocus_p2v * (2*np.pi/seal_parameters['wavelength_meter'])
    defocus_phase = unit_defocus * phase_p2v
    print('defocus_p2v value:',defocus_p2v)
    print('phase_p2v value', phase_p2v)
    return defocus_phase

    # Sinusoid Function Helpers
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
        # Every PSF looks “equally bright” at the core
        return np.asarray(I)

def otf_from_psf_numpy(psf):
        """
        We input psf as a real, normalized focal plane intensity array returned by psf_from_wavefront.
        OTF is the fourier transform of the PSF, and the DC component is the middle of the OTF after fftshift.
        Images are stored with DC at the center for visualization, ifftshift moves the DC to the array origin, fft2 transforms, 
        than fftshift moves the OTF DC back to center to help with finding the peak we wanted(not 100% sure on the math here.)
        
        We take the magnitude and normalize. 
        """
        OTF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        #Note her, PSFs want to be graphed in log, OTF grpahed with linear
        mag = np.abs(OTF)
        #mag = mag / mag.max() if mag.max() > 0 else mag
        #fourier mag normalized so peak =1
        #Removes the fact that even the DC component weakens with defocus.
        #Good for comparing relative transfer of spatial frequencies (since transfer functions are often normalized to DC = 1 by definition).
        #bad if we want to show absolute contrast loss as defocus grows.
        return mag

def find_otf_sidepeaks_1D(OTF, kill_core_pix=9, subpixel=True):
        """
        we have 2d OTF magnitude image, the FFT of the PSf. We find the two symetric side peaks that the single
        sinusoidal pupil aberration does in the OTF. look along the central horizontal row(y=0), since injection 
        is 1-d in x. 

        OTF : 2-D np array, peak normalized s.t DC is center
        kill_core_pix : integer window in px around DC set to 0 b4 finding peak, scaled with defocus
        subpixel: 3-point refinement to estimate sub-ipixel peak locations
        """
        row = OTF[OTF.shape[0] // 2, :].copy() # extract the central y=0 row
        c = len(row) // 2 # find center index
        row[c-kill_core_pix:c+kill_core_pix+1] = 0.0 # remove DC bias
        left, right = row[:c], row[c+1:] # split halves
        il, ir = int(np.argmax(left)), int(np.argmax(right)) # local maxima
        amp_l, amp_r = float(left[il]), float(right[ir]) # 

        '''
        Looked this up, not 100% sure about it
        '''
        def refine(i, a):
            if not subpixel or i <= 0 or i >= len(a)-1: return float(i)
            y0, y1, y2 = a[i-1], a[i], a[i+1]
            d = (y0 - 2*y1 + y2)
            return i + 0.5*(y0 - y2)/d if d != 0 else float(i)

        pos_l = refine(il, left)
        pos_r = refine(ir, right) + c + 1
        off_l = -(c - pos_l); off_r = (pos_r - c) # negative/poistive : left/right peaks
        offset = 0.5*(abs(off_l)+abs(off_r)) # ag pixel distance from DC
        amp = 0.5*(amp_l+amp_r) # avg peak height
        return float(offset), float(amp), (float(off_l), float(off_r)), (amp_l, amp_r)
    
def f_number(f_m, D_m):
        '''
        computes f-number, which sets the beam cone angle and connects the mechanical defocus to image-plane blur
        '''
        return f_m / D_m

def projected_pupil_diameter_mm(defocus_mm, f_m, D_m):
         '''
         estimating image-plane radius of the defocused area, from given dx. 
         
         returns diamter of blur circle in mm at image plane
         will tell how wide PSF core gets as we move through dz
         works with DC kill window 

         ''' 
         N= f_number(f_m, D_m)
         return abs(defocus_mm) / N

def angular_pixel_scale_rad(focal_grid, lam_m, D_m, num_airy):
        '''
        computes the angular pixel scales dtheta of focal grid

        returns dtheta in radians/pixel
        '''
        ny, nx = focal_grid.shape
        return (2.0 * num_airy * (lam_m / D_m)) / nx

def coc_pixels(defocus_mm, focal_grid, seal_parameters):
        '''
        finds the circle of confusion in pixels on focal grid for given dz. 

        returns the diamter in px of defocused area. 
        works with kill_core_pix to be able to proporitionally change kill window for any dz

        '''
        f_m = seal_parameters['focal_length_meters']
        D_m = seal_parameters['pupil_size']
        lam = seal_parameters['wavelength_meter']
        num_airy = seal_parameters['Num_airycircles']
        N = f_number(f_m, D_m)
        c_m = abs(defocus_mm) * 1e-3 / N                 # circle-of-confusion diameter at the image plane [m]
        dtheta = angular_pixel_scale_rad(focal_grid, lam, D_m, num_airy)
        return c_m / (f_m * dtheta)                      # pixels of focal grid
    
def dz_mm_from_a_hat(a_hat, f_m, D_m, lam_m):
        dz_m = 8.0 * a_hat * (f_m**2) * lam_m / (D_m**2)
        return 1e3 * dz_m  # mm
    
def defocus_a_hat(defocus_mm, f_m, D_m, lam_m):
        '''
        COmputes a, which is the normalized defoucs PV in waves used by Dean/Bowers. 
        returns scalar a, which is proportional to dz. ie bigger dz, bigger a'''
        delta_m = abs(defocus_mm)* 1e-3
        return (delta_m * (D_m**2)) / (8*(f_m**2)*lam_m)
    # helper: map a_hat <-> dz_mm for a top axis
def a_to_dz_mm(a):
    # vectorized using conversion
        return dz_mm_from_a_hat(a, f_m, D_m, lam_m)

def dz_mm_to_a(dz):
    # inverse of the above;  already have defocus_a_hat()
        return defocus_a_hat(dz, f_m, D_m, lam_m)


'''
Sinusoidal Injection
'''
run_sinusoid = True
if run_sinusoid:
    # Adjustable
    cycles_per_aperture = 10.0        # set injected spatial frequency to _ cycles/aperture
    m_waves = 0.10              # peak amplitude of the sinusoid in waves
    single_defocus_mm = 100.0       # one snapshot defocus [mm] for PSF/OTF demo
    #sweep_defocus_mm = np.linspace(1, 100.0, 200)  # defocus sweep [mm] for tracking the OTF, 
                                                 # CANNOT INCLUDE ZERO IF USING FDPR
    '''
    Thinking about doing geometric sweep because curves would become smoother and more revealing as dz grows
    '''
    dz_min_mm, dz_max_mm, N_dz = (0.5, 200.0, 200)
    linear=True
    logarithmic=False
    geometric=False
    if linear:
        sweep_defocus_mm= np.linspace(dz_min_mm, dz_max_mm, N_dz)
    elif logarithmic:
        sweep_defocus_mm =np.logspace(np.log10(dz_min_mm), np.log10(dz_max_mm), (N_dz))
    elif geometric:
        sweep_defocus_mm =np.geomspace(dz_min_mm, dz_max_mm, N_dz) # could also use logspace
    log_psf_vmin = -5.0         # log10 PSF visuals

    '''
    Building our PSFs
    '''
    # Use function to construct a single spatial-frequency phase pattern over our pupil_grid
    #(OPD/lambda) or sinusoid in waves
    phi_sine_waves = make_sinusoidal_phase_waves(pupil_grid,
                                                 seal_parameters['pupil_size'],
                                                 cycles_per_aperture,
                                                 m_waves)
    
    # waves -> radians for HCIPy field multiplication, need this to apply phase to a complex field?
    #[radians of phase]
    phi_sine_rad = 2 * np.pi * phi_sine_waves

    # Focused wf using Fourier Injection, no defocus. Than get focused PSF 
    wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), seal_parameters['wavelength_meter']) 
    psf_focus = psf_from_wavefront(wf_focus)
    #plt.imshow(np.log10(psf_focus/psf_focus.max())), vmin =-5, cmap='inferno'

    # One defocus, using combined defocus distance and sinusoid term
    phi_def_demo = calculate_defocus_phase(seal_parameters, single_defocus_mm)  # [rad] on pupil_grid
    wf_demo = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def_demo)), seal_parameters['wavelength_meter'])
    
    psf_demo = psf_from_wavefront(wf_demo)
    #plt.imshow(np.log10(psf_demo/psf_demo.max())), vmin =-5, cmap='inferno'

    # Find the OTF magnitude by FFT of PSF. 
    otf_demo = otf_from_psf_numpy(psf_demo)
    off_demo, amp_demo, offs_lr_demo, amps_lr_demo = find_otf_sidepeaks_1D(otf_demo)
    #this is plotted with imshow(otf_demo)

    #Empty arrays to store distance in OTF px of sidepeak from DC, side-peak ampltitude. 
    offsets_px, amps = [], []

    '''

    Iterate over the defocus range in [mm], and convert mechanical defocus [mm] to defocus phase and add both the sin
    and defocus term to the WF. Build aberrated wf, p2f for intensity, FT the PSF for OTF mag, get all measured peak
    position and strength for a dz, put into np array. RN this is a skeleton for when i do cycles/aperture. trying to 
    think about the math for it. 

    '''

    #two 3-D arrays to store one psf/otf for every defocus in sweep
    psf_fits = np.empty((len(sweep_defocus_mm), focal_grid.shape[0], focal_grid.shape[1]), dtype=float)
    otf_fits=np.empty((len(sweep_defocus_mm), focal_grid.shape[0], focal_grid.shape[1]), dtype=float)

    #CHANGED SEP 3
    #computing angular pixel scale of focal grid, then normalized OTF frequency step per pixel
    #essentially letting us go from pix->angles
    #theta = lambda/D * (pixel index)
    #so each pixel corresponds to dtheta radians on sky
    dtheta = (2 * seal_parameters['Num_airycircles']* (seal_parameters['wavelength_meter']/seal_parameters['pupil_size'])) / focal_grid.shape[0]
    u_hat_per_px = (seal_parameters['wavelength_meter']/seal_parameters['pupil_size']) / (focal_grid.shape[1] *dtheta)
    #cycles/aperture per pixel, ie with airy=64,  OTF pixel= 1/128 cycles/aperture


    a_hats, vhat_pred, off_pred_px=[], [], []

    '''
    What this loop does:
    For a given dz:
        computes defocus ohase on pupil
        injects sinusoid and defocus into complex pupil and creates wavefront object
        propagates to the focal plane to get PSF
        FFT the psf to get OTF
        '''
    norm = plt.Normalize(vmin=dz_min_mm, vmax = dz_max_mm)
    cmap=cm.viridis
    for i, dz_mm in enumerate(sweep_defocus_mm):
        phi_def = calculate_defocus_phase(seal_parameters, float(dz_mm))
        wf = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)),
                       seal_parameters['wavelength_meter'])
        psf = psf_from_wavefront(wf)
        otf = otf_from_psf_numpy(psf)
        psf_fits[i] = psf

        #sets an adaptive kill window proprotional to coc for current dz. 
        kill = max(5, int(.25*coc_pixels(dz_mm, focal_grid, seal_parameters)))
        off, a, _, _ = find_otf_sidepeaks_1D(otf, kill_core_pix=kill, subpixel=True)
        a_hat = defocus_a_hat(dz_mm,
                              seal_parameters['focal_length_meters'],
                              seal_parameters['pupil_size'],
                              seal_parameters['wavelength_meter'])
        a_hats.append(a_hat)
        v0_pred = cycles_per_aperture / (8.0 * max(a_hat, 1e-12))
        vhat_pred.append(v0_pred)
        off_pred_px.append(v0_pred / u_hat_per_px)
        otf_fits[i] = otf
        offsets_px.append(off)
        amps.append(a)


        
    offsets_px = np.array(offsets_px)
    vhat_meas = offsets_px *u_hat_per_px
    amps = np.array(amps)

    '''    
    hdu_psf = fits.PrimaryHDU(psf_fits)
    hdul_psf =fits.HDUList([hdu_psf])
    hdul_psf.writeto("sinusoidal_data_psf_logspace.fits", overwrite=True)

    hdu_otf = fits.PrimaryHDU(otf_fits)
    hdul_otf = fits.HDUList([hdu_otf])
    hdul_otf.writeto("sinusoidal_data_otf_logspace.fits", overwrite=True)
    '''
    #1/dz overlay
    # (Dean & Bowers predict v̂0 ~ n̂0/(8 â);  have a 1/dz trend in pixels for comparison.)


    

    '''
    PLOTS
    
    '''
    #IDK ABOUT THIS PLOT
    dz_show = np.logspace(np.log10(dz_min_mm), np.log10(dz_max_mm), 10)
    dc_kill=False
    fig, ax = plt.subplots(figsize=(7,4.5))
    for dz in dz_show:
        i = int(np.argmin(np.abs(sweep_defocus_mm - dz))) # for a dz, find index i of nearest fram in stored OTF cube
        O = otf_fits[i]# OTF mag for frame i, 
        row = O[O.shape[0]//2, :].copy()#takes central row, 
        #row /= row.max() if row.max() > 0 else 1.0 #normalize to unit peak

        # normalized-frequency axis: v̂ = k * û_per_px
        k = np.arange(row.size) - (row.size//2)# freq index array centered at 0
        vhat_axis = k * u_hat_per_px#scaled to convert pixel offset->normalized OTF requency; line 285

        # adaptice DC mask tied to coc
        if dc_kill:
            kill = max(5, int(0.25 * coc_pixels(dz, focal_grid, seal_parameters)))
            c = row.size//2
            row[c - kill : c + kill + 1] = 0.0
        #overlays one line per dz on a shared v axis
        ax.plot(vhat_axis, row, lw=1.2, label=fr'Δz={dz:.2f} mm', color =cmap(norm(dz)))

    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # required for Matplotlib to know there's data
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Defocus Δz [mm]')

    ax.set_xlim(-0.35, 0.35)
    ax.set_xlabel('Normalized spatial frequency v')
    ax.set_ylabel('OTF magnitude (central row, norm.)')
    ax.set_title('OTF central row vs defocus')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    # Build 2D array: rows = defocus values, cols = frequency samples
    heatmap_mm = True
    heatmap_ahat=False
    
    if heatmap_mm:
        #Storage
        rows =[] # heatmap rows
        k_heatmaps =[] # predicted spatial frequency
        measured_peaks, measured_peaks_photo = [], [] #predicted peak intensity, aperture photometry
        px_preds = [] # predicted position in px 
        px_measured=[]
        vhat_preds = [] # predicted position in terms of normalized spatial frequency
        vhat_measured=[] #measured side peak locations [cycles/aperture]

        #Constants
        f_m=seal_parameters['focal_length_meters']
        D_m = seal_parameters['pupil_size']
        lam_m = seal_parameters['wavelength_meter']
        N_airy = seal_parameters['Num_airycircles']
        F = f_m/D_m

        delta_x = 2 * N_airy* (lam_m/D_m)*f_m
        dx=delta_x/focal_grid.shape[0] # pixel size in focal plane

        delta_k = (2*np.pi)/dx
        dk = (2*np.pi)/delta_x
        otf_heatmap_grid = make_uniform_grid([512,512], delta_k) # step 1 for pixel coords

        for i, dz in enumerate(sweep_defocus_mm):
            #grab otf central row, normalize per row
            O = otf_fits[i]
            row = O[O.shape[0]//2, :].copy()
            row /= row.max() if row.max() > 0 else 1.0
            rows.append(row)

            #measured peaks, one way
            off_px, amp, _, _= find_otf_sidepeaks_1D(O, kill_core_pix=5, subpixel=True)
            measured_peaks.append(amp)
            px_measured.append(off_px)
            vhat_measured.append(off_px*u_hat_per_px)#px to cycles/aperture

            #measured peaks, photometry

            #kept 
            #predicted peak position
            d_proj = dz*1e-3/F #[m]
            p_pred = d_proj/cycles_per_aperture # [m]
            k_pred= (2*np.pi)/p_pred #[rad/m]
            k_heatmap = (2*np.pi)/p_pred
            k_heatmaps.append(k_heatmap)
            px_heatmap = delta_x/p_pred

            vhat_pred = k_pred * (lam_m/D_m)
            vhat_preds.append(vhat_pred)

            px_pred = delta_x/p_pred # pixels on OTF array
            px_preds.append(px_pred)
            peak_aperture = make_circular_aperture(3*dk, center=[+px_pred*dk, 0])
            mask = peak_aperture(otf_heatmap_grid) > 0
            vals=[]
            amp_masked = O[mask.shaped].max() if np.any(mask.shaped) else np.nan
            measured_peaks_photo.append(amp_masked) 
        
        H = np.array(rows)   # shape (N_defocus, N_freq)
        measured_peaks =np.array(measured_peaks)
        measured_peaks_photo=np.array(measured_peaks_photo)

        # Frequency axis (same for all rows)
        k = np.arange(H.shape[1]) - H.shape[1]//2
        vhat_axis = k* u_hat_per_px #cyles/aperture

        # Side peak intensity vs defocus
        plt.figure(figsize=(6,4))
        plt.plot(sweep_defocus_mm, measured_peaks)
        plt.xlabel('Defocus [mm]')
        plt.ylabel('measured peak intensity')        
        plt.title(f'measured peak vs defocus(non-photometry), avg peak {np.mean(measured_peaks):.1f}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6,4))
        plt.plot(sweep_defocus_mm, measured_peaks_photo)
        plt.xlabel('Defocus [mm]')
        plt.ylabel('measured peak intensity')        
        plt.title(f'measured peak vs defocus(photometry), avg peak {np.mean(measured_peaks_photo):.1f}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



        #heatmap
        plt.figure(figsize=(7,5))
        #this might be -delta_k/2 -> delta_k/2
        extent = [(-delta_k/2), (delta_k/2),
                sweep_defocus_mm.min(), sweep_defocus_mm.max()]


        plt.imshow(H, aspect='auto', origin='lower', extent=extent,
                cmap='viridis', vmin=0, vmax=1)
        plt.plot(k_heatmaps, sweep_defocus_mm)
        plt.plot(vhat_preds, sweep_defocus_mm)
        plt.xlim(extent[0], extent[1])
        plt.colorbar(label="OTF magnitude (central row, norm.)")
        plt.xlabel(r'Physical Spatial Frequency, k, [rad/m]')
        plt.ylabel(r'Defocus $\Delta z$ [mm]')
        plt.title('Central OTF row vs defocus (2D map) with linear spacing, delta_k extent')
        plt.tight_layout()
        plt.show()
        
        #heatmap
        plt.figure(figsize=(7,5))
        #this might be -delta_k/2 -> delta_k/2
        extent = [(vhat_axis.min()), (vhat_axis.max()),
               sweep_defocus_mm.min(), sweep_defocus_mm.max()]


        plt.imshow(H, aspect='auto', origin='lower', extent=extent,
                cmap='viridis', vmin=0, vmax=1)
        plt.plot(k_heatmaps, sweep_defocus_mm)
        plt.plot(vhat_preds, sweep_defocus_mm)
        plt.xlim(extent[0], extent[1])
        plt.colorbar(label="OTF magnitude (central row, norm.)")
        plt.xlabel(r'Normalized spatial frequency $\hat{v}$')
        plt.ylabel(r'Defocus $\Delta z$ [mm]')
        plt.title('Central OTF row vs defocus (2D map) with linear spacing, vhat_axis extent')
        plt.tight_layout()
        plt.show()
    elif heatmap_ahat:
        a_hats = np.array([
            defocus_a_hat(dz,
                        seal_parameters['focal_length_meters'],
                        seal_parameters['pupil_size'],
                        seal_parameters['wavelength_meter'])
            for dz in sweep_defocus_mm
        ])

        # build 2D array of central rows
        rows = []
        for i in range(len(sweep_defocus_mm)):
            O = otf_fits[i]
            row = O[O.shape[0]//2, :].astype(float).copy()
            row /= row.max() if row.max() > 0 else 1.0   # normalize per row
            rows.append(row)
        H = np.array(rows)

        # frequency axis (same for all rows)
        k = np.arange(H.shape[1]) - H.shape[1]//2
        vhat_axis = k * u_hat_per_px

        #plot heatmap
        plt.figure(figsize=(7,5))
        extent = [vhat_axis.min(), vhat_axis.max(),
                a_hats.min(), a_hats.max()]

        plt.imshow(H, aspect='auto', origin='lower', extent=extent,
                cmap='viridis', vmin=0, vmax=1)

        plt.colorbar(label="OTF magnitude (central row, norm.)")
        plt.xlabel(r'Normalized spatial frequency $\hat{v}$')
        plt.ylabel(r'Defocus $\hat{a}$ [waves P–V]')
        plt.title('Central OTF row vs defocus (in waves)')
        plt.tight_layout()
        plt.show()


    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(np.log10(psf_focus + 1e-12), vmin=log_psf_vmin)
    axs[0].set_title(f"Focused PSF (sinusoid {cycles_per_aperture:.0f} cyc/ap, m={m_waves:.2f} waves)")
    axs[0].axis('off')
    axs[1].imshow(np.log10(psf_demo + 1e-12), vmin=log_psf_vmin)
    axs[1].set_title(f"Defocused PSF (dz = {single_defocus_mm:.1f} mm)")
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()
    ##put the expected peak(k_pred) on both sides
    ##if available do aperture photometry and take max
    F = seal_parameters['focal_length_meters']/seal_parameters['pupil_size']
    d_proj = single_defocus_mm*1e-3/F
    p_pred = d_proj/cycles_per_aperture
    k_pred= (2*np.pi)/p_pred
    delta_x = 2 * seal_parameters['Num_airycircles']* (seal_parameters['wavelength_meter']/seal_parameters['pupil_size'])*seal_parameters['focal_length_meters']

    # convert angular frequency k_pred -> normalized frequency vhat_pred
    vhat_pred = k_pred * (seal_parameters['wavelength_meter'] / seal_parameters['pupil_size'])
    # convert normalized freq to pixel offset
    px_pred = delta_x/p_pred
    dx=delta_x/focal_grid.shape[0]
    delta_k = (2*np.pi)/dx
    dk = (2*np.pi)/delta_x

    otf_grid = make_uniform_grid([512,512], delta_k)
    peak_aperture = make_circular_aperture(3*(dk), center = [px_pred*dk,0])
    peak_aperture_mask = peak_aperture(otf_grid)>0
    peak = otf_demo[peak_aperture_mask.shaped].max()
    plt.figure(figsize=(5, 4))
    plt.imshow(otf_demo, origin='upper')
    c0 = otf_demo.shape[1] // 2
    r0 = otf_demo.shape[0] // 2
    # measured peaks (already in offs_lr_demo)
    plt.plot([c0 + offs_lr_demo[0], c0 + offs_lr_demo[1]],
            [r0, r0], 'r.', ms=10, label='Measured peaks')
    # predicted peaks
    c_left  = int(round(c0 - px_pred))
    c_right = int(round(c0 + px_pred))
    plt.plot([c_left, c_right], [r0, r0], 'go', ms=8, label='Predicted peaks')
    plt.xlim(c0-40, c0+40)
    plt.ylim(r0-20, r0+20)  
    plt.title(f"OTF magnitude at Δz = {single_defocus_mm:.1f} mm\n"
            f"Measured ±{off_demo:.1f} px vs Predicted ±{px_pred:.1f} px, \n"
            f" with peak at {peak:.1f} is unitless bc normalized")
    plt.axis('off')
    plt.colorbar()
    plt.legend()
    plt.tight_layout()
    plt.show()


    ''' 
    plt.figure(figsize=(6, 4))
    plt.plot(sweep_defocus_mm, amps, '-o', lw=1.5, ms=4)
    plt.xlabel("Defocus dz [mm]")
    plt.ylabel("OTF sidepeak amplitude ")
    plt.title("Sidepeak strength vs defocus using logspace")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(sweep_defocus_mm, vhat_meas, '-o', lw=1.5, ms=4, label='Measured OTF peak pos (px)')
    plt.xlabel("Defocus dz [mm]")
    plt.ylabel("OTF v_hat measured")
    plt.title("OTF sidepeak position using v_hat vs defocus using logspace")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    '''

    dean_bowers_math=True
    if dean_bowers_math:
        '''
        brute force mathing it out here
        '''
        v0= float(cycles_per_aperture)   # injected spatial freq [cycles/aperture]
        # a_hat sweep 
        a_min, a_max = 0.001, 50           # x-extent
        a_hat = np.linspace(a_min, a_max, 800)
        # theory curve
        y = -np.sin( np.pi * (v0**2) / (8.0 * a_hat) )
        #  "reversal" extrema when argument hits this (2n-1)*pi/2
        # a_hat_n = v0^2 / [4*(2n-1)]
        n_vals = np.arange(1, 5)  # 
        a_rev = (v0**2) / (4.0 * (2*n_vals - 1))
        # first "maximum diversity defocus", n=1:
        a_rev_max = (v0**2) / 4.0   # equals a_rev[0]
        #plot
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(a_hat, y, lw=2.0)
        ax.set_xlim(2, 16)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel('a_hat  (defocus in waves P–V)')
        ax.set_ylabel('-1 * sin(pi*v0^2/8*a_hat)')
        ax.set_title(f'Position of maximum diversity defocus, v0={v0:.0f} cycles/aperture')
        #  first reversal max
        ax.axvline(a_rev_max, color='red', label=f'a_hat=(v0^2)/4={a_rev_max:.1f}')
        #  several extrema locations
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        # top axis in mechanical defocus [mm]
        secax = ax.secondary_xaxis('top')
        secax.set_xlabel('Defocus delta_z [mm]')
        plt.tight_layout()
        plt.show()

        #  chosen defocus values (manually selected)

        fixed_defocus_mm_custom = np.array([215.8, 222.1, 237.6, 228.3, 234.4, 200.0, 166.0, 175.0, 197.0])
        extra_defocus_mm = np.linspace(fixed_defocus_mm_custom.min(), 
                                    fixed_defocus_mm_custom.max(), 
                                    10)  # choose number of points you want


        fixed_defocus_mm_list = np.unique(np.concatenate([fixed_defocus_mm_custom, extra_defocus_mm]))

        v0_list = np.linspace(3, 80, 150)  #  spatial frequency [cycles/aperture]
        predicted_px = []
        k_fixed_dzs=[]
        delta_k = (2*np.pi)/dx
        dk = (2*np.pi)/delta_x
        otf_fixed_dz_grid = make_uniform_grid([512,512], delta_k)
        
        for fixed_defocus_mm in fixed_defocus_mm_list:
            amps_vs_v0=[]
            for v0 in v0_list:
                # build sinusoidal phase pattern for this v0
                phi_sine = make_sinusoidal_phase_waves(pupil_grid,
                                                    seal_parameters['pupil_size'],
                                                    v0,   # cycles/aperture
                                                    m_waves)
                phi_sine_rad = 2*np.pi*phi_sine

                # build defocus phase for fixed delta z
                dz_float = float(fixed_defocus_mm)
                phi_def = calculate_defocus_phase(seal_parameters, dz_float)

                # combine, propagate to PSF
                wf = Wavefront(telescope_pupil * np.exp(1j*(phi_sine_rad + phi_def)),
                            seal_parameters['wavelength_meter'])
                psf = psf_from_wavefront(wf)
                otf = otf_from_psf_numpy(psf)

                #predicted peak position
                d_proj = fixed_defocus_mm*1e-3/F #[m]
                p_pred = d_proj/cycles_per_aperture # [m]
                k_pred= (2*np.pi)/p_pred #[rad/m]

                k_fixed_dz = (2*np.pi)/p_pred
                k_fixed_dzs.append(k_fixed_dz)
                
                vhat_pred = k_pred * (lam_m/D_m)
                vhat_preds.append(vhat_pred)

                px_pred = delta_x/p_pred # pixels on OTF array
                px_preds.append(px_pred)

                # store predicted px offset 
                p_pred = d_proj/v0
                px_pred = delta_x/p_pred
                predicted_px.append(px_pred)

                peak_aperture = make_circular_aperture(3*dk, center=[+px_pred*dk, 0])
                mask = peak_aperture(otf_fixed_dz_grid) > 0
                vals=[]
                amp_masked_fixed = O[mask.shaped].max() if np.any(mask.shaped) else np.nan
                amps_vs_v0.append(amp_masked_fixed) 
            amps_vs_v0 = np.array(amps_vs_v0)
        
            hdu_psf = fits.PrimaryHDU(psf)
            hdul_psf =fits.HDUList([hdu_psf])
            hdul_psf.writeto(f"sinusoidal_data_psf_fixed_dz_{fixed_defocus_mm:.0f}.fits", overwrite=True)

            hdu_otf = fits.PrimaryHDU(otf)
            hdul_otf = fits.HDUList([hdu_otf])
            hdul_otf.writeto(f"otf_fixed_dz_{fixed_defocus_mm:.0f}mm.fits", overwrite=True)

    fixed_heatmap=True
    if fixed_heatmap:
            
        fixed_dz_heatmap= np.linspace(5,250,80) # defocus values in [mm]
        v0_heatmap = np.linspace(3,80,150) # cycles/aperture
        otf_fixed_heatmap_grid = make_uniform_grid([512,512], delta_k) # otf space

        #rows:defocus, columns:spatial, storing side peak amp per (dz, v0)
        H = np.zeros((len(fixed_dz_heatmap), len(v0_heatmap)))   
        
        
        for i, dz in enumerate(fixed_dz_heatmap):
            dz_float = float(dz)
            amps_heatmap_ph= []
            #defocus phase abberation for specific dz instance 
            phi_def = calculate_defocus_phase(seal_parameters, dz_float)

            for j, v0 in enumerate(v0_heatmap):
                # sinusoidal phase across phase in waves
                phi_sine = make_sinusoidal_phase_waves(pupil_grid,
                                                    D_m,
                                                    v0,
                                                    m_waves)
                #conversion to rads
                phi_sine_rad = 2*np.pi*phi_sine

                # wavefront and OTF
                wf = Wavefront(telescope_pupil * np.exp(1j*(phi_sine_rad + phi_def)),
                            lam_m)
                psf = psf_from_wavefront(wf)
                otf = otf_from_psf_numpy(psf)

                # predicted offset in px
                d_proj = dz_float*1e-3 / F
                p_pred = d_proj / v0
                px_pred = delta_x / p_pred

                # mask around predicted peak
                peak_aperture = make_circular_aperture(3*dk, center=[+px_pred*dk, 0])
                mask = peak_aperture(otf_fixed_heatmap_grid) > 0
                #peak amp inside mask, i put Nan instead if its nothign
                amp = otf[mask.shaped].max() if np.any(mask.shaped) else np.nan
                H[i, j] = amp

        
        fixed_defocus_mm_custom = [215.8, 222.1, 237.6, 228.3, 234.4, 200.0, 166.0, 175.0, 197.0]
        row_custom_dz = [int(np.argmin(np.abs(fixed_dz_heatmap-dzz))) for dzz in fixed_defocus_mm_custom]
        dz_snapto = [float(fixed_dz_heatmap[i]) for i in row_custom_dz]



        plt.figure(figsize=(11,10))
        for dz_fixed, i in zip(dz_snapto,row_custom_dz):
            row = H[i, ].astype(float).copy()
            finite=np.isfinite(row)
            plt.plot(v0_heatmap[finite], row[finite], label = fr'dz={dz_fixed:.1f} mm ')
        plt.legend()
        plt.xlabel("Spatial frequency n0 [cycles/aperture]")
        plt.ylabel("OTF sidepeak amplitude")
        plt.title("Contrast vs injected frequency for heatmap defocus values")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        ''' 
        Working on quadrature here
        dz_pair = [10,20]
        pair_row_custom_dz = [int(np.argmin(np.abs(fixed_dz_heatmap-dzz))) for dzz in dz_pair]
        dz_snapto_pair = [float(fixed_dz_heatmap[i]) for i in pair_row_custom_dz]
        rows = [H[i: ].astype(float)for i in pair_row_custom_dz]
        '''
        #normalization if we want
        rowwise_normalize = False   # toggle 
        if rowwise_normalize:
            # Each row is normalized to its own max = 1, looked this one up
            #I have 3 normalizations, one for PSF, one for OTF, and this one
            for i in range(H.shape[0]):
                row_max = np.nanmax(H[i])
                if row_max > 0:
                    H[i] /= row_max
            clim = (0, 1.0)
        else:
            # No per-row scaling
            clim = (0, np.nanmax(H))
        # heatmapo
        plt.figure(figsize=(8,6))
        extent = [v0_heatmap.min(), v0_heatmap.max(),
                fixed_dz_heatmap.min(), fixed_dz_heatmap.max()]
        plt.imshow(H, aspect='auto', origin='lower', extent=extent,
                cmap='viridis', vmin=clim[0], vmax=clim[1])
        # Label colorbar based on mode
        cbar = plt.colorbar()
        if rowwise_normalize:
            cbar.set_label("OTF side-peak amplitude, no intensity normalization (row-normalized)")
        else:
            cbar.set_label("OTF side-peak amplitude, no  Normalization (global)")
        plt.xlabel("Spatial frequency v0 [cycles/aperture]")
        plt.ylabel("Defocus dz [mm]")
        plt.title("OTF side-peak amplitude heatmap vs (dz, v0)")
        plt.tight_layout()
        plt.show()
        ##Save the Heatmap
        np.savez("OTF_heatmap_data_testing.npz", H=H, fixed_dz_heatmap=fixed_dz_heatmap, v0_heatmap=v0_heatmap)
        ##Load the Heatmap
load_heatmap=False
if load_heatmap:
            loaded_heatmap = np.load("OTF_heatmap_data_testing.npz")
            H = loaded_heatmap["H"]
            fixed_dz_heatmap= loaded_heatmap["fixed_dz_heatmap"]
            v0_heatmap = loaded_heatmap["v0_heatmap"]


want_sample_v0=True
if want_sample_v0:
            # Choose evenly spaced spatial frequency samples
            num_samples = 80
            sample_indices = np.linspace(0, len(v0_heatmap)-1, num_samples, dtype=int)
            sample_v0s = v0_heatmap[sample_indices]
            #chosen_indices=[1,2,3,4,5,6,7,8]
            #chosen_v0s=v0_heatmap[chosen_indices]
            dz_col_max=[]
            intensity_max=[]
            for j in sample_indices:
                #the chosen frequency goes to nearest column on simulated grid
                #j = np.argmin(np.abs(v0_heatmap-v0))

                column=H[:,j]
                if np.all(np.isnan(column)):
                    dz_col_max.append(np.nan)
                else:
                    i_max = np.nanargmax(column)             # row index of max (ignores NaN automatically)
                    dz_col_max.append(fixed_dz_heatmap[i_max])
                    intensity_max.append(column[i_max])
                    print(f'Max intensity of {fixed_dz_heatmap[i_max]} at {j} v0_indice')
                
            dz_col_max = np.array(dz_col_max)
            dz_col_max = np.array(dz_col_max)
            intensity_max = np.array(intensity_max)

            # Histogram of defocus values at which maxima occur
            plt.figure(figsize=(6,4))
            plt.hist(dz_col_max[~np.isnan(dz_col_max)], bins=15, alpha=0.7, edgecolor='black')
            plt.xlabel("Defocus Δz at column maximum [mm]")
            plt.ylabel("Count (number of frequencies)")
            plt.title("Histogram of Δz values at which maxima occur")
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(6,4))
            plt.plot(sample_indices, dz_col_max, 'o-', label='Max OTF defocus vs frequency')
            plt.xlabel("Spatial frequency v_0 [cycles/aperture]")
            plt.ylabel("Defocus dz_max [mm]")
            plt.title("Defocus value at maximum OTF response")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()


        # manual otf
manual_otf=False
if manual_otf:
            dz_target = 60.0   # mm
            v0_target = 20.0    # cycles/aperture

            # indice search
            i = np.argmin(np.abs(fixed_dz_heatmap - dz_target))
            j = np.argmin(np.abs(v0_heatmap - v0_target))

            print(f"Using dz={fixed_dz_heatmap[i]:.1f} mm, v0={v0_heatmap[j]:.1f} cycles/aperture")

            # wf
            phi_def = calculate_defocus_phase(seal_parameters, float(fixed_dz_heatmap[i]))
            phi_sine = make_sinusoidal_phase_waves(pupil_grid, D_m, v0_heatmap[j], m_waves)
            phi_sine_rad = 2*np.pi*phi_sine

            wf = Wavefront(telescope_pupil * np.exp(1j*(phi_sine_rad + phi_def)), lam_m)
            psf = psf_from_wavefront(wf)
            otf = otf_from_psf_numpy(psf)

            # save
            hdu = fits.PrimaryHDU(otf)
            hdul = fits.HDUList([hdu])
            hdul.writeto(f"DARK_OTF_dz{dz_target}_v0{v0_target}.fits", overwrite=True)

        ##TO-DO
'''
        From Researching, Ive found the type of noise i need to input would be primarily photon statistics using the Poisson distribution
        Read-Noise is secondary, while dark current and sky background are tertiary
        
        (sigma_rn)**2 = n_pix * (RN**2) : #pix * readout noise^2
        (sigma_poisson)**2 = N_photoelectrons=FT

        im assuming we are going to be read-noise limited primarily, therefore 

        Signal-to-Noise Ratio
        SNR_rn = FT / (RN)*sqrt(n_pix)
        specific time interval : T 
        average flux : F (photons/sec) is fixed
'''
noise_implemented = True
if noise_implemented:
        from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
        conf = InstrumentConfiguration(seal_param_config)
        N_trials=1
        sigma_e = 11
        seed = 12345
        rng = np.random.default_rng(seed) 
        pupil_mask = np.array(telescope_pupil.shaped >0, dtype=bool)
        Nd, Nv = len(fixed_dz_heatmap), len(v0_heatmap)
        rms_mean_nm = np.zeros((Nd, Nv)) * np.nan
        rms_std_nm  = np.zeros((Nd, Nv)) * np.nan

        def add_read_noise(image_electron, sigma_e, rng):
            read_noise = rng.normal(scale = sigma_e, size=image_electron.shape)
            return image_electron+read_noise
        
        #Monte Carlo
        std_dev_stack =[]
        mean_stack = []
        masked_stack = []
        wavelength_um = seal_param_config['wavelength']
        for i, dz in enumerate(fixed_dz_heatmap):
             phi_def = calculate_defocus_phase(seal_parameters, float(dz))
             for j, v0 in enumerate(v0_heatmap):
                        phi_sine_waves = make_sinusoidal_phase_waves(pupil_grid, D_m, v0, m_waves)
                        phi_sine_rad   = 2*np.pi*phi_sine_waves

                        # Make CLEAN PSFs 
                        wf_focus = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad)), lam_m)
                        psf_focus_clean = psf_from_wavefront(wf_focus) 
                        wf_defocused = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
                        psf_defoc_clean = psf_from_wavefront(wf_defocused)

                        rms_trials = []
                        real_pupil_stack = []
                        for t in range(N_trials):
                            psf0_noisy = add_read_noise(psf_focus_clean, sigma_e, rng)
                            psfd_noisy = add_read_noise(psf_defoc_clean, sigma_e, rng)
                            psf_list_stack = [psf0_noisy, psfd_noisy]
                            dx_list = [2.0071] * (len(psf_list_stack) - 1)

                            mp = FocusDiversePhaseRetrieval(psf_list_stack, wavelength_um, dx_list, [dz] )
                            for _ in range(200):
                                psf00=mp.step()
                            raw_pupil = np.angle(mft_rev(psf00,conf))
                            real_pupil = resize(raw_pupil, (256,256))*telescope_pupil.shaped
                            real_pupil_stack.append(real_pupil)
                            masked = real_pupil[telescope_pupil.shaped==1]
                            print((f"trial number N={N_trials} is fnished"))
                        real_pupil_stack_array=np.asarray(real_pupil_stack)
                        mean_stack.append(np.mean(real_pupil_stack_array, axis=0))
                        std_dev_stack.append(np.std(real_pupil_stack_array, axis = 0))
                        masked_stack.append(np.std(masked, axis=0))
                        rms_nm = np.sqrt(np.mean(real_pupil_stack_array**2))
                        rms_trials.append(rms_nm)
                        rms_mean_nm[i, j] = np.nanmean(rms_trials)
                        rms_std_nm[i, j]  = np.nanstd(rms_trials) # added in quadrature of the std deviation image masked by telexcope
                        #append into std dev list, than make heatmap
                        

        extent = [v0_heatmap.min(), v0_heatmap.max(),
                fixed_dz_heatmap.min(), fixed_dz_heatmap.max()]
        plt.figure(figsize=(8,6))
        im = plt.imshow(rms_mean_nm, aspect='auto', origin='lower', extent=extent,
                        cmap='magma_r')
        plt.colorbar(im, label="FDPR phase RMS (nm) — mean over trials")
        plt.xlabel(r"Spatial frequency $v_0$ [cycles/aperture]")
        plt.ylabel(r"Defocus $\Delta z$ [mm]")
        plt.title(f"FDPR Monte-Carlo mean RMS (N={N_trials}, σ_read={sigma_e:.0f} e⁻/px)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,6))
        im2 = plt.imshow(rms_std_nm, aspect='auto', origin='lower', extent=extent,vmin=np.nanmin(rms_std_nm),
                         vmax=np.nanmax(rms_std_nm), cmap='viridis')
        plt.colorbar(im2, label="FDPR phase RMS (nm) — std over trials")
        plt.xlabel(r"Spatial frequency $v_0$ [cycles/aperture]")
        plt.ylabel(r"Defocus $\Delta z$ [mm]")
        plt.title(f"FDPR Monte-Carlo RMS variability (N={N_trials})")
        plt.tight_layout()
        plt.show()

                            


