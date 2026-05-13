from DM_Sock import DM
from Track_Cam_cmds import TC_cmds
from FAM_cmds import FAM_cmds # for fiber alignment mirror
# import ktl # to acess keywords, generally don't do this
import matplotlib.pyplot as plt
import numpy as np
# Add datapyao to path
import sys
sys.path.append('/nfiudata/jaren_image_sharpening/data_pyao/data_ao_libs/')
from image_sharpening import NLOptPRFixedSampling,NLOptPRModel
from skimage.restoration import unwrap_phase
from scipy.ndimage import center_of_mass,shift
from scipy.signal import medfilt
from scipy.optimize import minimize

from prysm.coordinates import cart_to_polar
from prysm.polynomials import noll_to_nm,zernike_nm_sequence,lstsq,sum_of_2d_modes

from image_sharpening import ft_center_array,mft_rev,mft_fwd,crop_center,phase_unwrap_2d


# command to turn kpic light source off
#ktl.write('k2aopower','OUTLET_HB3','off')
# command to check if KPIC light source is on
#ktl.read('k2aopower','OUTLET_HB3')
# command to turn kpic light source on
#ktl.write('k2aopower','OUTLET_HB3','on')

# zero dm map
# DM.zeroAll()
# set surface
# DM.setSurf(numpy_array_surface)
# get surface
# DM.getSurf()
# poke Zernike
# DM.pokeZernike(amplitude,noll_index,bias=map you started with), applies & returns map applied
# flat map with Jacques
# jis/dm_maps/dm_flat.py

DM_ACTUATOR_STROKE = 3.5e-6 # verify this
DM_ACTUATOR_PITCH = 400e-6
CRED2_PIXEL_SCALE = 15 # microns
CRED2_EFL = 437.869 * 1e3 # um
CRED2_WVL = 1.550 # um
CRED2_D   = CRED2_EFL/32

def defocus_coeff_to_distance(P,fno):
    """defocus to image displacement from telescope-optics.net
    
    Parameters
    ----------
    P : float
        PV defocus in units of distance
        
    fno : float
        system focal ratio
    """
    return -8*P*fno**2

def gaussian_filter_psf(psf,waist=4):

    waist *= CRED2_WVL/CRED2_D # convert to radians
    waist *= CRED2_EFL # convert to microns
    print(f'waist microns = {waist}')
    shapex = psf.shape[0]
    det_size = shapex*CRED2_PIXEL_SCALE
    print(f'detector size = {det_size}')
    x = np.linspace(-det_size/2,det_size/2,shapex)
    x,y = np.meshgrid(x,x)
    r = np.sqrt(x**2 + y**2)
    gauss = np.exp(-(r/waist)**2)
    
    return gauss * psf
    
def fft_center(array):

    center = np.where(array == np.max(array))
    cen_x,cen_y = center[0],center[1]
    
    # build tilt phasor
    npix = array.shape[0]
    x,y = np.indices(array)
    x -= npix/2
    y -= npix/2
    
    u = x/(npix/2)
    v = y/(npix/2)
   
    phase = (-np.pi/2*x) - (np.pi/2*y)
    tilt_phasor = I*np.exp(1j*phase)
    

def high_contrast_psf(cam,FAM,framerates=[40,10,4],nframes=10):

    image_list = []
    dark_list = []
    light_list = []
    mask_list = []
    thresholds_max = [9000,9000,9000]
    thresholds_min = [1000,1000,-500]
    
    # dark frames
    FAM.set_pos('background')
    for fps in framerates:
    
        cam.set_fps(fps)
        
        # take darks
        dark_frames = cam.grab_n(nframes)[0].data
        median_dark = np.median(dark_frames,axis=0)
        dark_list.append(median_dark)
    
    # light frames
    FAM.set_pos('center')
    for fps,dark,mint,maxt in zip(framerates,dark_frames,thresholds_min,thresholds_max):
    
        cam.set_fps(fps)
        
        # take lights
        light_frames = cam.grab_n(nframes)[0].data
        median_light = np.median(light_frames,axis=0)
        median_light -= dark
        
        # threshold saturated / low signal
        median_light[median_light < mint] = 0
        median_light[median_light > maxt] = 0
        mask_list.append((median_light != 0).astype(int))
        
        # counts / s
        median_light *= fps
        
        light_list.append(median_light)
    psf_result = np.sum(light_list,axis=0)/np.sum(mask_list,axis=0)
    psf_result[np.isnan(psf_result)] = 0
    psf_result = medfilt(psf_result,kernel_size=3)
    
    return psf_result
      

# 'modify -s k2aopower OUTLET_HB3=on'

if __name__ == '__main__':

    threshold = 1
    niters = 50
    gaussian_filter = False
    modemax = 12
    blank = lambda phase: phase
    unwrapper = unwrap_phase
    
    # configure defocus positions
    defocus_coeffs = np.arange(0,0.3,0.1)
    defocus_dz = [defocus_coeff_to_distance(P,32) for P in defocus_coeffs] # 
    
    fps_states = []
    psf_list = []
    
    # cred2 params
    wavelength = 1.550e-6
    efl = 437.869e-3
    fno = 32
    image_dx = 15e-6
    image_dx_um = [image_dx*1e6 for i in range(len(defocus_dz))]
    D = efl/fno
    rad_per_LD = wavelength/D
    rad_per_pix = image_dx/efl
    pix_per_LD = rad_per_pix/rad_per_LD # 1 L/D is a wave of tilt
    
    # init classes
    cam = TC_cmds()
    DM = DM()
    FAM = FAM_cmds()
    
    # get current DM state
    dm_flat = DM.getSurf()
    

    # take multiple exposures and combine into single PSF
    
    # get psf
    psf_stacked = high_contrast_psf(cam,FAM)
    
    
    # make image square - TODO: make this automatic
    data = np.copy(psf_stacked)
    
    shapex,shapey = data.shape[0],data.shape[1]
    center = [int(shapex/2),int(shapey/2)]
    print('array center')
    print(center)
    cut = 210
    
    # 216.8 212.5 x=8 y=2
    offset_x = 16
    offset_y = 15
    cropped_data = data[center[0]-cut+offset_x:center[0]+cut+offset_x,
                        center[1]-cut+offset_y:center[1]+cut+offset_y]

    
    # threshold negative values to be near 0
    cropped_data[cropped_data < threshold] = 1e-10
    
    centered_data,computed_center = ft_center_array(cropped_data)
        
    
    # TODO: EXPERIMENTAL apply Gaussian filter
    if gaussian_filter:
        centered_data = gaussian_filter_psf(centered_data,waist=2)
    
    plt.figure(figsize=[10,5])
    imc = int(cropped_data.shape[0]/2)
    #cropped_data[imc,imc] = np.nan
    #centered_data[imc,imc] = np.nan
    cut = 100
    plt.subplot(121)
    plt.title('Before Centering')
    plt.imshow(np.log10(cropped_data),vmin=3)
    plt.xlim(imc-cut,imc+cut)
    plt.ylim(imc-cut,imc+cut)
    plt.colorbar()
    plt.subplot(122)
    plt.title('After Centering')
    plt.imshow(np.log10(centered_data),vmin=3)
    plt.xlim(imc-cut,imc+cut)
    plt.ylim(imc-cut,imc+cut)
    plt.colorbar()
    plt.show()
    
    data_maximum = np.max(cropped_data) # for strehl calculations
    
    
    # construct pupil mask
    x = np.linspace(-1,1,cropped_data.shape[0])
    x,y = np.meshgrid(x,x)
    pupil_amplitude = np.zeros_like(x)
    pupil_amplitude[x**2 + y**2 < 1] = 1
    
    # zernike basis
    r,t = cart_to_polar(x,y)
    nms = [noll_to_nm(i) for i in range(2,modemax)]
    basis = list(zernike_nm_sequence(nms,r,t))
    max_norm_basis = [mode/np.max(mode) for mode in basis]
        
    # re-apply dm map
    DM.setSurf(dm_flat)
    
    pr = NLOptPRFixedSampling(pupil_amplitude,wavelength*1e6,basis,centered_data)
    
    
    results = minimize(pr.fwd,np.random.random(len(basis))*1e-1,method='L-BFGS-B',
                       options={'maxiter':3000})
                       
    print(results.message)
    print(results.x)
    unwrapped_phase = sum_of_2d_modes(basis,results.x)
    pupil_estimation = np.exp(1j*2*np.pi/(wavelength*1e6)*unwrapped_phase)*pupil_amplitude
    psf_estimation = mft_fwd(pupil_estimation)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(unwrap_phase(np.angle(pupil_estimation)*pupil_amplitude),cmap='RdBu_r')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.log10(np.abs(psf_estimation)**2),cmap='inferno')
    plt.colorbar()
    plt.show()
    
    #sys.exit('Breakpoint')
    #unwrapped_phase = unwrapper(np.angle(pupil_estimation))
    #unwrapped_phase /= 2*np.pi # to waves
    #unwrapped_phase *= wavelength * 1e6 # to microns
    #unwrapped_phase /= 3.5*2 # to DM volt units
    #unwrapped_phase -= np.mean(unwrapped_phase[pupil_amplitude==1])
    # Now perform a Zernike decomposition
    # coeffs = lstsq(max_norm_basis,unwrapped_phase/pupil_amplitude)
    #i = 2
    #print('       GS          CG')
    #print('-----------------------------')
    #for cgs,ccg in zip(coeffs_gs,coeffs_cg):
    #    print(f'Z{i} = {cgs:.2e}   |   {ccg:.2e}')
    #    i += 1
        
    # convert to DM Map and send to surf
    dm_mask = DM.Mask
    dm_shape = dm_mask.shape[0]
    
    x = np.linspace(-1,1,dm_shape)
    x,y = np.meshgrid(x,x)
    r,t = cart_to_polar(x,y)
    
    dm_basis = list(zernike_nm_sequence(nms,r,t))
    dm_norm_basis = [mode/np.max(mode)*dm_mask for mode in dm_basis]
    dm_map = sum_of_2d_modes(dm_norm_basis,results.x) / 2 # * 2 * np.pi / (wavelength*1e6)
    
    dm_map = np.rot90(dm_map,k=1) # correct for DM mis-registration
    
    updated_map = dm_map+dm_flat
    updated_map[updated_map > 1] = 1
    updated_map[updated_map < 0] = 0
    updated_map *= dm_mask
    
    plt.figure()
    plt.imshow(unwrapped_phase,cmap='RdBu_r')
    plt.colorbar()
    plt.show()
    
    dm_limit = 1.0
    if np.max(np.abs(updated_map)) > dm_limit:
        sys.exit(f'Applied DM voltage exceeds hardcoded limit of {dm_limit}')
    
    DM.setSurf(updated_map)
    corrected_psf = high_contrast_psf(cam,FAM)
    print('-'*20,'percent difference in peak irradiance','-'*20)
    print((data_maximum - np.max(corrected_psf))/data_maximum * 100)
    
    cropped_after = corrected_psf[center[0]-cut+offset_x:center[0]+cut+offset_x,
                                  center[1]-cut+offset_y:center[1]+cut+offset_y]
                                  
    cut = 32
    plt.figure(figsize=[16,5])
    plt.subplot(131)
    plt.title('PSF')
    plt.imshow((cropped_data),cmap='inferno',vmin=-4)
    plt.xlim(420/2-cut,420/2+cut)
    plt.ylim(420/2-cut,420/2+cut)
    plt.colorbar()
    plt.subplot(132)
    plt.title('Estimated Phase wrapped')
    plt.imshow(np.angle(pupil_estimation)*pupil_amplitude/pupil_amplitude,cmap='RdBu_r')
    plt.colorbar(label='radians')
    plt.subplot(133)
    plt.title('Estimated Phase unwrapped')
    plt.imshow(unwrapped_phase*pupil_amplitude/pupil_amplitude,cmap='RdBu_r')
    plt.colorbar(label='dm volts')
    plt.show()
    
    
