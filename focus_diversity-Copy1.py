#!/usr/bin/env python
# coding: utf-8

# In[2]:


import image_sharpening


# In[3]:


from hcipy import *
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


import image_sharpening


# In[5]:


# Set up a pupil size / focal length that are roughly reasonable 
# We should double check these for our system

pupil_size = 10.12e-3 # KiloDM pupil = 10.12 mm
focal_length = 500e-3 # focal length into detector 500 mm

pupil_grid = make_pupil_grid(256, pupil_size)
aperture = circular_aperture(pupil_size)
telescope_pupil = aperture(pupil_grid)


imshow_field(telescope_pupil, cmap='gray')


# In[6]:


# Build a wavefront at our lab's laser light wavelength 
# Build a focal grid 
# Make a propagation function that moves from focal --> pupil (f2p) and pupil --> focal (p2f)

wavefront = Wavefront(telescope_pupil, wavelength=650e-9)
focal_grid = make_focal_grid(q=16, num_airy=16, pupil_diameter=pupil_size, focal_length=focal_length, reference_wavelength=650e-9)
prop_p2f = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)
prop_f2p = FraunhoferPropagator(focal_grid, pupil_grid, focal_length=focal_length)


# Ooooh pupil plane
pupil_image=wavefront.copy()
imshow_field(wavefront.intensity)


# In[7]:


# Ahhh focal plane 

#focal_image = prop_p2f.forward(wavefront)
#imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5)
#imshow_field(focal_image.phase)##bright point with aery rings
focal_image = prop_p2f.forward(wavefront)
imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5)
perfect_focal = focal_image.copy()


# In[8]:


# Using zernike modes built into hcipy we can pull out defocus 
# This is number 4 

# What is the scale height for the influence function here? 
# One of life's great mysteries 
influence_functions = make_zernike_basis(256, pupil_size, pupil_grid)
imshow_field(influence_functions[3])##This is defocus, maps out of lens being out of their determined focal length
plt.colorbar()

# And we can solve that mystery (kind of) by seeing what the dynamic range of this aberration is 
# Looks like it is a nice round number of ... 1.5ish radians? 
(np.max(influence_functions[4]) - np.min(influence_functions[4]))/np.pi


# In[9]:


# This tom-foolery is how we can put this aberration into our optical propagation system 
# We're essentially resetting the e field component of the wavefront we defined earlier 
# I think in a neat and ordered world you would wrap this operation in a fancy hcipy wrapper
# but here we thrive in chaos

scale_factor = 1 ##Changebale to whatever you want it to be, resets the peak to valley, specifically can reset the phase
pupil_image.electric_field = np.exp(complex(0, 1)*telescope_pupil*influence_functions[3]*scale_factor)

# We have put in a scale factor here -- it doesn't do anything yet but it could ...

imshow_field(pupil_image.phase*telescope_pupil)##Want this to look similar to cut above
plt.colorbar()


# In[10]:


# I've defined some convenience functions here 
# For the sake of good coding practices and demos I've made these PEP-8 and numpy docstring conventions
# but frankly this is probably overkill for the 3 lines of code...

# I've defined some convenience functions here 
# For the sake of good coding practices and demos I've made these PEP-8 and numpy docstring conventions
# but frankly this is probably overkill for the 3 lines of code...

def phase_to_m(phase, wv):
    """ Converts phase in radians to meters. 
    Parameters
    ----------
    phase : float or array of floats
        The phase input to be converted in radians. 
    wv: float
        The wavelength to use for conversion in meters. 
    Returns
    -------
    The phase information in units of meters.
    """
    return phase * wv / (2*np.pi)

def p_to_delta(P, f, D): 
    """ Converts P (the peak to valley error in the pupil plane) 
    induced by a delta (the longitudinal distance) offset.
    I.e., given some defocused image we can recontruct what that 
    distance would have been. Note that f and D always need to be
    in the same units -- mm is common, and P and delta will have 
    the same units. 
    Parameters
    ----------
    P : float
        The peak to valley error. 
    f : float
        The focal length leading up to this plane. 
    D : float
        The pupil size/telescope diameter of this plane. 
    Returns
    -------
    The delta defocus that would have been needed to create the P2V 
    we see. 
    """
    return 8 * P * (f/D)**2 


def delta_to_p(delta, f, D):
    """ p_to_delta() but in reverse. Note that f and D always need 
    to be in the same units -- mm is common, and P and delta will 
    have the same units. 
    Parameters
    ----------
    delta : float
        The longitudinal defocus distance. 
    f : float
        The focal length leading up to this plane. 
    D : float
        The pupil size of this plane. 
    Returns
    -------
    The P2V we would see based on the input delta. 
    """
    return -1*delta / (8 * (f/D)**2)


# ## Have some nicely written up math for the logic we just encapsualted
# 
# $\Delta = - 8PF^2$ 
# 
# where $\Delta$ is the physical longitudinal distance by which the optic is defocused 
# 
# $P$ is the peak to valley error in the pupil plane after said defocus 
# 
# $F = \frac{f}{D}$ is the F number, or the focal length divided by the telescope diameter (or in our case, pupil size.)

# In[11]:


# Now let's test out this relation
# Generate a zernike
##example_defocus = influence_functions[4].shaped ##shaped spits out zernike as a array


# Calculate the P2V in radians 
##p2v_radians = (np.max(example_defocus) - np.min(example_defocus))##returns in phase
##p2v_m = phase_to_m(p2v_radians, 650e-9)##650 is wavelength 

##print(f'Our example defocus would have been caused by ~ {np.round(p2v_m*1e6, 1)} um., ')
##print(f'p2v_radians = {p2v_radians}')\
example_defocus = influence_functions[4].shaped

# Calculate the P2V in radians 
p2v_radians = np.max(example_defocus) - np.min(example_defocus)
p2v_m = phase_to_m(p2v_radians, 650e-9)

print(f'Our example defocus has a P2V of ~ {np.round(p2v_m*1e9, 1)} nm or {p2v_m/(650e-9)} waves.')
print(f'Our example defocus as a defocus distance of {p_to_delta(p2v_m, focal_length, pupil_size)*1e3} mm')


# In[12]:


# Okay, what does this look like in the pupil plane? 
focal_image = prop_p2f(pupil_image)##propagating pupil to focal, can asjust pupil but want to see output of focal
imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5)


# ## Now it's your turn 
# 
# 1. Run through this notebook, make sure everything makes sense -- ask me lots of questions
# 2. Go through and generate defocus with different peak to valley errors. 
# 3. Propogate those through the system and see what they look like in the the focal plane. 
# 4. calculate the delta distance that would correspond to them. 
# 
# *kind reminder*: if you build an aberration with a p2v > 2pi, you will see phase wrapping ... 
# 

# In[13]:


# Let's see this in practice ... 
# Spin up a dramatic defocus 

drama_factor = 10
(np.min(drama_factor*example_defocus) - np.max(drama_factor*example_defocus))/np.pi
plt.imshow(drama_factor*example_defocus)
plt.colorbar()


# In[14]:


# But oops when we propagate it we lose the full P2V

pupil_image.electric_field = np.exp(complex(0, 1)*telescope_pupil*influence_functions[3]*drama_factor)
imshow_field(pupil_image.phase*telescope_pupil)
plt.colorbar()


# In[15]:


# We'll use skimage to handle that for now ... 

from skimage.restoration import unwrap_phase

# Unwrap the phase, sets lowest point of image to 0, looks different bc zero point is off
unwrapped = unwrap_phase(pupil_image.phase.shaped)

# And plot it out 
# Does anything look weird about this? 
plt.imshow(unwrapped*telescope_pupil.shaped)
plt.colorbar()


# In[16]:


# It should have the same P2V as the original zernike
# but if you want it to match you'll need to reset the zero point of the image
plt.imshow((unwrapped - np.mean(unwrapped))*telescope_pupil.shaped)
plt.colorbar()


# In[17]:


#Now let's do phase diversity.

#Spin up an optical aberation. We can throw in a rogue zernike mode to start, but could also be fun to build up a messy lens
#Given some known defocuses and their distances, propagate through the defocus + the aberation, and build a set of focal plane images : defocus distances.
#Throw them into Jaren's algorithm and see what happens...
#This is where we start pulling content from Jaren's notebook -- note that we are using hcipy not prysm so a lot of the optical propagation logic is totally different, but the phase diversity content is the same.


# In[18]:


# Let's put in a mild vertical coma
test_ab = 0.5*influence_functions[6]
imshow_field(test_ab)
plt.colorbar()
np.min(test_ab), np.max(test_ab)


# In[19]:


# Remind ourselves what this example defocus looks like

plt.imshow(example_defocus)
p2v_radians = (np.max(example_defocus) - np.min(example_defocus))
p2v_m = phase_to_m(p2v_radians, 650e-9)
print('P2V: ', p2v_m*1e9, ' nm error')


# In[20]:


# Now let's go through, define some defocuses, and estimate their defocus distance

D = pupil_size
f = focal_length

defocus_phase_1 = example_defocus * 2
p2v_radians = (np.max(defocus_phase_1) - np.min(defocus_phase_1))
p2v_m = phase_to_m(p2v_radians, 650e-9)
delta_1 = p_to_delta(p2v_m, f, D)
print(f'P2V error: {p2v_radians} rad, {p2v_radians/(2*np.pi)} waves, defocus distance: {delta_1*1e6} microns')


# In[21]:


defocus_phase_2 = example_defocus / 2
p2v_radians = (np.max(defocus_phase_2) - np.min(defocus_phase_2))
p2v_m = phase_to_m(p2v_radians, 650e-9)
delta_2 = p_to_delta(p2v_m, f, D)
print(f'P2V error: {p2v_radians/(2*np.pi)} waves, defocus distance: {delta_2*1e6} microns')


# In[22]:


# NOTE -- here adding the negative makes it a negative defocus 
# you have to add that negative manually when calculating the defocus distance

defocus_phase_3 = -1 * example_defocus 
p2v_radians = (np.max(defocus_phase_3) - np.min(defocus_phase_3))
p2v_m = phase_to_m(p2v_radians, 650e-9)
delta_3 = p_to_delta(p2v_m, f, D)
print(f'P2V error: {p2v_radians} rad, {-1*p2v_radians/(2*np.pi)} waves, defocus distance: {-1*delta_3*1e6} microns')


# In[23]:


# Now let's start again with a clean wavefront 
clean_wf = wavefront = Wavefront(telescope_pupil, wavelength=650e-9)
clean_focal = prop_p2f(clean_wf)
imshow_field(np.log10(clean_focal.intensity / clean_focal.intensity.max()), vmin=-5)


# In[24]:


# Now add the test aberration to the defocuses and propagate images to the focal plane
pupil_image.electric_field = np.exp(complex(0, 1)*(test_ab + defocus_phase_1.ravel()))*telescope_pupil
focal_image_1 = prop_p2f(pupil_image)
imshow_field(np.log10(focal_image_1.intensity / focal_image_1.intensity.max()), vmin=-5)


# In[25]:


pupil_image.electric_field = np.exp(complex(0, 1)*(test_ab + defocus_phase_2.ravel()))*telescope_pupil
focal_image_2 = prop_p2f(pupil_image)
imshow_field(np.log10(focal_image_2.intensity / focal_image_2.intensity.max()), vmin=-5)


# In[26]:


pupil_image.electric_field = np.exp(complex(0, 1)*(test_ab + defocus_phase_3.ravel()))*telescope_pupil
focal_image_3 = prop_p2f(pupil_image)
imshow_field(np.log10(focal_image_3.intensity / focal_image_3.intensity.max()), vmin=-5)


# In[27]:


# And we need one example that's no defocus, just the abberation we are trying to sense

pupil_image.electric_field = np.exp(complex(0, 1)*(test_ab))*telescope_pupil
focal_image_0 = prop_p2f(pupil_image)
imshow_field(np.log10(focal_image_0.intensity / focal_image_0.intensity.max()), vmin=-5)


# In[28]:


# dx_list is the width of the image in microns, so let's take a minute to pull that out of the image
# there may be an easier way to do it but I wanted to make sure you had *at least one* way to do it
# every focal image has a grid with a coordinates object, which is in meters (in a more realistic scenario 
# would we literally know the physcial extent but here it is being set by the focal length and the optical 
# propagation we set up so I'm not 100% sure what it is without checking)
# I'm pulling out that image, shaping it into something I can plot so I can see that it moves logically 
# from negative to positive across the image as I'd expect
# And then pulling the maximum/minimum value so I now how many meters across the image is 
plt.imshow(np.array(focal_image_0.grid.coords)[0].reshape(512,512))
plt.colorbar()
dx = focal_image_0.grid.coords[0][0] - focal_image_0.grid.coords[0][1]
print(dx)
# this plot is just me checking that I am pulling out coordinates in a way that makes sense 


# In[29]:


# PSF list starts with no-defocus image, and then has your known defocus inputs  
psf_list = [np.array(focal_image_0.intensity.shaped), 
            np.array(focal_image_1.intensity.shaped), 
            np.array(focal_image_2.intensity.shaped), 
            np.array(focal_image_3.intensity.shaped)]
distance_list = [13994.908389652417, 3498.7270974131043, -6997.454194826209]
dx_list = [2.0071, 2.0071, 2.0071] 


# In[30]:


# Okay, let's run the phase diversity algorithm!
# if this import doesn't work, note that we had to add the path manually
# the command to do this should be sourced in your bash profile
from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration

mp = FocusDiversePhaseRetrieval(psf_list,650e-3,dx_list,distance_list)
test_phase = 0
for i in range(200):
    psf00 = mp.step() # returns a model of the first PSF


# In[31]:


# Now let's inspect the output intensity -- just to make sure things look like we expect
# This should match the input no-defocus focal plane image

plt.imshow(np.angle(psf00))
plt.colorbar()


# In[32]:


# Now we define a special dictionary with SEAL params which will let us do a fourier transform

seal_params = {'image_dx': 2.0071, # 
               'efl': focal_length*1e3, # SEAL effective focal length, mm
               'wavelength': 0.65, # SEAL center wavelength, microns
                'pupil_size': pupil_size*1e3, # Keck entrance pupil diameter
                    }
conf = InstrumentConfiguration(seal_params)

# Take a fourier transform to convert this to the pupil plane 
# I think something about the simulation output does not play as well with hcipy 
# so this is the best way to get pupil plane phase for now
raw_pupil_phase = np.angle(mft_rev(psf00, conf))
plt.imshow(raw_pupil_phase)
plt.colorbar()
# hmmm, this is hard to parse so ...


# In[33]:


# We need to resize the image to our desired output and let's crop out the noise outside of the pupil 
from skimage.transform import resize
pupil_phase = resize(raw_pupil_phase, (256, 256))*telescope_pupil.shaped
plt.imshow(pupil_phase)
plt.colorbar()
print(f'P2V error: {np.max(pupil_phase) - np.min(pupil_phase)}')
# NOW it looks pretty good..


# In[34]:


# Compare to the original error we injected and are trying to recover 
pupil_image.electric_field = np.exp(complex(0, 1)*telescope_pupil*(test_ab))
plt.imshow(unwrap_phase(pupil_image.phase.shaped))
plt.colorbar()
print(f'P2V error: {np.max(pupil_image.phase.shaped) - np.min(pupil_image.phase.shaped)}')


# In[35]:


# And finally, lets see how well the reconstruction and the original input comare 
# Looks pretty good??? 
##ask jaren and jules to explain a little

med_subtracted = pupil_phase - np.median(pupil_phase[np.array(telescope_pupil.shaped, dtype=bool)])
plt.imshow(pupil_image.phase.shaped - med_subtracted)
plt.colorbar()
check_error_region = (pupil_image.phase.shaped - med_subtracted)[np.array(telescope_pupil.shaped, dtype=bool)]
print(f'Median error of {np.median(check_error_region)} radians.')


# In[36]:


# We can also check how well the cost function of the algorithm is converging, this is a good diagnostic
# The cost function for all three images is converging so this looks good
##cost function wants to be as low as possible ->signify working. 
plt.semilogy(mp.cost_functions[0], label='defocus 1', color='cyan')
plt.semilogy(mp.cost_functions[1], label='defocus 2', color='grey')
plt.semilogy(mp.cost_functions[2], label='defocus 3', color='teal')
plt.legend()
plt.show()


# In[37]:


# as a treat, if we need an alternative phase wrapping routine ...
# different phase unwrapping routine take from Jaren
def phase_unwrap_2d(phase_wrapped):

    """phase unwrapping routine based on the phaseunwrap2d.go script in IDL and the following proceedings:
    M.D. Pritt; J.S. Shipman, "Least-squares two-dimensional phase unwrapping using FFT's",
    IEEE Transactions on Geoscience and Remote Sensing ( Volume: 32, Issue: 3, May 1994),
    DOI: 10.1109/36.297989

    Uses a finite differences approach to determine the partial derivative of the wrapped phase in x and y,
    then solves the solution in the fourier domain

    TODO: Test this function against the prior in IDL, it doesn't appear to reconstruct phase well

    Parameters
    ----------
    phase_wrapped : numpy.ndarray
        array containing 2D signal to unwrap

    Returns
    -------
    numpy.ndarray
        unwrapped phase
    """

    imsize = phase_wrapped.shape
    M = imsize[0]
    N = imsize[1]

    Nmirror = 2 * (N )
    Mmirror = 2 * (M )

    phmirror = np.ones([Mmirror,Nmirror])

    # Quadrant 3
    phmirror[:M,:N] = phase_wrapped

    # First mirror reflection Quadrant 2
    phmirror[M:,:N] = np.flipud(phase_wrapped)

    # Second mirror reflection Quadrant 4
    phmirror[:M,N:] = np.fliplr(phase_wrapped)

    # Final reflection Quadrant 1
    phmirror[M:,N:] = np.flipud(np.fliplr(phase_wrapped))

    phroll = np.zeros_like(phmirror)
    phroll[:M,:N-1] = phmirror[:M,1:N]
    phroll[:M,N-1] = phmirror[:M,0]
    deltafd = phroll-phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)

    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdx = deltafd

    # compute forward difference
    phroll = np.zeros_like(phmirror)
    phroll[:M-1,:N] = phmirror[1:M,:N]
    phroll[M,:N] = phmirror[0,:N]
    deltafd = phroll - phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)
    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdy = deltafd

    # Solve system of equations formed by min LS -> phi
    D_n = np.fft.fft2(deltafdx)
    D_m = np.fft.fft2(deltafdy)
    inc_n = 2 * np.pi / Nmirror
    inc_m = 2 * np.pi / Mmirror

    nn = np.ones([Mmirror,1]) @ (np.arange(Nmirror))[np.newaxis]
    mm = np.ones([Nmirror,1]) @ (np.arange(Mmirror))[np.newaxis]
    mm = mm.transpose()
    
    i = 1j
    mult_n = np.ones([Mmirror,Nmirror]) - np.exp(-nn * i * inc_n)
    mult_m = np.ones([Mmirror,Nmirror]) - np.exp(-mm * i * inc_m)
    divisor = (np.cos(mm*inc_m) + np.cos(nn*inc_n) - np.ones([Mmirror,Nmirror])*2)*2
    divisor[0,0] = 1
    phi = (D_n*mult_n + D_m*mult_m) / divisor
    phi[0,0] = 0
    phi = np.fft.ifft2(phi)[:M,:N]
    phout = np.real(phi)
    return phout


# In[55]:


import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
from skimage.restoration import unwrap_phase

# Constants and parameters
pupil_size = 10.12e-3  # 10.12 mm
focal_length = 500e-3  # 500 mm
wavelength = 650e-9  # 650 nm
num_images = 10  # Start with 3 phase diversity inputs

# Generate pupil and focal grids
pupil_grid = make_pupil_grid(256, pupil_size)
aperture = make_circular_aperture(pupil_size)
telescope_pupil = aperture(pupil_grid)
wavefront = Wavefront(telescope_pupil, wavelength=wavelength)
focal_grid = make_focal_grid(q=16, num_airy=16, pupil_diameter=pupil_size, focal_length=focal_length, reference_wavelength=wavelength)
prop_p2f = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)


def apply_defocus_and_calculate_1(P_values, drama_factors, f, D):
    results = []
    for P in P_values:
        for drama_factor in drama_factors:
            pupil_image = wavefront.copy()
            pupil_image.electric_field = np.exp(1j * telescope_pupil * drama_factor)
            p2v_radians = np.max(pupil_image.phase) - np.min(pupil_image.phase)
            p2v_m = phase_to_m(p2v_radians, wavelength)
            p2v_mm = p2v_m * 1e3
            calculated_delta = p_to_delta(p2v_mm, f, D)
            results.append((drama_factor, p2v_radians, calculated_delta))
            
            # Unwrap and plot the phase
            unwrapped = unwrap_phase(pupil_image.phase.shaped)
            plt.imshow((unwrapped - np.mean(unwrapped)) * telescope_pupil.shaped)
            plt.colorbar()
            plt.title(f'Dramatic Defocus (Drama Factor: {drama_factor})')
            plt.show()
            
            print(f'For drama_factor = {drama_factor}, P2V_rad = {p2v_radians:.2f} radians, delta = {calculated_delta:.4f} mm.')
    return results

P_values = [0.05]
drama_factors = [2, 4, 6, 8, 10, -2, -4, -6, -8, -10]
results = apply_defocus_and_calculate_1(P_values, drama_factors, focal_length, pupil_size)

# Extracting values for plotting
drama_factors, p2v_radians, calculated_delta = zip(*results)

# Plot drama factor vs P2V (radians)
plt.figure()
plt.plot(drama_factors, p2v_radians, 'bo-', label='P2V (radians)')
plt.xlabel('Drama Factor')
plt.ylabel('P2V (radians)')
plt.legend()
plt.title('Drama Factor vs P2V (radians)')
plt.grid()
plt.show()

# Plot drama factor vs Physical Delta (mm)
plt.figure()
plt.plot(drama_factors, calculated_delta, 'ro-', label='Physical Delta (mm)')
plt.xlabel('Drama Factor')
plt.ylabel('Physical Delta (mm)')
plt.legend()
plt.title('Drama Factor vs Physical Delta (mm)')
plt.grid()
plt.show()

## Make a plot with drama factor with x, p2v in y ; drama factor in x and physical delta in y using plt
##Does phase reconstructuion perform better depending on the drama factor; look at the cost function vs iterative 
##look at residual plot; use drama factor x median

##num of images (with fixed drama factor) plot with median check error residual(how does residual change with num of images)
## use heatmap with colorbar as residual error ; one axis is drama factor the other is num images
## semi-optimized parameter would be the lowest median check error coinciding with the drama factor and num of image


# In[57]:


def generate_psf_list(drama_factor, num_images):
    psf_list = []
    dx_list = []
    distance_list = []

    # Initial PSF without defocus
    pupil_image = wavefront.copy()
    pupil_image.electric_field = np.exp(1j * telescope_pupil * drama_factor)
    psf = prop_p2f(pupil_image)
    psf_list.append(psf.intensity.shaped)

    for _ in range(num_images):
        defocus_amount = 2 * np.pi * np.random.rand()  # Random defocus
        pupil_image = wavefront.copy()
        pupil_image.electric_field = np.exp(1j * telescope_pupil * (drama_factor + defocus_amount))
        psf = prop_p2f(pupil_image)
        psf_list.append(psf.intensity.shaped)
        dx_list.append(pupil_grid.delta)
        distance_list.append(focal_length)

    return psf_list, dx_list, distance_list

def evaluate_performance(drama_factors, num_images_list):
    residuals = np.zeros((len(drama_factors), len(num_images_list)))

    for i, drama_factor in enumerate(drama_factors):
        for j, num_images in enumerate(num_images_list):
            psf_list, dx_list, distance_list = generate_psf_list(drama_factor, num_images)

            mp = FocusDiversePhaseRetrieval(psf_list, 650e-3, dx_list, distance_list)
            cost_functions = []

            for _ in range(200):
                psf00 = mp.step()
                cost_functions.append(mp.cost_functions)

            # Plot cost function convergence
            plt.figure()
            for idx in range(len(cost_functions[0])):
                plt.semilogy([cost[idx] for cost in cost_functions], label=f'Defocus {idx+1}')
            plt.legend()
            plt.title(f'Cost Function Convergence\nDrama Factor: {drama_factor}, Num Images: {num_images}')
            plt.xlabel('Iteration')
            plt.ylabel('Cost Function Value')
            plt.grid()
            plt.show()

            # Convert to pupil plane phase
            seal_params = {'image_dx': 2.0071, 'efl': focal_length * 1e3, 'wavelength': 0.65, 'pupil_size': pupil_size * 1e3}
            conf = InstrumentConfiguration(seal_params)
            raw_pupil_phase = np.angle(mft_rev(psf00, conf))
            pupil_phase = resize(raw_pupil_phase, (256, 256)) * telescope_pupil.shaped

            # Calculate residual error
            med_subtracted = pupil_phase - np.median(pupil_phase[np.array(telescope_pupil.shaped, dtype=bool)])
            check_error_region = (pupil_image.phase.shaped - med_subtracted)[np.array(telescope_pupil.shaped, dtype=bool)]
            median_error = np.median(check_error_region)
            residuals[i, j] = median_error

    return residuals

# Drama factors and number of images to test
drama_factors = [2, 4, 6, 8, 10, -2, -4, -6, -8, -10]
num_images_list = [1, 2, 3, 4, 5]

# Evaluate performance and plot residuals
residuals = evaluate_performance(drama_factors, num_images_list)

# Plot heatmap of residuals
plt.figure()
plt.imshow(residuals, aspect='auto', cmap='viridis', extent=[min(num_images_list), max(num_images_list), min(drama_factors), max(drama_factors)])
plt.colorbar(label='Residual Error')
plt.xlabel('Number of Images')
plt.ylabel('Drama Factor')
plt.title('Residual Error Heatmap')
plt.show()

# Identify semi-optimized parameters
min_residual_index = np.unravel_index(np.argmin(residuals, axis=None), residuals.shape)
optimal_drama_factor = drama_factors[min_residual_index[0]]
optimal_num_images = num_images_list[min_residual_index[1]]
print(f'Semi-optimized parameters: Drama Factor = {optimal_drama_factor}, Number of Images = {optimal_num_images}')


# In[ ]:


# Plot heatmap of residuals
plt.figure()
plt.imshow(residuals, aspect='auto', cmap='viridis', extent=[min(num_images_list), max(num_images_list), min(drama_factors), max(drama_factors)])
plt.colorbar(label='Residual Error')
plt.xlabel('Number of Images')
plt.ylabel('Drama Factor')
plt.title('Residual Error Heatmap')
plt.show()

# Identify semi-optimized parameters
min_residual_index = np.unravel_index(np.argmin(residuals, axis=None), residuals.shape)
optimal_drama_factor = drama_factors[min_residual_index[0]]
optimal_num_images = num_images_list[min_residual_index[1]]
print(f'Semi-optimized parameters: Drama Factor = {optimal_drama_factor}, Number of Images = {optimal_num_images}')


# In[39]:


# Generate Zernike modes
zernike_modes = make_zernike_basis(36, pupil_size, pupil_grid)# creating the zernike basis
defocus_mode = zernike_modes[4]  # Influence function 3 for defocus

# Function to apply defocus and calculate delta
def apply_defocus_and_calculate_2(P_values, f, D):
    results = []
    for P in P_values:
        drama_factor = P / (np.max(defocus_mode) - np.min(defocus_mode))  # Scale factor to match P2V
        pupil_image = wavefront.copy()# for changing
        pupil_image.electric_field = np.exp(1j * telescope_pupil * defocus_mode * drama_factor)# defocus applied to pupil
        p2v_radians = np.max(pupil_image.phase) - np.min(pupil_image.phase)# p2v in radians
        p2v_m = phase_to_m(p2v_radians, wavelength)# p2v in meters
        p2v_mm = p2v_m * 1e3 # p2v than to mm
        calculated_delta = p_to_delta(p2v_mm, f, D) # defocus distance
        results.append((P, drama_factor, p2v_radians, calculated_delta))
        
        # Unwrap and plot the phase
        unwrapped = unwrap_phase(pupil_image.phase.shaped)
        plt.imshow((unwrapped - np.mean(unwrapped)) * telescope_pupil.shaped)
        plt.colorbar()
        plt.title(f'Dramatic Defocus (P2V Error: {P:.1f} nm)')
        plt.show()
        
        print(f'For P2V Error = {P:.1f} nm, Drama Factor = {drama_factor:.2f}, P2V_rad = {p2v_radians:.2f} radians, Delta = {calculated_delta:.4f} mm.')
    return results

# Example values (range of P2V errors in nm)
P_values = np.arange(3, 100, 3)  # From 3 nm to 100 nm
results = apply_defocus_and_calculate_2(P_values, focal_length, pupil_size)

# Analyze results
for P, drama_factor, p2v_radians, calculated_delta in results:
    print(f'P2V Error: {P} nm, Drama Factor: {drama_factor:.2f}, P2V (radians): {p2v_radians:.2f}, Delta (mm): {calculated_delta:.4f}')


# In[40]:


##Seems the upper limit for drama_factor is around 83/84, until weird images and abberations start to occur. 
##As the error increased from there, it became a geomtric pattern, with a diamond(4 triangles at each point)
##The lower I go doesnt seem to have an effect, until 0. Thus, .0001<|drama_factor<|83.5


# In[41]:


##HW: Getting the delta from P for various samples, 3/4 good examples 
P = 0.0253 ##pick any number
delta = p_to_delta(P, focal_length, pupil_size)
print(f'The defocus distance delta is {delta:.4f} mm.')
##apply defocus, than find p2v, convert to mm, than the delta(practical)

#Delta = 
#P = 
#Drama_factor = 
#Influence_function = 


# In[42]:


# Apply defocus, find P2V, convert to mm, and calculate delta (practical)
#For each P, it calculates the defocus distance (delta) using the p_to_delta function.
def apply_defocus_and_calculate(P_values, drama_factors, f, D):
    for P in P_values:
        # delta = p_to_delta(P, f, D)
        # print(f'For P = {P}, the defocus distance delta is {delta:.4f} mm.')
        
        
#Inside the loop for P_values, another loop iterates over drama_factors.
#For each drama_factor, it applies the dramatic defocus by scaling the Zernike mode 4 (defocus) influence function.
#The pupil image's electric field is modified to include the defocus.
#P2V error in radians is calculated by finding the difference between the maximum and minimum phases of the pupil image.
#This P2V in radians is converted to meters using the phase_to_m function.

        for drama_factor in drama_factors:
            # Apply dramatic defocus
            pupil_image.electric_field = np.exp(1j * telescope_pupil * influence_functions[4] * drama_factor)
    
            # Calculate P2V in radians
            p2v_radians = np.max(pupil_image.phase) - np.min(pupil_image.phase)
            print(p2v_radians/(2*np.pi))
            # Convert P2V to meters
            p2v_m = phase_to_m(p2v_radians, 650e-9)
            p2v_mm = p2v_m * 1e3
            # Calculate delta
            calculated_delta = p_to_delta(p2v_mm, f, D)
            print(f'For drama_factor = {drama_factor}, P2V_rad = {p2v_radians:.2f} radians, delta = {calculated_delta:.4f} mm.')
            
             # Unwrap the phase using skimage
            unwrapped = unwrap_phase(pupil_image.phase.shaped)
            
            # Zero the unwrapped phase
            plt.figure()
            plt.imshow((unwrapped - np.mean(unwrapped)) * telescope_pupil.shaped)
            plt.colorbar()
            plt.title(f'Zeroed Unwrapped Phase of Pupil Plane with Dramatic Defocus (Drama Factor: {drama_factor})')
            plt.show()
            ##Ask why the coloring is messed up, too many tries -- should be same as p2v but NOT >:(
            
drama_factors = [2]
apply_defocus_and_calculate(P_values, drama_factors, focal_length, pupil_size)


# In[43]:


drama_factor = 2
pupil_image.electric_field = np.exp(1j * telescope_pupil * influence_functions[4] * drama_factor)
unwrapped = unwrap_phase(pupil_image.phase.shaped)
plt.figure()
plt.imshow((unwrapped - np.mean(unwrapped)) * telescope_pupil.shaped)
plt.colorbar()
##avg p2v ~4.5ish


# In[44]:


p2v_radians = np.max(pupil_image.phase) - np.min(pupil_image.phase)
print(p2v_radians)


# In[45]:


p2v_m = phase_to_m(p2v_radians, 650e-9)
p2v_nm = p2v_m * 1e9
print(p2v_nm)


# In[46]:


from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

import image_sharpening

# Set up a pupil size / focal length that are roughly reasonable 
# We should double check these for our system

pupil_size = 10.12e-3 # KiloDM pupil = 10.12 mm
focal_length = 500e-3 # focal length into detector 500 mm

pupil_grid = make_pupil_grid(256, pupil_size)
aperture = circular_aperture(pupil_size)
telescope_pupil = aperture(pupil_grid)


imshow_field(telescope_pupil, cmap='gray')

# Build a wavefront at our lab's laser light wavelength 
# Build a focal grid 
# Make a propagation function that moves from focal --> pupil (f2p) and pupil --> focal (p2f)

wavefront = Wavefront(telescope_pupil, wavelength=650e-9)
focal_grid = make_focal_grid(q=16, num_airy=16, pupil_diameter=pupil_size, focal_length=focal_length, reference_wavelength=650e-9)
prop_p2f = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)
prop_f2p = FraunhoferPropagator(focal_grid, pupil_grid, focal_length=focal_length)


# Ooooh pupil plane
pupil_image=wavefront.copy()
imshow_field(wavefront.intensity)

# Ahhh focal plane 

#focal_image = prop_p2f.forward(wavefront)
#imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5)
#imshow_field(focal_image.phase)##bright point with aery rings
focal_image = prop_p2f.forward(wavefront)
imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5)
perfect_focal = focal_image.copy()

# Using zernike modes built into hcipy we can pull out defocus 
# This is number 4 

# What is the scale height for the influence function here? 
# One of life's great mysteries 
influence_functions = make_zernike_basis(256, pupil_size, pupil_grid)
imshow_field(influence_functions[3])##This is defocus, maps out of lens being out of their determined focal length
plt.colorbar()

# And we can solve that mystery (kind of) by seeing what the dynamic range of this aberration is 
# Looks like it is a nice round number of ... 1.5ish radians? 
(np.max(influence_functions[4]) - np.min(influence_functions[4]))/np.pi

# This tom-foolery is how we can put this aberration into our optical propagation system 
# We're essentially resetting the e field component of the wavefront we defined earlier 
# I think in a neat and ordered world you would wrap this operation in a fancy hcipy wrapper
# but here we thrive in chaos

scale_factor = 1 ##Changebale to whatever you want it to be, resets the peak to valley, specifically can reset the phase
pupil_image.electric_field = np.exp(complex(0, 1)*telescope_pupil*influence_functions[3]*scale_factor)

# We have put in a scale factor here -- it doesn't do anything yet but it could ...

imshow_field(pupil_image.phase*telescope_pupil)##Want this to look similar to cut above
plt.colorbar()

# I've defined some convenience functions here 
# For the sake of good coding practices and demos I've made these PEP-8 and numpy docstring conventions
# but frankly this is probably overkill for the 3 lines of code...

def phase_to_m(phase, wv):
    """ Converts phase in radians to meters. 
    Parameters
    ----------
    phase : float or array of floats
        The phase input to be converted in radians. 
    wv: float
        The wavelength to use for conversion in meters. 
    Returns
    -------
    The phase information in units of meters.
    """
    return phase * wv / (2*np.pi)

def p_to_delta(P, f, D): 
    """ Converts P (the peak to valley error in the pupil plane) 
    induced by a delta (the longitudinal distance) offset.
    I.e., given some defocused image we can recontruct what that 
    distance would have been. Note that f and D always need to be
    in the same units -- mm is common, and P and delta will have 
    the same units. 
    Parameters
    ----------
    P : float
        The peak to valley error. 
    f : float
        The focal length leading up to this plane. 
    D : float
        The pupil size/telescope diameter of this plane. 
    Returns
    -------
    The delta defocus that would have been needed to create the P2V 
    we see. 
    """
    return 8 * P * (f/D)**2

def delta_to_p(delta, f, D):
    """ p_to_delta() but in reverse. Note that f and D always need 
    to be in the same units -- mm is common, and P and delta will 
    have the same units. 
    Parameters
    ----------
    delta : float
        The longitudinal defocus distance. 
    f : float
        The focal length leading up to this plane. 
    D : float
        The pupil size of this plane. 
    Returns
    -------
    The P2V we would see based on the input delta. 
    """
    return -1*delta / (8 * (f/D)**2)

## Have some nicely written up math for the logic we just encapsualted

$\Delta = - 8PF^2$ 

where $\Delta$ is the physical longitudinal distance by which the optic is defocused 

$P$ is the peak to valley error in the pupil plane after said defocus 

$F = \frac{f}{D}$ is the F number, or the focal length divided by the telescope diameter (or in our case, pupil size.)

# Now let's test out this relation
# Generate a zernike
##example_defocus = influence_functions[4].shaped ##shaped spits out zernike as a array


# Calculate the P2V in radians 
##p2v_radians = (np.max(example_defocus) - np.min(example_defocus))##returns in phase
##p2v_m = phase_to_m(p2v_radians, 650e-9)##650 is wavelength 

##print(f'Our example defocus would have been caused by ~ {np.round(p2v_m*1e6, 1)} um., ')
##print(f'p2v_radians = {p2v_radians}')\
example_defocus = influence_functions[3].shaped

# Calculate the P2V in radians 
p2v_radians = np.max(example_defocus) - np.min(example_defocus)
p2v_m = phase_to_m(p2v_radians, 650e-9)

print(f'Our example defocus has a P2V of ~ {np.round(p2v_m*1e9, 1)} nm or {p2v_m/(650e-9)} waves.')
print(f'Our example defocus as a defocus distance of {p_to_delta(p2v_m, focal_length, pupil_size)*1e3} mm')

# Okay, what does this look like in the pupil plane? 
focal_image = prop_p2f(pupil_image)##propagating pupil to focal, can asjust pupil but want to see output of focal
imshow_field(np.log10(focal_image.intensity / focal_image.intensity.max()), vmin=-5)

## Now it's your turn 

1. Run through this notebook, make sure everything makes sense -- ask me lots of questions
2. Go through and generate defocus with different peak to valley errors. 
3. Propogate those through the system and see what they look like in the the focal plane. 
4. calculate the delta distance that would correspond to them. 

*kind reminder*: if you build an aberration with a p2v > 2pi, you will see phase wrapping ... 


# Let's see this in practice ... 
# Spin up a dramatic defocus 

drama_factor = 10
(np.min(drama_factor*example_defocus) - np.max(drama_factor*example_defocus))/np.pi
plt.imshow(drama_factor*example_defocus)
plt.colorbar()

# But oops when we propagate it we lose the full P2V

pupil_image.electric_field = np.exp(complex(0, 1)*telescope_pupil*influence_functions[3]*drama_factor)
imshow_field(pupil_image.phase*telescope_pupil)
plt.colorbar()

# We'll use skimage to handle that for now ... 

from skimage.restoration import unwrap_phase

# Unwrap the phase, sets lowest point of image to 0, looks different bc zero point is off
unwrapped = unwrap_phase(pupil_image.phase.shaped)

# And plot it out 
# Does anything look weird about this? 
plt.imshow(unwrapped*telescope_pupil.shaped)
plt.colorbar()

# It should have the same P2V as the original zernike
# but if you want it to match you'll need to reset the zero point of the image
plt.imshow((unwrapped - np.mean(unwrapped))*telescope_pupil.shaped)
plt.colorbar()

#Now let's do phase diversity.

#Spin up an optical aberation. We can throw in a rogue zernike mode to start, but could also be fun to build up a messy lens
#Given some known defocuses and their distances, propagate through the defocus + the aberation, and build a set of focal plane images : defocus distances.
#Throw them into Jaren's algorithm and see what happens...
#This is where we start pulling content from Jaren's notebook -- note that we are using hcipy not prysm so a lot of the optical propagation logic is totally different, but the phase diversity content is the same.


# Let's put in a mild vertical coma
test_ab = 0.5*influence_functions[6]
imshow_field(test_ab)
plt.colorbar()
np.min(test_ab), np.max(test_ab)


# Remind ourselves what this example defocus looks like

plt.imshow(example_defocus)
p2v_radians = (np.max(example_defocus) - np.min(example_defocus))
p2v_m = phase_to_m(p2v_radians, 650e-9)
print('P2V: ', p2v_m*1e9, ' nm error')

# Now let's go through, define some defocuses, and estimate their defocus distance

D = pupil_size
f = focal_length

defocus_phase_1 = example_defocus * 2
p2v_radians = (np.max(defocus_phase_1) - np.min(defocus_phase_1))
p2v_m = phase_to_m(p2v_radians, 650e-9)
delta_1 = p_to_delta(p2v_m, f, D)
print(f'P2V error: {p2v_radians} rad, {p2v_radians/(2*np.pi)} waves, defocus distance: {delta_1*1e6} microns')

defocus_phase_2 = example_defocus / 2
p2v_radians = (np.max(defocus_phase_2) - np.min(defocus_phase_2))
p2v_m = phase_to_m(p2v_radians, 650e-9)
delta_2 = p_to_delta(p2v_m, f, D)
print(f'P2V error: {p2v_radians/(2*np.pi)} waves, defocus distance: {delta_2*1e6} microns')

# NOTE -- here adding the negative makes it a negative defocus 
# you have to add that negative manually when calculating the defocus distance

defocus_phase_3 = -1 * example_defocus 
p2v_radians = (np.max(defocus_phase_3) - np.min(defocus_phase_3))
p2v_m = phase_to_m(p2v_radians, 650e-9)
delta_3 = p_to_delta(p2v_m, f, D)
print(f'P2V error: {p2v_radians} rad, {-1*p2v_radians/(2*np.pi)} waves, defocus distance: {-1*delta_3*1e6} microns')

# Now let's start again with a clean wavefront 
clean_wf = wavefront = Wavefront(telescope_pupil, wavelength=650e-9)
clean_focal = prop_p2f(clean_wf)
imshow_field(np.log10(clean_focal.intensity / clean_focal.intensity.max()), vmin=-5)

# Now add the test aberration to the defocuses and propagate images to the focal plane
pupil_image.electric_field = np.exp(complex(0, 1)*(test_ab + defocus_phase_1.ravel()))*telescope_pupil
focal_image_1 = prop_p2f(pupil_image)
imshow_field(np.log10(focal_image_1.intensity / focal_image_1.intensity.max()), vmin=-5)

pupil_image.electric_field = np.exp(complex(0, 1)*(test_ab + defocus_phase_2.ravel()))*telescope_pupil
focal_image_2 = prop_p2f(pupil_image)
imshow_field(np.log10(focal_image_2.intensity / focal_image_2.intensity.max()), vmin=-5)

pupil_image.electric_field = np.exp(complex(0, 1)*(test_ab + defocus_phase_3.ravel()))*telescope_pupil
focal_image_3 = prop_p2f(pupil_image)
imshow_field(np.log10(focal_image_3.intensity / focal_image_3.intensity.max()), vmin=-5)

# And we need one example that's no defocus, just the abberation we are trying to sense

pupil_image.electric_field = np.exp(complex(0, 1)*(test_ab))*telescope_pupil
focal_image_0 = prop_p2f(pupil_image)
imshow_field(np.log10(focal_image_0.intensity / focal_image_0.intensity.max()), vmin=-5)

# dx_list is the width of the image in microns, so let's take a minute to pull that out of the image
# there may be an easier way to do it but I wanted to make sure you had *at least one* way to do it
# every focal image has a grid with a coordinates object, which is in meters (in a more realistic scenario 
# would we literally know the physcial extent but here it is being set by the focal length and the optical 
# propagation we set up so I'm not 100% sure what it is without checking)
# I'm pulling out that image, shaping it into something I can plot so I can see that it moves logically 
# from negative to positive across the image as I'd expect
# And then pulling the maximum/minimum value so I now how many meters across the image is 
plt.imshow(np.array(focal_image_0.grid.coords)[0].reshape(512,512))
plt.colorbar()
dx = focal_image_0.grid.coords[0][0] - focal_image_0.grid.coords[0][1]
print(dx)
# this plot is just me checking that I am pulling out coordinates in a way that makes sense 

# PSF list starts with no-defocus image, and then has your known defocus inputs  
psf_list = [np.array(focal_image_0.intensity.shaped), 
            np.array(focal_image_1.intensity.shaped), 
            np.array(focal_image_2.intensity.shaped), 
            np.array(focal_image_3.intensity.shaped)]
distance_list = [13994.908389652417, 3498.7270974131043, -6997.454194826209]
dx_list = [2.0071, 2.0071, 2.0071] 

# Okay, let's run the phase diversity algorithm!
# if this import doesn't work, note that we had to add the path manually
# the command to do this should be sourced in your bash profile
from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration

mp = FocusDiversePhaseRetrieval(psf_list,650e-3,dx_list,distance_list)
test_phase = 0
for i in range(200):
    psf00 = mp.step() # returns a model of the first PSF


# Now let's inspect the output intensity -- just to make sure things look like we expect
# This should match the input no-defocus focal plane image

plt.imshow(np.angle(psf00))
plt.colorbar()


# Now we define a special dictionary with SEAL params which will let us do a fourier transform

seal_params = {'image_dx': 2.0071, # 
               'efl': focal_length*1e3, # SEAL effective focal length, mm
               'wavelength': 0.65, # SEAL center wavelength, microns
                'pupil_size': pupil_size*1e3, # Keck entrance pupil diameter
                    }
conf = InstrumentConfiguration(seal_params)

# Take a fourier transform to convert this to the pupil plane 
# I think something about the simulation output does not play as well with hcipy 
# so this is the best way to get pupil plane phase for now
raw_pupil_phase = np.angle(mft_rev(psf00, conf))
plt.imshow(raw_pupil_phase)
plt.colorbar()
# hmmm, this is hard to parse so ...

# We need to resize the image to our desired output and let's crop out the noise outside of the pupil 
from skimage.transform import resize
pupil_phase = resize(raw_pupil_phase, (256, 256))*telescope_pupil.shaped
plt.imshow(pupil_phase)
plt.colorbar()
print(f'P2V error: {np.max(pupil_phase) - np.min(pupil_phase)}')
# NOW it looks pretty good..

# Compare to the original error we injected and are trying to recover 
pupil_image.electric_field = np.exp(complex(0, 1)*telescope_pupil*(test_ab))
plt.imshow(unwrap_phase(pupil_image.phase.shaped))
plt.colorbar()
print(f'P2V error: {np.max(pupil_image.phase.shaped) - np.min(pupil_image.phase.shaped)}')

# And finally, lets see how well the reconstruction and the original input comare 
# Looks pretty good??? 
med_subtracted = pupil_phase - np.median(pupil_phase[np.array(telescope_pupil.shaped, dtype=bool)])
plt.imshow(pupil_image.phase.shaped - med_subtracted)
plt.colorbar()
check_error_region = (pupil_image.phase.shaped - med_subtracted)[np.array(telescope_pupil.shaped, dtype=bool)]
print(f'Median error of {np.median(check_error_region)} radians.')

# We can also check how well the cost function of the algorithm is converging, this is a good diagnostic
# The cost function for all three images is converging so this looks good
plt.semilogy(mp.cost_functions[0], label='defocus 1', color='cyan')
plt.semilogy(mp.cost_functions[1], label='defocus 2', color='grey')
plt.semilogy(mp.cost_functions[2], label='defocus 3', color='teal')
plt.legend()
plt.show()

# as a treat, if we need an alternative phase wrapping routine ...
# different phase unwrapping routine take from Jaren
def phase_unwrap_2d(phase_wrapped):

    """phase unwrapping routine based on the phaseunwrap2d.go script in IDL and the following proceedings:
    M.D. Pritt; J.S. Shipman, "Least-squares two-dimensional phase unwrapping using FFT's",
    IEEE Transactions on Geoscience and Remote Sensing ( Volume: 32, Issue: 3, May 1994),
    DOI: 10.1109/36.297989

    Uses a finite differences approach to determine the partial derivative of the wrapped phase in x and y,
    then solves the solution in the fourier domain

    TODO: Test this function against the prior in IDL, it doesn't appear to reconstruct phase well

    Parameters
    ----------
    phase_wrapped : numpy.ndarray
        array containing 2D signal to unwrap

    Returns
    -------
    numpy.ndarray
        unwrapped phase
    """

    imsize = phase_wrapped.shape
    M = imsize[0]
    N = imsize[1]

    Nmirror = 2 * (N )
    Mmirror = 2 * (M )

    phmirror = np.ones([Mmirror,Nmirror])

    # Quadrant 3
    phmirror[:M,:N] = phase_wrapped

    # First mirror reflection Quadrant 2
    phmirror[M:,:N] = np.flipud(phase_wrapped)

    # Second mirror reflection Quadrant 4
    phmirror[:M,N:] = np.fliplr(phase_wrapped)

    # Final reflection Quadrant 1
    phmirror[M:,N:] = np.flipud(np.fliplr(phase_wrapped))

    phroll = np.zeros_like(phmirror)
    phroll[:M,:N-1] = phmirror[:M,1:N]
    phroll[:M,N-1] = phmirror[:M,0]
    deltafd = phroll-phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)

    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdx = deltafd

    # compute forward difference
    phroll = np.zeros_like(phmirror)
    phroll[:M-1,:N] = phmirror[1:M,:N]
    phroll[M,:N] = phmirror[0,:N]
    deltafd = phroll - phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)
    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdy = deltafd

    # Solve system of equations formed by min LS -> phi
    D_n = np.fft.fft2(deltafdx)
    D_m = np.fft.fft2(deltafdy)
    inc_n = 2 * np.pi / Nmirror
    inc_m = 2 * np.pi / Mmirror

    nn = np.ones([Mmirror,1]) @ (np.arange(Nmirror))[np.newaxis]
    mm = np.ones([Nmirror,1]) @ (np.arange(Mmirror))[np.newaxis]
    mm = mm.transpose()
    
    i = 1j
    mult_n = np.ones([Mmirror,Nmirror]) - np.exp(-nn * i * inc_n)
    mult_m = np.ones([Mmirror,Nmirror]) - np.exp(-mm * i * inc_m)
    divisor = (np.cos(mm*inc_m) + np.cos(nn*inc_n) - np.ones([Mmirror,Nmirror])*2)*2
    divisor[0,0] = 1
    phi = (D_n*mult_n + D_m*mult_m) / divisor
    phi[0,0] = 0
    phi = np.fft.ifft2(phi)[:M,:N]
    phout = np.real(phi)
    return phout





##Seems the upper limit for drama_factor is around 83/84, until weird images and abberations start to occur. 
##As the error increased from there, it became a geomtric pattern, with a diamond(4 triangles at each point)
##The lower I go doesnt seem to have an effect, until 0. Thus, .0001<|drama_factor<|83.5

##HW: Getting the delta from P for various samples, 3/4 good examples 
P = 0.0253 ##pick any number
delta = p_to_delta(P, focal_length, pupil_size)
print(f'The defocus distance delta is {delta:.4f} mm.')
##apply defocus, than find p2v, convert to mm, than the delta(practical)

#Delta = 
#P = 
#Drama_factor = 
#Influence_function = 



# Apply defocus, find P2V, convert to mm, and calculate delta (practical)
#For each P, it calculates the defocus distance (delta) using the p_to_delta function.
def apply_defocus_and_calculate(P_values, drama_factors, f, D):
    for P in P_values:
        # delta = p_to_delta(P, f, D)
        # print(f'For P = {P}, the defocus distance delta is {delta:.4f} mm.')
        
        
#Inside the loop for P_values, another loop iterates over drama_factors.
#For each drama_factor, it applies the dramatic defocus by scaling the Zernike mode 4 (defocus) influence function.
#The pupil image's electric field is modified to include the defocus.
#P2V error in radians is calculated by finding the difference between the maximum and minimum phases of the pupil image.
#This P2V in radians is converted to meters using the phase_to_m function.

        for drama_factor in drama_factors:
            # Apply dramatic defocus
            pupil_image.electric_field = np.exp(1j * telescope_pupil * influence_functions[4] * drama_factor)
    
            # Calculate P2V in radians
            p2v_radians = np.max(pupil_image.phase) - np.min(pupil_image.phase)
            print(p2v_radians/(2*np.pi))
            # Convert P2V to meters
            p2v_m = phase_to_m(p2v_radians, 650e-9)
            p2v_mm = p2v_m * 1e3
            # Calculate delta
            calculated_delta = p_to_delta(p2v_mm, f, D)
            print(f'For drama_factor = {drama_factor}, P2V_rad = {p2v_radians:.2f} radians, delta = {calculated_delta:.4f} mm.')
            
             # Unwrap the phase using skimage
            unwrapped = unwrap_phase(pupil_image.phase.shaped)
            
            # Zero the unwrapped phase
            plt.figure()
            plt.imshow((unwrapped - np.mean(unwrapped)) * telescope_pupil.shaped)
            plt.colorbar()
            plt.title(f'Zeroed Unwrapped Phase of Pupil Plane with Dramatic Defocus (Drama Factor: {drama_factor})')
            plt.show()
            ##Ask why the coloring is messed up, too many tries -- should be same as p2v but NOT >:(
            
drama_factors = [2]
apply_defocus_and_calculate(P_values, drama_factors, focal_length, pupil_size)





drama_factor = 2
pupil_image.electric_field = np.exp(1j * telescope_pupil * influence_functions[4] * drama_factor)
unwrapped = unwrap_phase(pupil_image.phase.shaped)
plt.figure()
plt.imshow((unwrapped - np.mean(unwrapped)) * telescope_pupil.shaped)
plt.colorbar()
##avg p2v ~4.5ish

p2v_radians = np.max(pupil_image.phase) - np.min(pupil_image.phase)
print(p2v_radians)

p2v_m = phase_to_m(p2v_radians, 650e-9)
p2v_nm = p2v_m * 1e9
print(p2v_nm)






# In[ ]:





# In[ ]:





# In[ ]:




