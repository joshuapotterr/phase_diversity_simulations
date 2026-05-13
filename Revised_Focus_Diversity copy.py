import matplotlib.pyplot as plt
import numpy as np

from hcipy import *
from skimage.transform import resize

from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
from processing import phase_unwrap_2d
#pep8 style


def convert_psf_estimate_to_phase(psf_estimate, seal_parameters):
    pass 

def check_phase_estimate():
    pass

def make_cost_function_plots():
    pass

def calculate_phase_retrieval_accuracy(system_truth, psf_estimate, cost_functions, verbose=False):

    phase_estimate = convert_psf_estimate_to_phase(psf_estimate, seal_parameters)
    
    phase_estimate_metrics = check_phase_estimate(system_truth, phase_estimate)

    if verbose:
        make_cost_function_plots(cost_functions)

    return phase_estimate_metrics


def simulate_focused_image():
    pass

def simulate_defocused_image():
    pass 

def run_phase_retrieval(system_truth, defocus_dictionary, seal_parameters):
    
    distance_list = list(defocus_dictionary.keys())
    psf_list = [system_truth] + [defocus_dictionary[key] for key in distance_list]
    dx_list = [seal_parameters['image_dx'] for key in distance_list]

    mp= FocusDiversePhaseRetrieval(psf_list, seal_parameters['wavelength'], dx_list, distance_list)
    for i in range(200):
        psf_product = mp.step() 
    
    psf_estimate = np.angle(psf_product)
    
    return psf_estimate, mp.cost_functions 


def focus_diverse_phase_retrieval(system_truth, phase_diverse_inputs):
    #Defocus Dictionary
    defocus_dictionary = {}
    for defocus_distance in phase_diverse_inputs:
        defocus_dictionary[defocus_distance] = simulate_defocused_image()

    psf_estimate, cost_functions = run_phase_retrieval(system_truth, defocus_dictionary, seal_parameters)
    
    return psf_estimate, cost_functions



def simulate_phase_diversity_grid(phase_diversity_grid):
    system_truth = simulate_focused_image() 
    for phase_diverse_inputs in phase_diversity_grid:
        
        psf_estimate, cost_functions =
        focus_diverse_phase_retrieval(system_truth, phase_diverse_inputs, seal_parameters)
        calculate_phase_retrieval_accuracy(sytem_truth, psf_estimate, cost_functions, seal_parameters)



def plot_phase_diversity_heat_map()

def main(seal_parameters):
    #Necessary to get Results for a Plot, using a predescribed unit
    simulate_phase_diversity_grid()
    #Last function is to plot, i.e 'plot phase diversity heat map'
    plot_phase_diversity_heat_map()


if __name__ == "__main__": 
    #Define Variables
    seal_parameters = {'image_dx': 2.0071, # 
               'efl': focal_length*1e3, # SEAL effective focal length, mm
               'wavelength': 0.65, # SEAL center wavelength, microns
                'pupil_size': pupil_size*1e3, # Keck entrance pupil diameter
                    }

    #function call(either 1 or specifically named call)
    main(seal_parameters)



