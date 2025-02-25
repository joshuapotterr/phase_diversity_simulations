import matplotlib.pyplot as plt
import numpy as np

from hcipy import *
from skimage.transform import resize

from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev, InstrumentConfiguration
from processing import phase_unwrap_2d
#pep8 style

    seal_params = {'image_dx': 2.0071, # 
               'efl': focal_length*1e3, # SEAL effective focal length, mm
               'wavelength': 0.65, # SEAL center wavelength, microns
                'pupil_size': pupil_size*1e3, # Keck entrance pupil diameter
                    }
    # Configure instrument
    conf = InstrumentConfiguration(seal_params)

def calculate_phase_retrieval_accuracy():


def simulate_focused_image():


def simulate_defocused_image():


def run_phase_retrieval():
    mp= FocusDiversePhaseRetrieval(psf_list,650e-3, dx_list, distance_list)
    for i in range(200):
        psf00 = mp.step() 



def focus_diverse_phase_retrieval(system_truth, phase_diverse_inputs):
    #Defocus Dictionary
    defocus_dictionary = {}
    for defocus_distance in phase_diverse_inputs:
        defocus_dictionary[defocus_distance] = simulate_defocused_image()


    run_phase_retrieval(system_truth, defocus_dictionary)




def simulate_phase_diversity_grid(phase_diversity_grid):
    system_truth = simulate_focused_image() 
    for phase_diverse_inputs in phase_diversity_grid:
        focus_diverse_phase_retrieval(system_truth, phase_diverse_inputs)
        calculate_phase_retrieval_accuracy()



def plot_phase_diversity_heat_map()

def main():
    #Necessary to get Results for a Plot, using a predescribed unit
    simulate_phase_diversity_grid()
    #Last function is to plot, i.e 'plot phase diversity heat map'
    plot_phase_diversity_heat_map()


if __name__ == "__main__": 
    #Define Variables
    #function call(either 1 or specifically named call)
    main()



