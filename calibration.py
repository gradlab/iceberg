#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import numpy as np


from parameters import N_vector, Initial_I_nr, p_retreatment, p_followup
from calibration_functions import loop_over_abc_iter

outfolder ='calibration/'

  

if __name__=='__main__':

    
    # ----------------------------------------------------------------------------
    # set the fixed parameters 
    # ----------------------------------------------------------------------------
    
    # define where import frequency and location
    n_groups=len(N_vector)

    # ----- define the fixed parameters
    mu_entry = [0.00013]*8
    mu_exit = [0.00013]*8
    
    p_treatment_failure = np.array([0.8]*n_groups)
    

    # ----- make initial compartments
    
    # starting values for proportion symptomatic
    p_symptoms = np.array([0.65, 0.65, 0.65, 0.65,
                           0.65, 0.65,
                           0.55, 0.55])
        
    # set starting compartments
    S0 = N_vector
    Iinc_r0 = np.zeros(shape=n_groups)
    Iinc_nr0 = np.zeros(shape=n_groups)
    Is_r0 = np.zeros(shape=n_groups)
    Is_nr0 = np.zeros(shape=n_groups)
    Ia_r0 = np.zeros(shape=n_groups)
    Ia_nr0 = np.zeros(shape=n_groups)
    It_nr0 = np.zeros(shape=n_groups)
    Id_nr0 = np.zeros(shape=n_groups)
    It1_s_r_success0 = np.zeros(shape=n_groups)
    It1_s_r_failure0 = np.zeros(shape=n_groups)
    Id_s_r0 = np.zeros(shape=n_groups)
    It1_a_r_success0 = np.zeros(shape=n_groups)
    It1_a_r_failure0 = np.zeros(shape=n_groups)
    Id_a_r0 = np.zeros(shape=n_groups)
    
    # then insert the infectious individuals
    Is_nr0 = (p_symptoms.T * Initial_I_nr).astype(int).astype(float)
    Ia_nr0 = Initial_I_nr - Is_nr0
    
    # update S0 to adjust for the extra people in the model
    S0 = N_vector - Ia_r0 - Initial_I_nr
    
    # check that non-negative
    assert (S0>=0).all()
    
    y0 = S0, Iinc_r0, Iinc_nr0, Is_r0, Is_nr0, Ia_r0, Ia_nr0, It_nr0, Id_nr0, It1_s_r_success0, It1_s_r_failure0, Id_s_r0, It1_a_r_success0, It1_a_r_failure0, Id_a_r0
    
    params_fixed = N_vector, p_symptoms, mu_entry, mu_exit, p_retreatment, p_followup, p_treatment_failure
    
    
    
    # ------------------------------------------
    #  Run the simulations
    # ------------------------------------------
    # define number of samples
    abc_iter = 5
    
    
    # run with parallelization:
    
    p = Process(target=loop_over_abc_iter, args=(abc_iter, outfolder, params_fixed, y0))            
    p.start()
    p.join()
    
    # run without parallelization:
    #loop_over_abc_iter(abc_iter, outfolder, params_fixed, y0)
    
