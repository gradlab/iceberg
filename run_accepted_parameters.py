#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""





import numpy as np
import pandas as pd
from functools import reduce


from functions import run_simulation, prep_input_parameters
from parameters import N_vector, Initial_I_nr, p_retreatment, p_followup



# place to save the simulation results
out_folder = 'outputs/'
# place where calibration results are saved
in_folder = 'calibration/'




# -----------------------------------------------------------------------------
# set simulation time and iterations
# -----------------------------------------------------------------------------

group_names = ['MSM, high risk', 'MSM, low risk',
               'MSMW, high risk', 'MSMW, low risk',
               'MSW, high risk', 'MSW, low risk', 
               'WSM, high risk', 'WSM, low risk']

group_names_short = ['MSM_h', 'MSM_l',
               'MSMW_h', 'MSMW_l',
               'MSW_h', 'MSW_l', 
               'WSM_h', 'WSM_l']

n_groups = len(group_names)


n_iter = 500
T = 365 * 5
   
# intro frequency 
n_intro = 1



# -----------------------------------------------------------------------------
# load accepted parameters
# -----------------------------------------------------------------------------

accepted = pd.read_csv(in_folder+'accepted-parameters.csv')




# -----------------------------------------------------------------------------
# fixed parameters
# -----------------------------------------------------------------------------



# ----- define the fixed parameters
mu_entry = [0.00013]*8
mu_exit = [0.00013]*8


p_treatment_failure = np.array([0.8]*n_groups)




# -----------------------------------------------------------------------------
# run simulation
# -----------------------------------------------------------------------------


for abc_par_iter in np.arange(accepted.shape[0]):
    
    tuned_pars = accepted.iloc[abc_par_iter,:]
    
    # -----------------------------------------------------------------------------
    # set the tuned parameters
    # -----------------------------------------------------------------------------
    
    msm_beta_ll = tuned_pars.msm_beta_ll
    msm_beta_hh = tuned_pars.msm_beta_hh
    mtow_beta_ll = tuned_pars.mtow_beta_ll
    mtow_beta_hh = tuned_pars.mtow_beta_hh
    wtom_beta_ll = tuned_pars.wtom_beta_ll
    wtom_beta_hh = tuned_pars.wtom_beta_hh
    contacts_msm_h = tuned_pars.contacts_msm_h
    contacts_msm_l_scaler = tuned_pars.contacts_msm_l_scaler
    contacts_msw_h = tuned_pars.contacts_msw_h
    contacts_msw_l_scaler = tuned_pars.contacts_msw_l_scaler
    contacts_wsm_h = tuned_pars.contacts_wsm_h
    contacts_wsm_l_scaler = tuned_pars.contacts_wsm_l_scaler
    contacts_msmw_h_m = tuned_pars.contacts_msmw_h_m
    contacts_msmw_l_m_scaler = tuned_pars.contacts_msmw_l_m_scaler
    contacts_msmw_h_w = tuned_pars.contacts_msmw_h_w
    contacts_msmw_l_w_scaler = tuned_pars.contacts_msmw_l_w_scaler
    epsilon_param = tuned_pars.epsilon_param
    p_symptoms_men = tuned_pars.p_symptoms_men
    p_symptoms_women = tuned_pars.p_symptoms_women
    duration_symptoms_to_treatment_men = tuned_pars.duration_symptoms_to_treatment_men
    duration_symptoms_to_treatment_women = tuned_pars.duration_symptoms_to_treatment_women
    natural_clearance_men = tuned_pars.natural_clearance_men
    natural_clearance_women = tuned_pars.natural_clearance_women
    duration_treatment_to_recovery_men = tuned_pars.duration_treatment_to_recovery_men
    duration_treatment_to_recovery_women = tuned_pars.duration_treatment_to_recovery_women
    asymptomatic_screening_rate_msm_high = tuned_pars.asymptomatic_screening_rate_msm_high
    asymptomatic_screening_rate_msmw_high = tuned_pars.asymptomatic_screening_rate_msmw_high
    asymptomatic_screening_rate_msw_high = tuned_pars.asymptomatic_screening_rate_msw_high
    asymptomatic_screening_rate_wsm_high = tuned_pars.asymptomatic_screening_rate_wsm_high
    asymptomatic_screening_rate_msm_low = tuned_pars.asymptomatic_screening_rate_msm_low
    asymptomatic_screening_rate_msmw_low = tuned_pars.asymptomatic_screening_rate_msmw_low
    asymptomatic_screening_rate_msw_low = tuned_pars.asymptomatic_screening_rate_msw_low
    asymptomatic_screening_rate_wsm_low = tuned_pars.asymptomatic_screening_rate_wsm_low
    p_detection_men = tuned_pars.p_detection_men
    p_detection_women = tuned_pars.p_detection_women
    incubation_period_men = tuned_pars.incubation_period_men
    incubation_period_women = tuned_pars.incubation_period_women
    



    # -----------------------------------------------------------------------------
    # loop over index group
    # -----------------------------------------------------------------------------

    for group_iter, group_name in enumerate(group_names_short):
        
    
        # -----------------------------------------------------------------------------
        # set up the starting conditions
        # -----------------------------------------------------------------------------
        
        p_symptoms = np.array([0.65, 0.65, 0.65, 0.65,
                               0.65, 0.65,
                               0.55, 0.55])

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
        # resistant strain
        Ia_r0[group_iter] = n_intro
        
        # update S0 to adjust for the extra people in the model
        S0 = N_vector - Ia_r0 - Initial_I_nr
        
        # check that non-negative
        assert (S0>=0).all()
        
        y0 = S0, Iinc_r0, Iinc_nr0, Is_r0, Is_nr0, Ia_r0, Ia_nr0, It_nr0, Id_nr0, It1_s_r_success0, It1_s_r_failure0, Id_s_r0, It1_a_r_success0, It1_a_r_failure0, Id_a_r0
        
        
        # -----------------------------------------------------------------------------
        # prep input parameters
        # -----------------------------------------------------------------------------


        params_tunable0 =  (msm_beta_ll, msm_beta_hh ,mtow_beta_ll , mtow_beta_hh , wtom_beta_ll, wtom_beta_hh, contacts_msm_h , contacts_msm_l_scaler , contacts_msw_h, contacts_msw_l_scaler , contacts_wsm_h , contacts_wsm_l_scaler , contacts_msmw_h_m ,contacts_msmw_l_m_scaler , contacts_msmw_h_w ,contacts_msmw_l_w_scaler, epsilon_param , p_symptoms_men , p_symptoms_women , duration_symptoms_to_treatment_men , duration_symptoms_to_treatment_women, natural_clearance_men, natural_clearance_women, duration_treatment_to_recovery_men, duration_treatment_to_recovery_women, asymptomatic_screening_rate_msm_high, asymptomatic_screening_rate_msmw_high, asymptomatic_screening_rate_msw_high, asymptomatic_screening_rate_wsm_high, asymptomatic_screening_rate_msm_low, asymptomatic_screening_rate_msmw_low, asymptomatic_screening_rate_msw_low, asymptomatic_screening_rate_wsm_low, p_detection_men, p_detection_women, incubation_period_men, incubation_period_women)
        params_fixed0 = N_vector, p_symptoms, mu_entry, mu_exit, p_retreatment, p_followup, p_treatment_failure

        params_input = prep_input_parameters(params_fixed0, params_tunable0)
        

        # -----------------------------------------------------------------------------
        # run stochastic model
        # -----------------------------------------------------------------------------

        current_undetected_r, current_undetected_nr, new_detected_r_stack, new_detected_nr_stack, new_cases_r_stack, new_cases_nr_stack, cum_detected_r_stack, cum_detected_nr_stack, cum_cases_r_stack, cum_cases_nr_stack, current_cases_r, current_cases_nr = run_simulation(n_iter, T, y0, params_input)
        
        
        
        # -----------------------------------------------------------------------------
        # save it as a dataset
        # -----------------------------------------------------------------------------
        
        
        iterables=[np.arange(n_iter),np.arange(T),group_names]
        index = pd.MultiIndex.from_product(iterables, names=['iteration','time','group'])
        
        
        temp1 = pd.DataFrame(cum_detected_r_stack.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'cum_detected_r'}, axis=1)
        temp2 = pd.DataFrame(cum_detected_nr_stack.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'cum_detected_nr'}, axis=1)
        temp3 = pd.DataFrame(cum_cases_r_stack.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'cum_cases_r'}, axis=1)
        temp4 = pd.DataFrame(cum_cases_nr_stack.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'cum_cases_nr'}, axis=1)
        temp5 = pd.DataFrame(current_undetected_r.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'current_undetected_r'}, axis=1)
        temp6 = pd.DataFrame(current_undetected_nr.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'current_undetected_nr'}, axis=1)
        temp7 = pd.DataFrame(new_detected_r_stack.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'new_detected_r'}, axis=1)
        temp8 = pd.DataFrame(new_detected_nr_stack.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'new_detected_nr'}, axis=1)
        temp9 = pd.DataFrame(new_cases_r_stack.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'new_cases_r'}, axis=1)
        temp10 = pd.DataFrame(new_cases_nr_stack.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'new_cases_nr'}, axis=1)
        temp11 = pd.DataFrame(current_cases_r.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'current_cases_r'}, axis=1)
        temp12 = pd.DataFrame(current_cases_nr.reshape(n_iter*T*n_groups),index=index).reset_index().rename({0:'current_cases_nr'}, axis=1)
        
        
        
        # combine into a dataset
        data_frames=[temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,temp11,temp12]
        
        combodf = reduce(lambda  left,right: pd.merge(left,right,on=['iteration','time','group'],
                                                    how='outer'), data_frames)
        
        
            
    
    
        # save
        combodf.to_csv(out_folder+'/simulation_data_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'.csv', index=False)
    
        # add metadata about the simulation
        combodf_metadata = pd.DataFrame({'vars':['index_cases', 'index_group','intro_time', 'intro_frequency'],
                                            'values':[n_intro, group_name, 0, 'single']})
        combodf_metadata.to_csv(out_folder+'/simulation_metadata'+group_name+'_abc-params-'+str(abc_par_iter)+'.csv', index=False)
        
        print(group_name)
    print(abc_par_iter)

