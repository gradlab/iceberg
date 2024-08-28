#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""



import numpy as np
import pandas as pd


from functions import run_simulation 



# ----------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------


def estBetaParams(mu, var):
    k = (mu*(1-mu)/var) - 1
    alpha = mu*k
    beta = (1-mu)*k
    return alpha, beta


def estGammaParams(mu, var):
    alpha = (mu**2)/var
    beta = var/mu
    return alpha, beta



# ----------------------------------------------------------------------------
# prepare inputs, run simulation, prepare outputs
# ----------------------------------------------------------------------------


def calibration_wrapper( params_tunable0, params_fixed, y0, burnin_T=5*365, n_iter = 3, T=6*365):

    assert burnin_T < T
    
    N_vector, p_symptoms, mu_entry, mu_exit, p_retreatment, p_followup, p_treatment_failure = params_fixed

    
    # ----- input preparation for simulation
    # expand the tunable parameters
    msm_beta_ll, msm_beta_hh ,mtow_beta_ll , mtow_beta_hh , wtom_beta_ll, wtom_beta_hh, contacts_msm_h , contacts_msm_l_scaler , contacts_msw_h, contacts_msw_l_scaler , contacts_wsm_h , contacts_wsm_l_scaler , contacts_msmw_h_m ,contacts_msmw_l_m_scaler , contacts_msmw_h_w ,contacts_msmw_l_w_scaler, epsilon_param , p_symptoms_men , p_symptoms_women, duration_symptoms_to_treatment_men , duration_symptoms_to_treatment_women, natural_clearance_men, natural_clearance_women, duration_treatment_to_recovery_men, duration_treatment_to_recovery_women, asymptomatic_screening_rate_msm_high, asymptomatic_screening_rate_msmw_high, asymptomatic_screening_rate_msw_high, asymptomatic_screening_rate_wsm_high, asymptomatic_screening_rate_msm_low, asymptomatic_screening_rate_msmw_low, asymptomatic_screening_rate_msw_low, asymptomatic_screening_rate_wsm_low, p_detection_men, p_detection_women, incubation_period_men, incubation_period_women = params_tunable0

    
    # make transmission matrix
    msm_beta_hl = np.sqrt(msm_beta_hh*msm_beta_ll)
    msm_beta_lh = np.sqrt(msm_beta_hh*msm_beta_ll)
    mtow_beta_lh = np.sqrt(mtow_beta_hh*mtow_beta_ll)
    mtow_beta_hl = np.sqrt(mtow_beta_hh*mtow_beta_ll)
    wtom_beta_lh = np.sqrt(wtom_beta_hh*wtom_beta_ll)
    wtom_beta_hl = np.sqrt(wtom_beta_hh*wtom_beta_ll)
    transmission_matrix = np.array([[msm_beta_hh, msm_beta_hl, msm_beta_hh, msm_beta_hl, 0, 0, 0, 0],
                                   [msm_beta_lh, msm_beta_ll,  msm_beta_lh, msm_beta_ll, 0, 0, 0, 0],
                                   [msm_beta_lh, msm_beta_ll,  msm_beta_lh, msm_beta_ll, 0, 0, mtow_beta_hh, mtow_beta_hl],
                                   [msm_beta_hh, msm_beta_hl, msm_beta_hh, msm_beta_hl, 0, 0, mtow_beta_lh, mtow_beta_ll],
                                   [0, 0, 0, 0, 0, 0, mtow_beta_hh, mtow_beta_hl],
                                   [0, 0, 0, 0, 0, 0, mtow_beta_lh, mtow_beta_ll],
                                   [0, 0, wtom_beta_hh, wtom_beta_hl, wtom_beta_hh, wtom_beta_hl, 0, 0 ],
                                   [0, 0, wtom_beta_lh, wtom_beta_ll, wtom_beta_lh, wtom_beta_ll, 0, 0 ],
                                   ])
    # contact rates
    contact_rates = np.array([contacts_msm_h, 
                              contacts_msm_h * contacts_msm_l_scaler,
                              contacts_msmw_h_m,
                              contacts_msmw_h_m * contacts_msmw_l_m_scaler,
                              contacts_msmw_h_w,
                              contacts_msmw_h_w* contacts_msmw_l_w_scaler,
                              contacts_msw_h,
                              contacts_msw_h* contacts_msw_l_scaler,
                              contacts_wsm_h,
                              contacts_wsm_h * contacts_wsm_l_scaler
                              ])
    # probability of symptoms
    p_symptoms = np.array([p_symptoms_men, p_symptoms_men, p_symptoms_men, p_symptoms_men,
                           p_symptoms_men, p_symptoms_men,
                           p_symptoms_women, p_symptoms_women])
    
    # time from symptoms to treatment
    duration_symptoms_to_treatment = np.array([duration_symptoms_to_treatment_men, duration_symptoms_to_treatment_men,
                                               duration_symptoms_to_treatment_men, duration_symptoms_to_treatment_men,
                                               duration_symptoms_to_treatment_men, duration_symptoms_to_treatment_men,
                                               duration_symptoms_to_treatment_women, duration_symptoms_to_treatment_women])
    # natural clearance
    p_recovery_innate = np.array([natural_clearance_men, natural_clearance_men, 
                                  natural_clearance_men, natural_clearance_men,
                                  natural_clearance_men, natural_clearance_men, 
                                  natural_clearance_women, natural_clearance_women])

    # likelihood of treatment
    # asymptomatic
    p_a_treat = np.array([asymptomatic_screening_rate_msm_high,
                          asymptomatic_screening_rate_msm_low,
                          asymptomatic_screening_rate_msmw_high,
                          asymptomatic_screening_rate_msmw_low,
                          asymptomatic_screening_rate_msw_high,
                          asymptomatic_screening_rate_msw_low,
                          asymptomatic_screening_rate_wsm_high,
                          asymptomatic_screening_rate_wsm_low
                          ])
    
    # duration treatment to recovery
    duration_treatment_to_recovery = np.array([duration_treatment_to_recovery_men, duration_treatment_to_recovery_men,
                                               duration_treatment_to_recovery_men, duration_treatment_to_recovery_men,
                                               duration_treatment_to_recovery_men, duration_treatment_to_recovery_men,
                                               duration_treatment_to_recovery_women, duration_treatment_to_recovery_women])
    # symptomatic
    p_detection_s = np.array([p_detection_men, p_detection_men, p_detection_men, p_detection_men,
                                       p_detection_men, p_detection_men, p_detection_women, p_detection_women])
    # asymptomatic
    p_detection_a = p_detection_s.copy()
    
    # incubation period
    incubation_period = np.array([incubation_period_men, incubation_period_men, incubation_period_men,
                                    incubation_period_men, incubation_period_men, incubation_period_men,
                                    incubation_period_women, incubation_period_women])

    
    # combine parameters for simulation
    params_general = mu_entry, mu_exit
    params_transmission = transmission_matrix, contact_rates, epsilon_param 
    params_symptoms = p_symptoms, incubation_period
    params_treatment = duration_symptoms_to_treatment, p_recovery_innate, p_a_treat, p_retreatment, duration_treatment_to_recovery, p_treatment_failure
    params_surveillance = p_detection_s, p_detection_a, p_followup
    
    # combo paramters 
    params0_input = params_general, params_transmission, params_symptoms, params_treatment, params_surveillance
    
    
    
    # ----- run simulation
    current_undetected_r, current_undetected_nr, new_detected_r_stack, new_detected_nr_stack, new_cases_r_stack, new_cases_nr_stack, cum_detected_r_stack, cum_detected_nr_stack, cum_cases_r_stack, cum_cases_nr_stack, current_cases_r, current_cases_nr = run_simulation(n_iter, T, y0, params0_input)         
    
    
    # ----- calculate outputs
    # calculate incidence (for each iteration) (r+nr)
    # (by gender)
    sim_incidence = new_detected_r_stack + new_detected_nr_stack
    
    # calculate prevalence (for each iteration)
    sim_prevalence = current_cases_r + current_cases_nr
    
    # fit to average incidence and prevalence after burn-in period
    # drop burn-in times
    sim_incidence_afterburnin = sim_incidence[:, burnin_T:, :]
    sim_prevalence_afterburnin = sim_prevalence[:, burnin_T:, :]
    
    # calculate average over simulation iterations and over remaining period
    sim_incidence_afterburnin_avg = np.mean(sim_incidence_afterburnin, axis=(0,1))
    sim_prevalence_afterburnin_avg = np.mean(sim_prevalence_afterburnin, axis=(0,1))
    
    # agg by gender
    sim_incidence_male = np.sum(sim_incidence_afterburnin_avg[:6], axis=-1)
    sim_incidence_female = np.sum(sim_incidence_afterburnin_avg[6:], axis=-1)
    
    # per capita
    sim_incidence_male = sim_incidence_male/np.sum(N_vector[:6])
    sim_incidence_female = sim_incidence_female/np.sum(N_vector[6:])
    sim_prevalence_afterburnin_avg = sim_prevalence_afterburnin_avg/N_vector
    
    out = [sim_incidence_male, sim_incidence_female] + list(sim_prevalence_afterburnin_avg)
    
    return out




# ----------------------------------------------------------------------------
# sample parameters from prior and run simulation
# ----------------------------------------------------------------------------


def loop_over_abc_iter(abc_iter, outfolder, params_fixed, y0):
    
    for aa in np.arange(abc_iter):
        
        sample_params_out = np.zeros(shape=[10**4, 48])
        
        # define colnames for output df
        df_colnames = ['sim_iteration',
                             'sim_incidence_men',
                             'sim_incidence_women',
                             'sim_prevalence_msm_h',
                             'sim_prevalence_msm_l',
                             'sim_prevalence_msmw_h',
                             'sim_prevalence_msmw_l',
                             'sim_prevalence_msw_h', 
                             'sim_prevalence_msw_l',
                             'sim_prevalence_wsm_h',
                             'sim_prevalence_wsm_l',                                                
                             'msm_beta_ll', 
                             'msm_beta_hh' ,
                             'mtow_beta_ll' , 
                             'mtow_beta_hh' , 
                             'wtom_beta_ll', 
                             'wtom_beta_hh', 
                             'contacts_msm_h' , 
                             'contacts_msm_l_scaler' , 
                             'contacts_msw_h', 
                             'contacts_msw_l_scaler' , 
                             'contacts_wsm_h' , 
                             'contacts_wsm_l_scaler' , 
                             'contacts_msmw_h_m' ,
                             'contacts_msmw_l_m_scaler' , 
                             'contacts_msmw_h_w',
                             'contacts_msmw_l_w_scaler', 
                             'epsilon_param' , 
                             'p_symptoms_men' , 
                             'p_symptoms_women', 
                             'duration_symptoms_to_treatment_men' , 
                             'duration_symptoms_to_treatment_women', 
                             'natural_clearance_men', 
                             'natural_clearance_women', 
                             'duration_treatment_to_recovery_men', 
                             'duration_treatment_to_recovery_women', 
                             'asymptomatic_screening_rate_msm_high', 
                             'asymptomatic_screening_rate_msmw_high', 
                             'asymptomatic_screening_rate_msw_high', 
                             'asymptomatic_screening_rate_wsm_high', 
                             'asymptomatic_screening_rate_msm_low', 
                             'asymptomatic_screening_rate_msmw_low', 
                             'asymptomatic_screening_rate_msw_low', 
                             'asymptomatic_screening_rate_wsm_low', 
                             'p_detection_men', 
                             'p_detection_women',
                             'incubation_period_men', 
                             'incubation_period_women'
                            ]
        
        # repeated sampling
        for ii in np.arange(10**4):
            
            # -- transmission --
            # transmission prob given infectious contact
            a, b = estBetaParams(0.59, 0.01)
            msm_beta_ll = np.random.beta(a=a, b=b)
            
            a, b = estBetaParams(0.5, 0.01)
            msm_beta_hh = np.random.beta(a=a, b=b)
            
            a, b = estBetaParams(0.8, 0.01)
            mtow_beta_ll = np.random.beta(a=a, b=b)
            wtom_beta_ll = np.random.beta(a=a, b=b)
            
            a, b = estBetaParams(0.72, 0.01)
            mtow_beta_hh = np.random.beta(a=a, b=b)
            wtom_beta_hh = np.random.beta(a=a, b=b)
            
            
            # contacts per year
            a, b = estGammaParams(30.5, 100)
            contacts_msm_h = np.random.gamma(a, b)
            contacts_msm_l_scaler = np.random.uniform(low=0.001, high=0.9)
        
            a, b = estGammaParams(20, 60)
            contacts_msw_h = np.random.gamma(a, b)
            contacts_msw_l_scaler = np.random.uniform(low=0.001, high=0.9)
            contacts_wsm_h = np.random.gamma(a, b)
            contacts_wsm_l_scaler = np.random.uniform(low=0.001, high=0.9)
            
            a, b = estGammaParams(30.5*0.75, 100)
            contacts_msmw_h_m = np.random.gamma(a, b)
            contacts_msmw_l_m_scaler = np.random.uniform(low=0.001, high=0.9)
            
            a, b = estGammaParams(20*0.75, 60)
            contacts_msmw_h_w = np.random.gamma(a, b)
            contacts_msmw_l_w_scaler = np.random.uniform(low=0.001, high=0.9)
            
            # assortativity
            epsilon_param = np.random.beta(a=2., b=2.)
            
            
            # -- symptoms --
            a, b = estBetaParams(0.65, 0.03)
            p_symptoms_men = np.random.beta(a, b)
            a, b = estBetaParams(0.54, 0.02)
            p_symptoms_women = np.random.beta(a, b)
            
            
            # -- treatment --
            
            # time to treatment (more than one day)
            duration_symptoms_to_treatment_men = 1
            duration_symptoms_to_treatment_women = 1
            a, b = estGammaParams(5.3, 4.2)
            while duration_symptoms_to_treatment_men <=1:
                duration_symptoms_to_treatment_men = np.random.gamma(a, b)
            a, b = estGammaParams(8.4, 5.2)
            while duration_symptoms_to_treatment_women <=1:
                duration_symptoms_to_treatment_women = np.random.gamma(a, b)
            
            
            # natural clearance rate 
            duration_natural_clearance_men=0.99
            duration_natural_clearance_women=0.99
            while duration_natural_clearance_men<1:
                a, b = estGammaParams(44, 4)
                duration_natural_clearance_men = np.random.gamma(a,b)
            natural_clearance_men = 1/duration_natural_clearance_men
            while duration_natural_clearance_women<1:
                a, b = estGammaParams(88, 4)
                duration_natural_clearance_women = np.random.gamma(a,b)
            natural_clearance_women = 1/duration_natural_clearance_women
            
            # recovery
            duration_treatment_to_recovery_men = 0.99
            duration_treatment_to_recovery_women = 0.99
            a, b = estGammaParams(3, 3)
            while duration_treatment_to_recovery_men < 1:
                duration_treatment_to_recovery_men = np.random.gamma(a,b)
            while duration_treatment_to_recovery_women < 1:
                duration_treatment_to_recovery_women = np.random.gamma(a,b)
            
            
            # screening rates

            # high risk groups - sampled from uniform
            screen_freq_msm_high = np.random.uniform(low=1, high=4)
            asymptomatic_screening_rate_msm_high = screen_freq_msm_high/365
            
            screen_freq_msmw_high = np.random.uniform(low=1, high=4)
            asymptomatic_screening_rate_msmw_high = screen_freq_msmw_high/365
            screen_freq_msw_high = np.random.uniform(low=0, high=1)
            asymptomatic_screening_rate_msw_high = screen_freq_msw_high/365
            
            screen_freq_wsm_high = np.random.uniform(low=0, high=1)
            asymptomatic_screening_rate_wsm_high = screen_freq_wsm_high/365
            
            # low risk groups - scalars
            screen_freq_msm_low_scalar = np.random.uniform(low=0.001, high=0.999)
            asymptomatic_screening_rate_msm_low = screen_freq_msm_low_scalar*screen_freq_msm_high/365
            
            screen_freq_msmw_low_scalar = np.random.uniform(low=0.001, high=0.999)
            asymptomatic_screening_rate_msmw_low = screen_freq_msmw_low_scalar*screen_freq_msmw_high/365
            
            screen_freq_msw_low_scalar = np.random.uniform(low=0.001, high=0.999)
            asymptomatic_screening_rate_msw_low = screen_freq_msw_low_scalar*screen_freq_msw_high/365
            
            screen_freq_wsm_low_scalar = np.random.uniform(low=0.001, high=0.999)
            asymptomatic_screening_rate_wsm_low = screen_freq_wsm_low_scalar*screen_freq_wsm_high/365
            
            
            # sequencing / susceptibility testing
            p_detection_men = np.random.uniform( low=0.001, high=0.3)
            p_detection_women = np.random.uniform( low=0.001, high=0.3)
            
            
            # incubation period
            incubation_period_men = 1
            incubation_period_women = 1
            a, b = estGammaParams(6.7, 2)
            while incubation_period_men <=1:
                incubation_period_men = np.random.gamma(a, b)
            a, b = estGammaParams(12, 4)
            while incubation_period_women <=1:
                incubation_period_women = np.random.gamma(a, b)
            
            
            

            # combine
            params_tunable0 =  (msm_beta_ll, msm_beta_hh ,mtow_beta_ll , mtow_beta_hh , wtom_beta_ll, wtom_beta_hh, contacts_msm_h , contacts_msm_l_scaler , contacts_msw_h, contacts_msw_l_scaler , contacts_wsm_h , contacts_wsm_l_scaler , contacts_msmw_h_m ,contacts_msmw_l_m_scaler , contacts_msmw_h_w ,contacts_msmw_l_w_scaler, epsilon_param , p_symptoms_men , p_symptoms_women, duration_symptoms_to_treatment_men , duration_symptoms_to_treatment_women, natural_clearance_men, natural_clearance_women, duration_treatment_to_recovery_men, duration_treatment_to_recovery_women, asymptomatic_screening_rate_msm_high, asymptomatic_screening_rate_msmw_high, asymptomatic_screening_rate_msw_high, asymptomatic_screening_rate_wsm_high, asymptomatic_screening_rate_msm_low, asymptomatic_screening_rate_msmw_low, asymptomatic_screening_rate_msw_low, asymptomatic_screening_rate_wsm_low, p_detection_men, p_detection_women, incubation_period_men, incubation_period_women)
            
            
            
            
            # ----------------------------------------------------------------------------
            # run simulation
            # ----------------------------------------------------------------------------
            
            simout = calibration_wrapper(params_tunable0, params_fixed, y0, burnin_T=200, n_iter = 3, T=365)
            
            
            

            # ----------------------------------------------------------------------------
            # Save parameters with output
            # ----------------------------------------------------------------------------
            
            
            
            temp = [ii] + simout + list(params_tunable0)                                         
            
            sample_params_out[ii,:] = temp
            print(ii)
    
        # make dataframe
        sample_params_out = pd.DataFrame(sample_params_out, columns = df_colnames)
        
        # save
        sample_params_out.to_csv(outfolder+'calibration_results_script'+str(aa)+'.csv', 
                                 index=False)
        
        
            
            
            
            