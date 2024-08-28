#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:23:20 2023

@author: kir312
"""



import numpy as np

from numba import njit
#from numba.typed import List





# ----------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------
@njit(debug=True)
def multinomial_for_vector(n,p):

    out = np.zeros(shape=(p.shape[0], n.shape[0]))
    for g in np.arange(n.shape[0]):
        out[:,g]= np.random.multinomial(n[g], p[:,g])

    return out


@njit(debug=True)
def multinomial_for_2d_array(n, p):

    out = np.zeros(shape=(p.shape[0], n.shape[0], n.shape[1]))
    for z in np.arange(n.shape[0]):
        for g in np.arange(n.shape[1]):
            out[:,z,g] = np.random.multinomial(n[z,g], p[:,z,g])

    return out





# ----------------------------------------------------------------------------
# one step simulation fct
# ----------------------------------------------------------------------------

@njit(debug=True)
def one_step(y, params_general, params_transmission, params_symptoms, params_treatment, params_surveillance):
    '''compute one step of the compartmental model

    '''
    
    # expand the compartments
    S, Iinc_r, Iinc_nr, Is_r, Is_nr, Ia_r, Ia_nr, It_nr, Id_nr, It1_s_r_success, It1_s_r_failure, Id_s_r, It1_a_r_success, It1_a_r_failure, Id_a_r = y
    
    # expand list of parameters
    mu_entry, mu_exit = params_general # per-person prob of entry and exit
    transmission_matrix, contact_rates, epsilon = params_transmission 
    p_symptoms, incubation_period = params_symptoms
    duration_symptoms_to_treatment, p_recovery_innate, p_a_treat, p_retreatment, duration_treatment_to_recovery, p_treatment_failure = params_treatment
    p_detection_s, p_detection_a, p_followup = params_surveillance

    # calculate pop sizes
    N_vector = S + Iinc_r + Iinc_nr + Is_r + Is_nr + Ia_r + Ia_nr + It_nr + Id_nr + It1_s_r_success + It1_s_r_failure + Id_s_r + It1_a_r_success + It1_a_r_failure + Id_a_r 
    
    n_groups = N_vector.shape[0]
    
    mixing_matrix = make_mixing_matrix(N_vector = N_vector, 
                                   epsilon =  epsilon, 
                                   contact_rates = contact_rates)
    # sum of I's
    I_r = Iinc_r + Is_r + Ia_r + It1_s_r_failure + It1_a_r_failure
    I_nr = Iinc_nr + Is_nr + Ia_nr 
    
    # -- entry flow
    flow_entry = np.zeros(shape=n_groups)
    
    for g in np.arange(n_groups):
        flow_entry[g] = np.random.binomial(int(N_vector[g]), mu_entry[g])
   
    
    prob_leaving_S = np.zeros(shape=(4,n_groups))

    for g in np.arange(n_groups):
        # (per-person) probability that group g is infected by resistant strain
        p_g_r = np.sum(I_r/N_vector * transmission_matrix[:,g] * mixing_matrix[g,:]/N_vector[g])
        # probability that group g is infected by non-resistant strain
        p_g_nr = np.sum(I_nr/N_vector * transmission_matrix[:,g] * mixing_matrix[g,:]/N_vector[g])
        # insert into matrix of probabilities
        exprob = mu_exit[g]*(S[g]/N_vector[g])
        prob_leaving_S[0,g] = p_g_r * (1-exprob)
        prob_leaving_S[1,g] = p_g_nr * (1-exprob)
        # add exit flow 
        prob_leaving_S[2,g] = exprob
        # add remainder
        prob_leaving_S[3,g] = 1 - (exprob + (p_g_nr * (1-exprob)) + (p_g_r * (1-exprob)))
        
  
    
    # -- exposure flows: from S to exposure
    temp = multinomial_for_vector(n=S.astype(np.int32), p = prob_leaving_S)
    leaving_S_r = temp[0]
    leaving_S_nr = temp[1]
    flow_S_exit = temp[2]    
      
    # -- split infections into symptomatic and asymptomatic
    flow_S_to_Iinc_r = np.zeros(shape=n_groups)
    flow_S_to_Iinc_nr = np.zeros(shape=n_groups)
    for g in np.arange(n_groups):
        flow_S_to_Iinc_r[g] = np.random.binomial(int(leaving_S_r[g]), p_symptoms[g])
        flow_S_to_Iinc_nr[g] = np.random.binomial(int(leaving_S_nr[g]), p_symptoms[g])
    
    # remaining exposures are asymptomatic    
    flow_S_to_Ia_r = leaving_S_r - flow_S_to_Iinc_r
    flow_S_to_Ia_nr = leaving_S_nr - flow_S_to_Iinc_nr
    
   

    
    # -- for symptomatic infection: incubation compartment
    # resistant strain
    p_leaving_Iinc_r = np.zeros(shape=(4,n_groups))
    for g in np.arange(n_groups):
        p_leaving_Iinc_r[0,g] = 1/incubation_period[g] * (1-mu_exit[g]*(Iinc_r[g]/N_vector[g])) # symptom onset (*not exit)
        p_leaving_Iinc_r[1,g] = p_recovery_innate[g] * (1-1/incubation_period[g]) * (1-mu_exit[g]*(Iinc_r[g]/N_vector[g])) # innate recovery (*not symptom onset and not exit)
        p_leaving_Iinc_r[2,g] = mu_exit[g]*(Iinc_r[g]/N_vector[g]) # population exit
        p_leaving_Iinc_r[3,g] = 1 - (p_leaving_Iinc_r[0,g] + p_leaving_Iinc_r[1,g] + p_leaving_Iinc_r[2,g] ) # remainder
    
    # stochastic flows
    temp = multinomial_for_vector(n=(Iinc_r).astype(np.int32), p=p_leaving_Iinc_r)
    flow_Iinc_r_to_Is_r = temp[0]
    flow_Iinc_r_to_S = temp[1] 
    flow_Iinc_r_exit = temp[2]
   
    
    # non-resistant strain
    p_leaving_Iinc_nr = np.zeros(shape=(4,n_groups))
    for g in np.arange(n_groups):
        p_leaving_Iinc_nr[0,g] = 1/incubation_period[g] * (1-mu_exit[g]*(Iinc_nr[g]/N_vector[g])) # symptom onset (*not exit)
        p_leaving_Iinc_nr[1,g] = p_recovery_innate[g] * (1-1/incubation_period[g]) * (1-mu_exit[g]*(Iinc_nr[g]/N_vector[g])) # innate recovery (*not symptom onset and not exit)
        p_leaving_Iinc_nr[2,g] = mu_exit[g]*(Iinc_nr[g]/N_vector[g]) # population exit
        p_leaving_Iinc_nr[3,g] = 1 - (p_leaving_Iinc_nr[0,g] + p_leaving_Iinc_nr[1,g] + p_leaving_Iinc_nr[2,g])
    
    
    # stochastic flows
    temp = multinomial_for_vector(n=(Iinc_nr).astype(np.int32), p=p_leaving_Iinc_nr)
    flow_Iinc_nr_to_Is_nr = temp[0]
    flow_Iinc_nr_to_S = temp[1] 
    flow_Iinc_nr_exit = temp[2]
    
    
    # -- leaving Is
    # resistant strain
    p_leaving_Is_r = np.zeros(shape = (4, n_groups))
    for g in np.arange(n_groups):
        p_leaving_Is_r[0,g] = 1/duration_symptoms_to_treatment[g] * (1-mu_exit[g]*(Is_r[g]/N_vector[g])) # treatment (*not exit)
        p_leaving_Is_r[1,g] = p_recovery_innate[g]  * (1 - 1/duration_symptoms_to_treatment[g]) * (1-mu_exit[g]*(Is_r[g]/N_vector[g])) # innate recovery (*not treatment and not exit)
        p_leaving_Is_r[2,g] = mu_exit[g]*(Is_r[g]/N_vector[g]) # population exit
        p_leaving_Is_r[3,g] = 1 - ( p_leaving_Is_r[0,g] +  p_leaving_Is_r[1,g] +  p_leaving_Is_r[2,g] ) # remainder
        
    # stochastic flows
    temp = multinomial_for_vector(n=(Is_r).astype(np.int32), p=p_leaving_Is_r)
    flow_Is_r_to_treat = temp[0]
    flow_Is_r_to_S = temp[1]
    flow_Is_r_exit = temp[2] 
    # split treatment into 'only treatment' and 'treatment + detection'
    # and into success and failure
    flow_Is_r_to_It1_s_r = np.zeros(shape=n_groups)
    for g in np.arange(n_groups):
        flow_Is_r_to_It1_s_r[g] = np.random.binomial(n=int(flow_Is_r_to_treat[g]), p=1-p_detection_s[g])
    flow_Is_r_to_Id_s_r = flow_Is_r_to_treat - flow_Is_r_to_It1_s_r
    # split flow_Is_r_to_It1_s_r into success and failure
    flow_Is_r_to_It1_s_r_success = np.zeros(shape=n_groups)
    for g in np.arange(n_groups):
        flow_Is_r_to_It1_s_r_success[g] = np.random.binomial(n=int(flow_Is_r_to_It1_s_r[g]), p=1-p_treatment_failure[g])
    flow_Is_r_to_It1_s_r_failure =  flow_Is_r_to_It1_s_r - flow_Is_r_to_It1_s_r_success
    
    
    # non-resistant strain
    p_leaving_Is_nr = np.zeros(shape = (4, n_groups))
    for g in np.arange(n_groups):
        p_leaving_Is_nr[0,g] = 1/duration_symptoms_to_treatment[g] * (1-mu_exit[g]*(Is_nr[g]/N_vector[g])) # treatment (*not exit)
        p_leaving_Is_nr[1,g] = p_recovery_innate[g]  * (1 - 1/duration_symptoms_to_treatment[g]) * (1-mu_exit[g]*(Is_nr[g]/N_vector[g])) # innate recovery (*not treatment and not exit)
        p_leaving_Is_nr[2,g] = mu_exit[g]*(Is_nr[g]/N_vector[g]) # population exit
        p_leaving_Is_nr[3,g] = 1 - (p_leaving_Is_nr[0,g] + p_leaving_Is_nr[1,g] + p_leaving_Is_nr[2,g]) # remainder
    
    # stochastic flows
    temp = multinomial_for_vector(n=(Is_nr).astype(np.int32), p=p_leaving_Is_nr)
    flow_Is_nr_to_treat = temp[0]
    flow_Is_nr_to_S = temp[1]
    flow_Is_nr_exit = temp[2] 
    
    
    # split treatment into 'only treatment' and 'treatment + detection'
    flow_Is_nr_to_It_nr = np.zeros(shape=n_groups)
    for g in np.arange(n_groups):
        flow_Is_nr_to_It_nr[g] = np.random.binomial(n=int(flow_Is_nr_to_treat[g]), p=1-p_detection_s[g])
    flow_Is_nr_to_Id_nr = flow_Is_nr_to_treat - flow_Is_nr_to_It_nr
    
    
    # -- resistant retreatment flow
    p_leaving_It1_s_r_failure = np.zeros(shape = (4, n_groups))
    for g in np.arange(n_groups):
        p_leaving_It1_s_r_failure[0,g] = p_retreatment[g] *  (1-mu_exit[g]*(It1_s_r_failure[g]/N_vector[g]))  # retreatment (*not exit)
        p_leaving_It1_s_r_failure[1,g] = p_recovery_innate[g] * (1-p_retreatment[g])* (1-mu_exit[g]*(It1_s_r_failure[g]/N_vector[g])) # innate recovery (*not exit)
        p_leaving_It1_s_r_failure[2,g] = mu_exit[g]*(It1_s_r_failure[g]/N_vector[g]) # exit
        p_leaving_It1_s_r_failure[3,g] = 1 - (p_leaving_It1_s_r_failure[0,g] + p_leaving_It1_s_r_failure[1,g] + p_leaving_It1_s_r_failure[2,g]) # remainder
        
    # stochastic flows
    temp = multinomial_for_vector(n=(It1_s_r_failure).astype(np.int32), p=p_leaving_It1_s_r_failure)
    flow_It1_s_r_failure_to_Id_s_r = temp[0]
    flow_It1_s_r_failure_to_S = temp[1]
    flow_It1_s_r_failure_exit = temp[2] 
    
   
    
    # -- resistant successful treatment flow
    p_leaving_It1_s_r_success = np.zeros(shape = (3, n_groups))
    for g in np.arange(n_groups):
        p_leaving_It1_s_r_success[0,g] = 1/duration_treatment_to_recovery[g] * (1-mu_exit[g]*(It1_s_r_success[g]/N_vector[g])) # recovery (*not exit)
        p_leaving_It1_s_r_success[1,g] = mu_exit[g]*(It1_s_r_success[g]/N_vector[g]) # exit
        p_leaving_It1_s_r_success[2,g] = 1 - (p_leaving_It1_s_r_success[0,g] + p_leaving_It1_s_r_success[1,g]) # remainder
        
    # stochastic flows
    temp = multinomial_for_vector((It1_s_r_success).astype(np.int32), p_leaving_It1_s_r_success)
    flow_It1_s_r_success_to_S = temp[0]
    flow_It1_s_r_success_exit = temp[1]
    


    # -- leaving Ia (to screening or innate recovery)
    # resistant strain
    p_leaving_Ia_r = np.zeros(shape = (4, n_groups))
    for g in np.arange(n_groups):
        p_leaving_Ia_r[0,g] = p_a_treat[g] * (1-mu_exit[g]*(Ia_r[g]/N_vector[g])) # combined treatment flows (*not exit)
        p_leaving_Ia_r[1,g] = p_recovery_innate[g] * (1-p_a_treat[g]) * (1-mu_exit[g]*(Ia_r[g]/N_vector[g])) # innate recovery (*not treatment and not exit)
        p_leaving_Ia_r[2,g] = mu_exit[g]*(Ia_r[g]/N_vector[g]) # population exit
        p_leaving_Ia_r[3,g] = 1 - (p_leaving_Ia_r[0,g] + p_leaving_Ia_r[1,g] + p_leaving_Ia_r[2,g]) # remainder
    
    # stochastic flows
    temp = multinomial_for_vector(n=(Ia_r).astype(np.int32), p=p_leaving_Ia_r)
    flow_Ia_r_to_treat = temp[0]
    flow_Ia_r_to_S = temp[1]
    flow_Ia_r_exit = temp[2]
    
    # split treatment into 'only treatment' and 'treatment + detection'
    # and into treatment success and treatment failure
    flow_Ia_r_to_It1_a_r = np.zeros(shape=n_groups)
    for g in np.arange(n_groups):
        flow_Ia_r_to_It1_a_r[g] = np.random.binomial(n=int(flow_Ia_r_to_treat[g]), p=1-p_detection_a[g])
    flow_Ia_r_to_Id_a_r = flow_Ia_r_to_treat - flow_Ia_r_to_It1_a_r
    # split treatment into success vs failure
    flow_Ia_r_to_It1_a_r_success = np.zeros(shape=n_groups) 
    for g in np.arange(n_groups):
        flow_Ia_r_to_It1_a_r_success[g] = np.random.binomial(n=int(flow_Ia_r_to_It1_a_r[g]), p=1-p_treatment_failure[g])
    flow_Ia_r_to_It1_a_r_failure = flow_Ia_r_to_It1_a_r - flow_Ia_r_to_It1_a_r_success

    # -- non-resistant successful treatment flow
    # recovery after successful treatment
    p_leaving_It1_a_r_success = np.zeros(shape = (3, n_groups))
    for g in np.arange(n_groups):
        p_leaving_It1_a_r_success[0,g] = 1/duration_treatment_to_recovery[g] * (1-mu_exit[g]*(It1_a_r_success[g]/N_vector[g])) # recovery (*not exit)
        p_leaving_It1_a_r_success[1,g] = mu_exit[g]*(It1_a_r_success[g]/N_vector[g]) # exit
        p_leaving_It1_a_r_success[2,g] = 1 - (p_leaving_It1_a_r_success[0,g] + p_leaving_It1_a_r_success[1,g]) # remainder
    
    # stochastic flows
    temp = multinomial_for_vector((It1_a_r_success).astype(np.int32), p_leaving_It1_a_r_success)
    flow_It1_a_r_success_to_S = temp[0]
    flow_It1_a_r_success_exit = temp[1]

   
    
    # non-resistant strain
    p_leaving_Ia_nr = np.zeros(shape = (4, n_groups))
    for g in np.arange(n_groups):
        p_leaving_Ia_nr[0,g] = p_a_treat[g] * (1-mu_exit[g]*(Ia_nr[g]/N_vector[g])) # combined treatment flows (*not exit)
        p_leaving_Ia_nr[1,g] = p_recovery_innate[g] * (1-p_a_treat[g]) * (1-mu_exit[g]*(Ia_nr[g]/N_vector[g])) # innate recovery (*not treatment and not exit)
        p_leaving_Ia_nr[2,g] = mu_exit[g]*(Ia_nr[g]/N_vector[g]) # population exit
        p_leaving_Ia_nr[3,g] = 1 - (p_leaving_Ia_nr[0,g] + p_leaving_Ia_nr[1,g] + p_leaving_Ia_nr[2,g]) # remainder

    # stochastic flows
    temp = multinomial_for_vector(n=(Ia_nr).astype(np.int32), p=p_leaving_Ia_nr)
    flow_Ia_nr_to_treat = temp[0]
    flow_Ia_nr_to_S = temp[1]
    flow_Ia_nr_exit = temp[2]
    # split treatment into 'only treatment' and 'treatment + detection'
    flow_Ia_nr_to_It_nr = np.zeros(shape=n_groups)
    for g in np.arange(n_groups):
        flow_Ia_nr_to_It_nr[g] = np.random.binomial(n=int(flow_Ia_nr_to_treat[g]), p=1-p_detection_a[g])
    flow_Ia_nr_to_Id_nr = flow_Ia_nr_to_treat - flow_Ia_nr_to_It_nr
    
    
    # -- asymptomatic retreatment: flows It1 -> Id and It1 -> Ia
    p_leaving_It1_a_r_failure = np.zeros(shape = (4, n_groups))
    for g in np.arange(n_groups):
        p_leaving_It1_a_r_failure[0,g] = (p_a_treat[g] + p_followup[g]) *(1-mu_exit[g]*(It1_a_r_failure[g]/N_vector[g])) # follow-up OR second screening (*not exit)
        p_leaving_It1_a_r_failure[1,g] = p_recovery_innate[g] * (1-(p_a_treat[g] + p_followup[g])) * (1-mu_exit[g]*(It1_a_r_failure[g]/N_vector[g])) # innate recovery (*not follow-up/screening and not exit)
        p_leaving_It1_a_r_failure[2,g] = mu_exit[g]*(It1_a_r_failure[g]/N_vector[g]) # exit
        p_leaving_It1_a_r_failure[3,g] = 1 - (p_leaving_It1_a_r_failure[0,g] + p_leaving_It1_a_r_failure[1,g] + p_leaving_It1_a_r_failure[2,g])
        
    # stochastic flows
    temp = multinomial_for_vector(n=(It1_a_r_failure).astype(np.int32), p=p_leaving_It1_a_r_failure)
    flow_It1_a_r_failure_to_Id_a_r = temp[0]
    flow_It1_a_r_failure_to_S = temp[1]
    flow_It1_a_r_failure_exit = temp[2]
    
    
    # -- return to S upon treatment
    # Id_s_r
    p_leaving_Id_s_r = np.zeros(shape = (3, n_groups))
    for g in np.arange(n_groups):
        p_leaving_Id_s_r[0,g] = 1/duration_treatment_to_recovery[g] * (1-mu_exit[g]*(Id_s_r[g]/N_vector[g])) # recovery (*not exit)
        p_leaving_Id_s_r[1,g] = mu_exit[g]*(Id_s_r[g]/N_vector[g]) # exit
        p_leaving_Id_s_r[2,g] = 1 - (p_leaving_Id_s_r[0,g] + p_leaving_Id_s_r[1,g]) # remainder
        
    # stochastic flows
    temp = multinomial_for_vector((Id_s_r).astype(np.int32), p_leaving_Id_s_r)
    flow_Id_s_r_to_S = temp[0]
    flow_Id_s_r_exit = temp[1]
    
    
    # It_nr
    p_leaving_It_nr = np.zeros(shape = (3, n_groups))
    for g in np.arange(n_groups):
        p_leaving_It_nr[0,g] = 1/duration_treatment_to_recovery[g] * (1-mu_exit[g]*(It_nr[g]/N_vector[g])) # recovery (*not exit)
        p_leaving_It_nr[1,g] = mu_exit[g]*(It_nr[g]/N_vector[g]) # exit
        p_leaving_It_nr[2,g] = 1 - (p_leaving_It_nr[0,g] + p_leaving_It_nr[1,g]) # remainder
        
    # stochastic flows
    temp = multinomial_for_vector((It_nr).astype(np.int32), p_leaving_It_nr)
    flow_It_nr_to_S = temp[0]
    flow_It_nr_exit = temp[1]
    
    
    # Id_nr
    p_leaving_Id_nr = np.zeros(shape = (3, n_groups))
    for g in np.arange(n_groups):
        p_leaving_Id_nr[0,g] = 1/duration_treatment_to_recovery[g] * (1-mu_exit[g]*(Id_nr[g]/N_vector[g])) # recovery (*not exit)
        p_leaving_Id_nr[1,g] = mu_exit[g]*(Id_nr[g]/N_vector[g]) # exit
        p_leaving_Id_nr[2,g] = 1 - (p_leaving_Id_nr[0,g] + p_leaving_Id_nr[1,g]) # remainder
    

    # stochastic flows
    temp = multinomial_for_vector((Id_nr).astype(np.int32), p_leaving_Id_nr)
    flow_Id_nr_to_S = temp[0]
    flow_Id_nr_exit = temp[1]
    
    
    # Id_a_r
    p_leaving_Id_a_r = np.zeros(shape = (3, n_groups))
    for g in np.arange(n_groups):
        p_leaving_Id_a_r[0,g] = 1/duration_treatment_to_recovery[g] * (1-mu_exit[g]*(Id_a_r[g]/N_vector[g])) # recovery (*not exit)
        p_leaving_Id_a_r[1,g] = mu_exit[g]*(Id_a_r[g]/N_vector[g]) # exit
        p_leaving_Id_a_r[2,g] = 1 - (p_leaving_Id_a_r[0,g] + p_leaving_Id_a_r[1,g]) # remainder
    
    # stochastic flows
    temp = multinomial_for_vector((Id_a_r).astype(np.int32), p_leaving_Id_a_r)
    flow_Id_a_r_to_S = temp[0]
    flow_Id_a_r_exit = temp[1]

    



    # ------------------
    # Flows
    # ------------------
    # combine  some of the flows
    
    # innate recovery and recovery from treatment
    all_flows_to_S = flow_Iinc_r_to_S + flow_Iinc_nr_to_S \
                                + flow_Ia_nr_to_S + flow_Ia_r_to_S \
                                + flow_Is_r_to_S + flow_Is_nr_to_S \
                                + flow_It1_a_r_success_to_S + flow_It1_a_r_failure_to_S \
                                + flow_It1_s_r_success_to_S + flow_It1_s_r_failure_to_S \
                                + flow_Id_s_r_to_S + flow_Id_a_r_to_S \
                                + flow_It_nr_to_S + flow_Id_nr_to_S 
                            
    new_detected_r =  flow_It1_s_r_failure_to_Id_s_r + flow_It1_a_r_failure_to_Id_a_r + flow_Is_r_to_Id_s_r + flow_Ia_r_to_Id_a_r
    new_detected_nr =  flow_Is_nr_to_Id_nr + flow_Ia_nr_to_Id_nr
    new_cases_r = flow_S_to_Ia_r + flow_S_to_Iinc_r
    new_cases_nr = flow_S_to_Ia_nr + flow_S_to_Iinc_nr
    
    # ------------------
    
    dS = + (flow_entry + all_flows_to_S) \
         - (leaving_S_r + leaving_S_nr + flow_S_exit)
    
    dIinc_r = + (flow_S_to_Iinc_r) \
              - (flow_Iinc_r_to_S + flow_Iinc_r_to_Is_r + flow_Iinc_r_exit)
    
    dIinc_nr = + (flow_S_to_Iinc_nr) \
               - (flow_Iinc_nr_to_S + flow_Iinc_nr_to_Is_nr + flow_Iinc_nr_exit)
    
    dIs_r = + (flow_Iinc_r_to_Is_r) \
            - (flow_Is_r_to_treat + flow_Is_r_to_S + flow_Is_r_exit)
    
    dIs_nr = + (flow_Iinc_nr_to_Is_nr) \
             - (flow_Is_nr_to_treat + flow_Is_nr_to_S + flow_Is_nr_exit)
    
    dIa_nr = + (flow_S_to_Ia_nr) \
             - (flow_Ia_nr_to_treat + flow_Ia_nr_to_S + flow_Ia_nr_exit)
    
    dIa_r = + (flow_S_to_Ia_r ) \
            - ( flow_Ia_r_to_treat + flow_Ia_r_to_S + flow_Ia_r_exit)
    
    dIt_nr = + (flow_Is_nr_to_It_nr + flow_Ia_nr_to_It_nr) \
             - (flow_It_nr_to_S + flow_It_nr_exit)
            
    dId_nr = + (flow_Is_nr_to_Id_nr + flow_Ia_nr_to_Id_nr) \
             - (flow_Id_nr_to_S + flow_Id_nr_exit)
            
    dIt1_s_r_success = + (flow_Is_r_to_It1_s_r_success) \
                        - (flow_It1_s_r_success_to_S + flow_It1_s_r_success_exit)

    dIt1_s_r_failure = + (flow_Is_r_to_It1_s_r_failure) \
                        - (flow_It1_s_r_failure_to_Id_s_r + flow_It1_s_r_failure_to_S + flow_It1_s_r_failure_exit)
            
    dId_s_r = + (flow_It1_s_r_failure_to_Id_s_r + flow_Is_r_to_Id_s_r) \
              - ( flow_Id_s_r_to_S + flow_Id_s_r_exit)
    
    dIt1_a_r_success = + (flow_Ia_r_to_It1_a_r_success) \
                        - (flow_It1_a_r_success_to_S + flow_It1_a_r_success_exit)
    dIt1_a_r_failure = + (flow_Ia_r_to_It1_a_r_failure) \
                        - (flow_It1_a_r_failure_to_Id_a_r + flow_It1_a_r_failure_to_S + flow_It1_a_r_failure_exit)
                
    dId_a_r  =  + (flow_Ia_r_to_Id_a_r + flow_It1_a_r_failure_to_Id_a_r) \
                - (flow_Id_a_r_to_S + flow_Id_a_r_exit)
            
    
    # update compartments
    S = S + dS
    Iinc_r = Iinc_r + dIinc_r
    Iinc_nr = Iinc_nr + dIinc_nr
    Is_r = Is_r + dIs_r
    Is_nr = Is_nr + dIs_nr
    Ia_r = Ia_r + dIa_r
    Ia_nr = Ia_nr + dIa_nr
    It_nr = It_nr + dIt_nr
    Id_nr = Id_nr + dId_nr
    It1_s_r_success = It1_s_r_success + dIt1_s_r_success
    It1_s_r_failure = It1_s_r_failure + dIt1_s_r_failure
    Id_s_r = Id_s_r + dId_s_r
    It1_a_r_success = It1_a_r_success + dIt1_a_r_success
    It1_a_r_failure = It1_a_r_failure + dIt1_a_r_failure
    Id_a_r = Id_a_r + dId_a_r
    
    y_new = S, Iinc_r, Iinc_nr, Is_r, Is_nr, Ia_r, Ia_nr, It_nr, Id_nr, It1_s_r_success, It1_s_r_failure, Id_s_r, It1_a_r_success, It1_a_r_failure, Id_a_r

            
    return y_new, new_detected_r, new_detected_nr, new_cases_r, new_cases_nr





# ----------------------------------------------------------------------------
# mixing matrix
# ----------------------------------------------------------------------------


@njit(debug=True)
def make_mixing_matrix(N_vector, epsilon, contact_rates):
    
    # expand the parameters
    contacts_msm_h, contacts_msm_l, contacts_msmw_h_m, contacts_msmw_l_m, contacts_msmw_h_w, contacts_msmw_l_w, contacts_msw_h, contacts_msw_l, contacts_wsm_h, contacts_wsm_l = contact_rates
    
    N_msm_h, N_msm_l, N_msmw_h, N_msmw_l, N_msw_h, N_msw_l, N_wsm_h, N_wsm_l = N_vector

    
    # demand-supply-balancing parameter
    balance_parameter = 0.4
    
    # 1) calculate demands
    
    # -- MSM network --
    # -- within group
    # msm high risk -> demand for high risk msm and msmw
    # -> epsilon * total msm demand * proportion of supply by msm_h of total supply of msm_h and msmw_h
    msm_hh = epsilon * (contacts_msm_h * N_msm_h) * ( (contacts_msm_h * N_msm_h) / ((contacts_msm_h * N_msm_h) + (contacts_msmw_h_m * N_msmw_h)) )
    msm_h_msmw_h = epsilon * (contacts_msm_h * N_msm_h) * ((contacts_msmw_h_m * N_msmw_h) / ((contacts_msm_h * N_msm_h) + (contacts_msmw_h_m * N_msmw_h))) 
    # msm_l -> demand for low risk msm and msmw                                                                                                  
    msm_ll = epsilon * (contacts_msm_l * N_msm_l) * ( (contacts_msm_l * N_msm_l) / ((contacts_msm_l * N_msm_l) + (contacts_msmw_l_m * N_msmw_l)) )
    msm_l_msmw_l = epsilon * (contacts_msm_l * N_msm_l) * ((contacts_msmw_l_m * N_msmw_l) / ((contacts_msm_l * N_msm_l) + (contacts_msmw_l_m * N_msmw_l))) 
    # msmw m high risk -> demand for high risk msm and msmw
    msmw_hh = epsilon * (contacts_msmw_h_m * N_msmw_h) * ( (contacts_msmw_h_m * N_msmw_h) / ((contacts_msm_h * N_msm_h) + (contacts_msmw_h_m * N_msmw_h)) )
    msmw_h_msm_h = epsilon * (contacts_msmw_h_m * N_msmw_h) * ( (contacts_msm_h * N_msm_h) / ((contacts_msm_h * N_msm_h) + (contacts_msmw_h_m * N_msmw_h)) )
    # msmw_l -> demand for low risk msm and msmw                                                                                                  
    msmw_ll = epsilon * (contacts_msmw_l_m * N_msmw_l) * ( (contacts_msmw_l_m * N_msmw_l) / ((contacts_msm_l * N_msm_l) + (contacts_msmw_l_m * N_msmw_l)) )
    msmw_l_msm_l = epsilon * (contacts_msmw_l_m * N_msmw_l) * ((contacts_msm_l * N_msm_l) / ((contacts_msm_l * N_msm_l) + (contacts_msmw_l_m * N_msmw_l))) 
    
    # -- between groups
    # from msm_h to msm_l and msmw_l
    msm_hl = (1-epsilon) * (contacts_msm_h * N_msm_h) * ( (contacts_msm_l * N_msm_l) / ((contacts_msm_l * N_msm_l) + (contacts_msmw_l_m * N_msmw_l)) )
    msm_h_msmw_l = (1-epsilon) * (contacts_msm_h * N_msm_h) * ( (contacts_msmw_l_m * N_msmw_l) / ((contacts_msm_l * N_msm_l) + (contacts_msmw_l_m * N_msmw_l)) )
    # from msm_l to msm_h and msmw_h
    msm_lh = (1-epsilon) * (contacts_msm_l * N_msm_l) * ( (contacts_msm_h * N_msm_h) / ((contacts_msm_h * N_msm_h) + (contacts_msmw_h_m * N_msmw_h)) )
    msm_l_msmw_h = (1-epsilon) * (contacts_msm_l * N_msm_l) * ( (contacts_msmw_h_m * N_msmw_h) / ((contacts_msm_h * N_msm_h) + (contacts_msmw_h_m * N_msmw_h)) )
    # from msmw_h to msm_l and msmw_l
    msmw_h_msm_l = (1-epsilon) * (contacts_msmw_h_m * N_msmw_h) * ( (contacts_msm_l * N_msm_l) / ((contacts_msm_l * N_msm_l) + (contacts_msmw_l_m * N_msmw_l)) )
    msmw_hl = (1-epsilon) * (contacts_msmw_h_m * N_msmw_h) * ( (contacts_msmw_l_m * N_msmw_l) / ((contacts_msm_l * N_msm_l) + (contacts_msmw_l_m * N_msmw_l)) )
    # from msmw_l to msm_h and msmw_h
    msmw_l_msm_h = (1-epsilon) * (contacts_msmw_l_m * N_msmw_l) * ( (contacts_msm_h * N_msm_h) / ((contacts_msm_h * N_msm_h) + (contacts_msmw_h_m * N_msmw_h)) )
    msmw_lh = (1-epsilon) * (contacts_msmw_l_m * N_msmw_l) * ( (contacts_msmw_h_m * N_msmw_h) / ((contacts_msm_h * N_msm_h) + (contacts_msmw_h_m * N_msmw_h)) )
    # msmw h to l
    
    
    # -- heterosexual network --
    # -- within group
    # msw
    msw_h_wsm_h = epsilon * (contacts_msw_h * N_msw_h)
    msw_l_wsm_l = epsilon * (contacts_msw_l * N_msw_l)
    # wsm
    wsm_h_msw_h = epsilon * (contacts_wsm_h * N_wsm_h) * ((contacts_msw_h * N_msw_h) / ( (contacts_msw_h * N_msw_h) + (contacts_msmw_h_w * N_msmw_h) ))
    wsm_h_msmw_h = epsilon * (contacts_wsm_h * N_wsm_h) * ((contacts_msmw_h_w * N_msmw_h) / ( (contacts_msw_h * N_msw_h) + (contacts_msmw_h_w * N_msmw_h) ))
    wsm_l_msw_l = epsilon * (contacts_wsm_l * N_wsm_l) * ((contacts_msw_l * N_msw_l) / ( (contacts_msw_l * N_msw_l) + (contacts_msmw_l_w * N_msmw_l) ))
    wsm_l_msmw_l = epsilon * (contacts_wsm_l * N_wsm_l) * ((contacts_msmw_l_w * N_msmw_l) / ( (contacts_msw_l * N_msw_l) + (contacts_msmw_l_w * N_msmw_l) ))
    # msmw
    msmw_h_wsm_h = epsilon * (contacts_msmw_h_w * N_msmw_h)
    msmw_l_wsm_l = epsilon * (contacts_msmw_l_w * N_msmw_l)
    
    # -- between group
    # - high to low
    # msw_h to wsm_l
    msw_h_wsm_l = (1-epsilon) * (contacts_msw_h * N_msw_h)
    msw_l_wsm_h = (1-epsilon) * (contacts_msw_l * N_msw_l)
    # wsm_h to msm_l and msmw_l
    wsm_h_msw_l = (1-epsilon) * (contacts_wsm_h * N_wsm_h) * ((contacts_msw_l * N_msw_l) / ( (contacts_msw_l * N_msw_l) + (contacts_msmw_l_w * N_msmw_l) ))
    wsm_h_msmw_l = (1-epsilon) * (contacts_wsm_h * N_wsm_h) * ((contacts_msmw_l_w * N_msmw_l) / ( (contacts_msw_l * N_msw_l) + (contacts_msmw_l_w * N_msmw_l) ))
    wsm_l_msw_h = (1-epsilon) * (contacts_wsm_l * N_wsm_l) * ((contacts_msw_h * N_msw_h) / ( (contacts_msw_h * N_msw_h) + (contacts_msmw_h_w * N_msmw_h) ))
    wsm_l_msmw_h = (1-epsilon) * (contacts_wsm_l * N_wsm_l) * ((contacts_msmw_h_w * N_msmw_h) / ( (contacts_msw_h * N_msw_h) + (contacts_msmw_h_w * N_msmw_h) ))
    # msmw_h to wsm_l
    msmw_h_wsm_l = (1-epsilon) * (contacts_msmw_h_w * N_msmw_h)
    msmw_l_wsm_h = (1-epsilon) * (contacts_msmw_l_w * N_msmw_l)
    
    
    # 2) balance supply and demand
    # diagonal entries are fixed
    # make off-diagonal entries symmetric, by:
      # - calculating ratio
      # - adjusting entries with ratio
    b12 = msm_hl/msm_lh
    msm_hl = msm_hl * b12**(balance_parameter-1)
    msm_lh = msm_lh * b12**balance_parameter
    b13 = msm_h_msmw_h/msmw_h_msm_h
    msm_h_msmw_h = msm_h_msmw_h * b13**(balance_parameter-1)
    msmw_h_msm_h = msmw_h_msm_h * b12**balance_parameter
    b14 = msm_h_msmw_l/msmw_l_msm_h
    msm_h_msmw_l = msm_h_msmw_l * b14**(balance_parameter-1)
    msmw_l_msm_h = msmw_l_msm_h * b14**balance_parameter
    b23 = msm_l_msmw_h/msmw_h_msm_l
    msm_l_msmw_h = msm_l_msmw_h * b23**(balance_parameter-1)
    msmw_h_msm_l = msmw_h_msm_l * b23**balance_parameter
    b24 = msm_l_msmw_l/msmw_l_msm_l
    msm_l_msmw_l = msm_l_msmw_l * b24**(balance_parameter-1)
    msmw_l_msm_l = msmw_l_msm_l * b24**balance_parameter
    b34  = msmw_hl/msmw_lh
    msmw_hl = msmw_hl * b34**(balance_parameter-1)
    msmw_lh = msmw_lh * b34**balance_parameter
    b37 = msmw_h_wsm_h/wsm_h_msmw_h
    msmw_h_wsm_h = msmw_h_wsm_h * b37**(balance_parameter-1)
    wsm_h_msmw_h = wsm_h_msmw_h * b37**balance_parameter
    b38 = msmw_h_wsm_l/wsm_l_msmw_h
    msmw_h_wsm_l = msmw_h_wsm_l * b38**(balance_parameter-1)
    wsm_l_msmw_h = wsm_l_msmw_h * b38**balance_parameter
    b47 = msmw_l_wsm_h/wsm_h_msmw_l
    msmw_l_wsm_h = msmw_l_wsm_h * b47**(balance_parameter-1)
    wsm_h_msmw_l = wsm_h_msmw_l * b47**balance_parameter
    b48 = msmw_l_wsm_l/wsm_l_msmw_l
    msmw_l_wsm_l = msmw_l_wsm_l * b48**(balance_parameter-1)
    wsm_l_msmw_l = wsm_l_msmw_l * b48**balance_parameter
    b57 = msw_h_wsm_h/wsm_h_msw_h
    msw_h_wsm_h = msw_h_wsm_h * b57**(balance_parameter-1)
    wsm_h_msw_h = wsm_h_msw_h * b57**balance_parameter
    b58 = msw_h_wsm_l/wsm_l_msw_h
    msw_h_wsm_l = msw_h_wsm_l * b58**(balance_parameter-1)
    wsm_l_msw_h = wsm_l_msw_h * b58**balance_parameter
    b67 = msw_l_wsm_h/wsm_h_msw_l
    msw_l_wsm_h = msw_l_wsm_h * b67**(balance_parameter-1)
    wsm_h_msw_l = wsm_h_msw_l * b67**balance_parameter
    b68 = msw_l_wsm_l/wsm_l_msw_l
    msw_l_wsm_l = msw_l_wsm_l * b68**(balance_parameter-1)
    wsm_l_msw_l = wsm_l_msw_l * b68**balance_parameter
    
    
    
    mixing_matrix = np.array([ [msm_hh, msm_hl, msm_h_msmw_h, msm_h_msmw_l, 0, 0, 0, 0],
                                    [msm_lh, msm_ll, msm_l_msmw_h, msm_l_msmw_l, 0, 0, 0, 0],
                                    [msmw_h_msm_h, msmw_h_msm_l, msmw_hh, msmw_hl, 0, 0, msmw_h_wsm_h, msmw_h_wsm_l],
                                    [msmw_l_msm_h, msmw_l_msm_l, msmw_lh, msmw_ll, 0, 0, msmw_l_wsm_h, msmw_l_wsm_l],
                                    [0, 0, 0, 0, 0, 0, msw_h_wsm_h, msw_h_wsm_l],
                                    [0, 0, 0, 0, 0, 0, msw_l_wsm_h, msw_l_wsm_l],
                                    [0, 0, wsm_h_msmw_h, wsm_h_msmw_l, wsm_h_msw_h, wsm_h_msw_l, 0, 0],
                                    [0, 0, wsm_l_msmw_h, wsm_l_msmw_l, wsm_l_msw_h, wsm_l_msw_l, 0, 0]])
    
    # convert to daily values
    mixing_matrix = mixing_matrix/365
    
    return mixing_matrix    







# ----------------------------------------------------------------------------
# run the simulation over time
# ----------------------------------------------------------------------------


def run_simulation(n_iter, T, y0, params0):
    """run simulation over time

    """
    
    
    
    # -- expand the parameters
    params_general0, params_transmission0, params_symptoms0, params_treatment0, params_surveillance0 = params0
    
    
    # -- prep the outputs    
    # new detections & cases
    new_detected_r_stack = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    new_detected_nr_stack = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    new_cases_r_stack = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    new_cases_nr_stack = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    # cumulative detections & cases
    cum_detected_r_stack = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    cum_detected_nr_stack = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    cum_cases_r_stack = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    cum_cases_nr_stack = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    # current undetected
    current_undetected_r = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    current_undetected_nr = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    # current cases (detected + undetected)
    current_cases_r = np.zeros(shape=(n_iter, T, y0[0].shape[0]))
    current_cases_nr = np.zeros(shape=(n_iter, T, y0[0].shape[0]))

    
    # ----------------------------------------
    # iterate over simulations
    # ----------------------------------------
    
    for i in np.arange(n_iter):
        # --at the start of each iteration, reset the starting conditions
        # compartments
        y = y0        
        # parameters
        params_general = params_general0
        params_transmission = params_transmission0
        params_symptoms = params_symptoms0
        params_treatment = params_treatment0
        params_surveillance = params_surveillance0
    
        # ----------------------------------------
        # iterate over time
        # ----------------------------------------
        
        for t in np.arange(T):
            # -- run a step
            y_new, new_detected_r, new_detected_nr, new_cases_r, new_cases_nr = one_step(y, params_general, params_transmission, params_symptoms, params_treatment, params_surveillance)
            
            # -- append outputs            
            # stacked new detections and cases
            new_detected_r_stack[i,t,:] = new_detected_r
            new_detected_nr_stack[i,t,:] = new_detected_nr
            new_cases_r_stack[i,t,:] = new_cases_r
            new_cases_nr_stack[i,t,:] = new_cases_nr
            
            # cumulative detections and cases
            if t==0:
                cum_detected_r_stack[i,t,:] = new_detected_r
                cum_detected_nr_stack[i,t,:] = new_detected_nr
                cum_cases_r_stack[i,t,:] = new_cases_r
                cum_cases_nr_stack[i,t,:] = new_cases_nr
            elif t>0:
                cum_detected_r_stack[i,t,:] = cum_detected_r_stack[i,t-1] + new_detected_r
                cum_detected_nr_stack[i,t,:] = cum_detected_nr_stack[i,t-1] + new_detected_nr
                cum_cases_r_stack[i,t,:] = cum_cases_r_stack[i,t-1] + new_cases_r
                cum_cases_nr_stack[i,t,:] = cum_cases_nr_stack[i,t-1] + new_cases_nr
     
            # current undetected cases (Iinc + Is + Ia + It1)
            current_undetected_r[i,t,:] = y_new[1] + y_new[3] + y_new[5] + y_new[9] + y_new[10] + y_new[12] + y_new[13]
            current_undetected_nr[i,t,:] = y_new[2] + y_new[4] + y_new[6] + y_new[7]
            
            # current cases (active infections): (Iinc + Is + Ia + It1) + Id
            current_cases_r[i,t,:] = y_new[1] + y_new[3] + y_new[5] + y_new[9] + y_new[10] + y_new[12] + y_new[13] + y_new[11] + y_new[14]
            current_cases_nr[i,t,:] = y_new[2] + y_new[4] + y_new[6] + y_new[7] + y_new[8]
            
            
            # -- updates before next time step
            # update y
            y = y_new

         
    return current_undetected_r, current_undetected_nr, new_detected_r_stack, new_detected_nr_stack, new_cases_r_stack, new_cases_nr_stack, cum_detected_r_stack, cum_detected_nr_stack, cum_cases_r_stack, cum_cases_nr_stack, current_cases_r, current_cases_nr







# ----------------------------------------------------------------------------
# prep inputs - for simulations with accepted parameters
# ----------------------------------------------------------------------------

def prep_input_parameters(params_fixed, params_tunable):
    
    # expand the fixed parameters
    N_vector, p_symptoms, mu_entry, mu_exit, p_retreatment, p_followup, p_treatment_failure = params_fixed

    # ----- input preparation for simulation
    # expand the tunable parameters
    msm_beta_ll, msm_beta_hh ,mtow_beta_ll , mtow_beta_hh , wtom_beta_ll, wtom_beta_hh, contacts_msm_h , contacts_msm_l_scaler , contacts_msw_h, contacts_msw_l_scaler , contacts_wsm_h , contacts_wsm_l_scaler , contacts_msmw_h_m ,contacts_msmw_l_m_scaler , contacts_msmw_h_w ,contacts_msmw_l_w_scaler, epsilon_param , p_symptoms_men , p_symptoms_women, duration_symptoms_to_treatment_men , duration_symptoms_to_treatment_women, natural_clearance_men, natural_clearance_women, duration_treatment_to_recovery_men, duration_treatment_to_recovery_women, asymptomatic_screening_rate_msm_high, asymptomatic_screening_rate_msmw_high, asymptomatic_screening_rate_msw_high, asymptomatic_screening_rate_wsm_high, asymptomatic_screening_rate_msm_low, asymptomatic_screening_rate_msmw_low, asymptomatic_screening_rate_msw_low, asymptomatic_screening_rate_wsm_low, p_detection_men, p_detection_women, incubation_period_men, incubation_period_women = params_tunable
    
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
    
    return params0_input









# ----------------------------------------------------------------------------
# fcts for ABC
# ----------------------------------------------------------------------------

@njit(debug=True)
def abc_sequence_timemask(cum_detected_r_stack, cumulative_filter_sequence=[2.,2.,2.]):
 
    n_iter = cum_detected_r_stack.shape[0]
    sequence_length = len(cumulative_filter_sequence)
    T = cum_detected_r_stack.shape[1]
    
    # sum over groups
    cum_detected_r_stack_AGG = np.sum(cum_detected_r_stack, axis=-1)
    
    # prep output
    iterations_timemask = np.zeros(shape=(n_iter,T), dtype=np.bool_)
    iterations_timemask_endseq = np.zeros(shape=(n_iter,T), dtype=np.bool_)
    iterations_mask = np.zeros(shape=(n_iter), dtype=np.bool_)
    # loop over the iterations
    for aa in np.arange(n_iter):
        # select a single iteration from the agg output
        one_iteration = cum_detected_r_stack_AGG[aa,:]
        
        # loop over time
        for tt in np.arange(T-sequence_length):
            
            # run sequence check
            filter_indicator = list(one_iteration[tt:tt+sequence_length]) == cumulative_filter_sequence
                
            # if indicator is true, can stop looping over time and update the mask
            
            if filter_indicator==True:
                # if remains true, update output
                # - time mask at first detection
                iterations_timemask[aa,tt] = filter_indicator
                # - time mask at end of sequence of zeros
                iterations_timemask_endseq[aa,tt+sequence_length] = filter_indicator
                # - iterations mask (without specifying time)
                iterations_mask[aa] = filter_indicator
                #print('made an update')
                break
    # calculate the percentage of iterations that remains after filtering
    percent_filtered = np.sum(iterations_mask) / iterations_mask.shape[0]

    
    return percent_filtered, iterations_mask, iterations_timemask, iterations_timemask_endseq



# implement abc filtering for a sequence of zeros - counting from end of sequence
@njit(debug=True)
def loop_over_zeros_abc_sequence_timemask(cum_detected_r_stack, max_zeros, n_iter, T, initial_detections):
    
    out = np.zeros(shape=(n_iter*T, max_zeros))
    # loop over zeros
    for n_zeros in np.arange(0,max_zeros):
        filter_seq = [initial_detections]*(n_zeros+1)
    
        # run abc filtering
        percent_filtered_temp, iterations_mask_sequence_temp, iterations_timemask_sequence_temp, iterations_timemask_sequence_endseq_temp = abc_sequence_timemask(cum_detected_r_stack, 
                                                                                       cumulative_filter_sequence=filter_seq)
    
        # reshape
        temp0 = iterations_timemask_sequence_endseq_temp.astype(np.int32).reshape(n_iter*T)
        
        # combine
        out[:,n_zeros] = temp0
       
    return out



@njit(debug=True)
def loop_over_SOME_zeros_abc_sequence_timemask(cum_detected_r_stack, set_of_zeros, n_iter, T, initial_detections):
    
    out = np.zeros(shape=(n_iter*T, len(set_of_zeros)))
    # loop over zeros
    for nn, n_zeros in enumerate(set_of_zeros):
        filter_seq = [initial_detections]*(n_zeros+1)
    
        # run abc filtering
        percent_filtered_temp, iterations_mask_sequence_temp, iterations_timemask_sequence_temp, iterations_timemask_sequence_endseq_temp = abc_sequence_timemask(cum_detected_r_stack, 
                                                                                       cumulative_filter_sequence=filter_seq)
    
        # reshape
        temp0 = iterations_timemask_sequence_endseq_temp.astype(np.int32).reshape(n_iter*T)
        
        # combine
        out[:,nn] = temp0
    
    return out




