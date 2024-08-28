#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""




import numpy as np


# ----------------------------------------------------------------------------
# initial conditions
# ----------------------------------------------------------------------------

N = 5679309 # population of MA in 2019
prop_MSM = 0.054 * 0.5 * 0.5
prop_bi = 0.054 * 0.5 * 0.5
prop_MSM_high = 0.15 
prop_hetero_high = 0.15
prop_bi_high = 0.15

N_vector = np.array([N * prop_MSM * prop_MSM_high,
                     N * prop_MSM * (1-prop_MSM_high),
                     N * prop_bi * prop_bi_high,
                     N * prop_bi * (1-prop_bi_high),
                     N/2 * (1-prop_bi-prop_MSM) * prop_hetero_high,
                     N/2 * (1-prop_bi-prop_MSM) * (1-prop_hetero_high),
                     N/2 * (1-prop_bi-prop_MSM) * prop_hetero_high,
                     N/2 * (1-prop_bi-prop_MSM) * (1-prop_hetero_high)
                     ]).astype(int).astype(float)

prop_women = (1-prop_bi-prop_MSM)/2

n_groups = len(N_vector)




# ----------------------------------------------------------------------------
# base strain prevalence by group
# ----------------------------------------------------------------------------


# overall prevalence
prev_men = 0.0126
prev_women = 0.0161
prev_msm = 0.03
prev_bi = 0.03
# compute prevalence in msw
prev_msw = (prev_men - prev_msm * prop_MSM/(1-prop_women) - prev_bi * prop_bi/(1-prop_women) ) / (1 - prop_MSM/(1-prop_women) - prop_bi/(1-prop_women))

# separate into risk groups
prev_msm_high = 0.08
prev_msm_low = prev_msm - prev_msm_high * (prop_MSM * prop_MSM_high) * 1/(prop_MSM * (1-prop_MSM_high))
prev_women_high = 0.027
prev_women_low = (prev_women - prev_women_high * prop_hetero_high/prop_women) / (1-prop_hetero_high/prop_women)
prev_bi_high = np.sqrt(prev_msm_high*prev_women_high)
prev_bi_low = (prev_bi - prev_bi_high * prop_bi_high/(1-prop_women)) / (1-prop_bi_high)/(1-prop_women)
prev_msw_high = prev_women_high
prev_msw_low = (prev_msw - prev_msw_high * prop_hetero_high/(1-prop_women)) / (1-prop_hetero_high)/(1-prop_women)


prevalence_susceptible_strain = np.array([prev_msm_high, prev_msm_low,
                                          prev_bi_high, prev_bi_low,
                                          prev_msw_high, prev_msw_low,
                                          prev_women_high, prev_women_low])



# number of active infections by group
Initial_I_nr = (N_vector * prevalence_susceptible_strain).astype(int).astype(float)



# ----------------------------------------------------------------------------
# parameters: ENTRY AND EXIT FLOWS
# ----------------------------------------------------------------------------


# -- group entry and exit rates (per person)
mu_entry = np.array([0.00013]*8)
mu_exit = np.array([0.00013]*8)




# ----------------------------------------------------------------------------
# parameters: RE-TREATMENT
# ----------------------------------------------------------------------------

# rate of retreatment if first treatment failed


duration_first_to_second_treatment_men = 7
duration_first_to_second_treatment_women = 7

p_retreatment_men = (1/duration_first_to_second_treatment_men)
p_retreatment_women =  (1/duration_first_to_second_treatment_women)

p_retreatment = np.array([p_retreatment_men, p_retreatment_men,
                          p_retreatment_men, p_retreatment_men,
                          p_retreatment_men, p_retreatment_men,
                          p_retreatment_women, p_retreatment_women
                          ])

# ----------------------------------------------------------------------------
# parameters: SURVEILLANCE
# ----------------------------------------------------------------------------

# calculate rates:
# -ln(p_followup)/duration 
# men: 0.248/10
# women: 0.478/10

# -- rate of follow-up
# i.e. prob of follow-up * (1/duration until follow-up)
prob_followup_men_h = 0.22
prob_followup_men_l = 0.22
prob_followup_women_h = 0.38
prob_followup_women_l = 0.38

duration_followup_men_h = 10
duration_followup_men_l = 10
duration_followup_women_h = 10
duration_followup_women_l = 10

p_followup_men_h = -np.log(prob_followup_men_h) * (1/duration_followup_men_h)
p_followup_men_l = -np.log(prob_followup_men_l)  * (1/duration_followup_men_l)
p_followup_women_h = -np.log(prob_followup_women_h)  * (1/duration_followup_women_h)
p_followup_women_l = -np.log(prob_followup_women_l)  * (1/duration_followup_women_l)

p_followup = np.array([p_followup_men_h, p_followup_men_l,
                       p_followup_men_h, p_followup_men_l,
                       p_followup_men_h, p_followup_men_l,
                       p_followup_women_h, p_followup_women_l
                       ])



