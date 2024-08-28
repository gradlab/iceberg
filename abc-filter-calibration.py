#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""



import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from parameters_version_2024_05_01 import N_vector


outfolder ='calibration/'
plotfolder = 'calibration/plot_prior_vs_posterior/'

# read  data               
abc_iter = 100 
sample_params_out = pd.read_csv(outfolder+'calibration_results_script0.csv')
for aa in np.arange(1,abc_iter):             
    temp = pd.read_csv(outfolder+'calibration_results_script'+str(aa)+'.csv')
    
    sample_params_out = pd.concat((sample_params_out, temp))

   
# ----------------------------------------------------------------------------
# ABC filtering - get accepted parameters
# ----------------------------------------------------------------------------

gool1 = sample_params_out.sim_incidence_men.between((181*0.75/100000)/365 , (181*1.25/100000)/365)
gool2 = sample_params_out.sim_incidence_women.between((80*0.75/100000)/365 , (80*1.25/100000)/365)

sample_params_out['sim_prevalence_msm_all'] = sample_params_out.sim_prevalence_msm_h * (N_vector[0]/np.sum(N_vector[0:5])) + sample_params_out.sim_prevalence_msm_l * (N_vector[1]/np.sum(N_vector[0:5])) + sample_params_out.sim_prevalence_msmw_h * (N_vector[2]/np.sum(N_vector[0:5])) + sample_params_out.sim_prevalence_msmw_l * (N_vector[3]/np.sum(N_vector[0:4]))  
sample_params_out['sim_prevalence_msw_all'] = sample_params_out.sim_prevalence_msw_h * (N_vector[4]/np.sum(N_vector[4:6])) + sample_params_out.sim_prevalence_msw_l * (N_vector[5]/np.sum(N_vector[4:6]))
sample_params_out['sim_prevalence_wsm_all'] = sample_params_out.sim_prevalence_wsm_h * (N_vector[6]/np.sum(N_vector[6:])) + sample_params_out.sim_prevalence_wsm_l * (N_vector[7]/np.sum(N_vector[6:]))

gool3 = sample_params_out.sim_prevalence_msm_all.between(0.02 , 0.06)
gool4 = sample_params_out.sim_prevalence_msw_all.between(0.003 , 0.02)
gool5 = sample_params_out.sim_prevalence_wsm_all.between(0.003 , 0.02)


abc_filter_range = gool1 * gool2 * gool3 * gool4 * gool5 

sum(abc_filter_range)

accepted = sample_params_out[abc_filter_range]
# save
accepted.to_csv(outfolder+'accepted-parameters.csv', index=False)




# ----------------------------------------------------------------------------
# -- plot prior vs posterior parameters
# ----------------------------------------------------------------------------


for vv in sample_params_out.columns[11:-2]:
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,3), sharex=True)
    sns.kdeplot(data=sample_params_out[abc_filter_range] ,
                 x=vv, color='orange',label='posterior',linewidth=3,
                 ax=axs)
    sns.kdeplot(data=sample_params_out,
                 x=vv, color = 'gray',label='prior', linewidth=3,
                 ax=axs)
    plt.legend(fontsize=8)
    axs.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(plotfolder+'prior-posterior_'+vv+'.pdf', dpi=300)
    plt.close()

# plot outcomes - incidence and prevalence (overall and accepted parameters)
for vv in (list(sample_params_out.columns[1:11]) + ['sim_prevalence_msm_all', 'sim_prevalence_msw_all', 'sim_prevalence_wsm_all']):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,3), sharex=True)
    sns.kdeplot(data=sample_params_out[abc_filter_range] ,
                 x=vv, color='orange',label='accepted parameters',linewidth=3,
                 ax=axs)
    sns.kdeplot(data=sample_params_out,
                 x=vv, color = 'gray',label='all paramters', linewidth=3,
                 ax=axs)
    plt.legend(fontsize=8)
    axs.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(plotfolder+'outcome-accepted-vs-overall_'+vv+'.pdf', dpi=300)
    plt.close()
    
    
    



