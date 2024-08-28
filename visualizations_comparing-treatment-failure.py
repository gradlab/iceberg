


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



in_folder_1 = 'calibration/'
in_folder_2 = 'outputs/sim_results_varying_treatment_failure/'
plot_folder = 'figures/compare_treatment_failure/'



# -----------------------------------------------------------------------------
# prep
# -----------------------------------------------------------------------------



group_names = ['MSM, high risk', 'MSM, low risk',
               'MSMW, high risk', 'MSMW, low risk',
               'MSW, high risk', 'MSW, low risk', 
               'WSM, high risk', 'WSM, low risk']

group_names_short = ['MSM_h', 'MSM_l',
               'MSMW_h', 'MSMW_l',
               'MSW_h', 'MSW_l', 
               'WSM_h', 'WSM_l']
  



# ABC filtering - maximum days without detection
#max_zeros = 121
# consider specific set of zeros
set_of_zeros = [0,1] + list(np.arange(5,250,5))



n_iter = 500
T = 365*5
n_groups = 8

# accepted parameters
accepted = pd.read_csv(in_folder_1+'accepted-parameters.csv',)

# group colors for visualizations
colors_groups = ['#363A5D', # dark space blue
          '#6870AE', # light space blue
          '#BB4430', # dark brick red
          '#D56F5D', # light brick red
          '#276864', # dark teal
          '#66B2B2', # light teal
          '#F39D2D', # yellow orange (dark yellow)
          '#F8C630', # saffron (yellow)
         ]


initial_detections = 2

# -------------------------------------------------------------------------------
# MAKE COMBO DATASETS
# -------------------------------------------------------------------------------


# confidence in elimination
elimdf = pd.DataFrame()

for treat_fail in [0.3, 0.5, 0.7, 0.9]:
    
    for group_iter, group_name in enumerate(group_names_short):
        
        elimdf_agg = pd.read_csv(in_folder_2+'elimdf_agg_intro-group-'+group_name+'_treat-fail-'+ str(int(treat_fail*10))+'.csv')

        elimdf_agg['treatment_failure'] = treat_fail

        elimdf = pd.concat((elimdf, elimdf_agg))

elimdf.to_csv(in_folder_2+'elimdf_all.csv', index=False)

#elimdf = pd.read_csv(in_folder_2+'elimdf_all.csv')

# disease burden

burdendf = pd.DataFrame()

for treat_fail in [0.3, 0.5, 0.7, 0.9]:

    for group_iter, group_name in enumerate(group_names_short):

        for abc_par_iter in np.arange(accepted.shape[0]):
            
            temp = pd.read_csv(in_folder_2+'simulation_data_with_abc_agg_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'_treat-fail-'+ str(int(treat_fail*10))+'.csv')

            
            
            for n_zeros in set_of_zeros:
                temptemp = temp[temp['ABC_filtering_'+str(n_zeros)]==1].loc[:,['current_undetected_r','current_cases_r', 'cum_cases_r']]
                
                temptemp['n_zeros'] = n_zeros
                temptemp['intro_group'] = group_name
                temptemp['abc_par_iter'] = abc_par_iter
                temptemp['initial_detections'] = initial_detections
                temptemp['treatment_failure'] = treat_fail
                
                burdendf = pd.concat((burdendf,temptemp))

burdendf.to_csv(in_folder_2+'burdendf_all.csv', index=False)

#burdendf = pd.read_csv(in_folder_2+'burdendf_all.csv')





# -------------------------------------------------------------------------------
# LINEPLOTS WITH VARYING TREATMENT FAILURE LEVELS
# -------------------------------------------------------------------------------




# -- lineplots 1 & 2: disease burden 

# cumulative undetected - 95 simint
fig, axs = plt.subplots(1,1, figsize=(10,4))

sns.lineplot(x='n_zeros', y='cum_cases_r', 
            hue='treatment_failure',
            palette = colors_groups,
            data=burdendf,
            ax=axs, 
            errorbar=(('pi', 95)),
            color='#6870AE',
           )
#axs.set_ylim((0,18))
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Cumulative infections (resistant strain)')
plt.savefig(plot_folder+'lineplot-95-simint_cum-cases-r_increasing-zeros_all-intro-groups_all-abc-params_compare-treat-fail.pdf', dpi=300)
plt.close()



# active infections 
fig, axs = plt.subplots(1,1, figsize=(10,4))

sns.lineplot(x='n_zeros', y='current_undetected_r', 
            hue='treatment_failure',
            palette = colors_groups,
            data=burdendf,
            ax=axs,
            errorbar=(('pi', 95)),
            color='#66B2B2'
           )
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Undetected infections (resistant strain)')
plt.savefig(plot_folder+'lineplot-95-simint_active-undetected-r_increasing-zeros_all-intro-groups_all-abc-params_compare-treat-fail-.pdf', dpi=300)
plt.close()


# -- lineplot 3: confidence in elimination

fig, axs = plt.subplots(1,1, figsize=(10,5))
   
sns.lineplot(x = 'n_zeros', y ='zero_incidence_after_detection',
             hue = 'treatment_failure',
             palette = colors_groups,
             data = elimdf, 
             ax=axs,
             alpha=0.9,
             errorbar=(('pi', 95)))
plt.legend(title='Treatment failure rate', loc='lower right')
#axs.set_ylim((0,1.5))
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Proportion of simulations without new undetected infections')
plt.savefig(plot_folder+'confidence-in-elimination_all-intro-groups_all-abc-params_simint95_compare-treat-fail-.pdf', dpi=300)
plt.close()


# statistics
# burden
stats_out_avg = burdendf.groupby(['n_zeros', 'treatment_failure']).agg({'current_undetected_r':'mean','current_cases_r':'mean', 'cum_cases_r':'mean'}).reset_index()
stats_out_quantiles = burdendf.drop(['intro_group', 'abc_par_iter'], axis=1).groupby(['n_zeros', 'treatment_failure']).quantile([0.025, 0.975]).reset_index()
# save
stats_out_avg.to_csv(plot_folder+'burdendf_statistics_avg.csv', index=False)
stats_out_quantiles.to_csv(plot_folder+'burdendf_statistics_percentiles.csv', index=False)
# elim
stats_out_avg = elimdf.groupby(['n_zeros', 'treatment_failure']).agg({'zero_incidence_after_detection':'mean'}).reset_index()
stats_out_quantiles = elimdf.loc[:,['n_zeros', 'treatment_failure','zero_incidence_after_detection']].groupby(['n_zeros', 'treatment_failure']).quantile([0.025, 0.975]).reset_index()
# save
stats_out_avg.to_csv(plot_folder+'elimdf_statistics_avg.csv', index=False)
stats_out_quantiles.to_csv(plot_folder+'elimdf_statistics_percentiles.csv', index=False)






