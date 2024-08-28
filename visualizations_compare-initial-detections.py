


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



in_folder_1 = 'calibration/'
in_folder_2 = 'outputs/sim_results_compare_initial_detections/'
plot_folder = 'figures/compare_initial_detections/'






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


max_initial_detections = 11


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



# -------------------------------------------------------------------------------
# MAKE COMBO DATASETS
# -------------------------------------------------------------------------------


# confidence in elimination
elimdf = pd.DataFrame()

for initial_detections in np.arange(1,max_initial_detections):

    for group_iter, group_name in enumerate(group_names_short):
     
        elimdf_agg = pd.read_csv(in_folder_2+'elimdf_agg_intro-group-'+group_name+'_initial-detections'+ str(initial_detections)+'.csv')

        elimdf_agg['initial_detections'] = initial_detections

        elimdf = pd.concat((elimdf, elimdf_agg))

elimdf.to_csv(in_folder_2+'elimdf_all.csv', index=False)

#elimdf = pd.read_csv(in_folder_2+'elimdf_all.csv')

# disease burden

burdendf = pd.DataFrame()

for initial_detections in np.arange(1,max_initial_detections):
#for initial_detections in [1]:

    for group_iter, group_name in enumerate(group_names_short):

        for abc_par_iter in np.arange(accepted.shape[0]):
            
            temp = pd.read_csv(in_folder_2+'/simulation_data_with_abc_agg_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'_initial-detections'+ str(initial_detections)+'.csv')

            
            for n_zeros in set_of_zeros:
                temptemp = temp[temp['ABC_filtering_'+str(n_zeros)]==1].loc[:,['current_undetected_r','current_cases_r', 'cum_cases_r']]
                temptemp['n_zeros'] = n_zeros
                temptemp['intro_group'] = group_name
                temptemp['abc_par_iter'] = abc_par_iter
                temptemp['initial_detections'] = initial_detections
                burdendf = pd.concat((burdendf,temptemp))

burdendf.to_csv(in_folder_2+'burdendf_all.csv', index=False)

#burdendf = pd.read_csv(in_folder_2+'burdendf_all.csv')


# -------------------------------------------------------------------------------
# COMPARING INITIAL DETECTIONS
# -------------------------------------------------------------------------------


# --- Heatmaps 1 and 2 : Undetected disease burden

# average over groups and parameters
usedata = burdendf.groupby(['initial_detections', 'n_zeros']).agg({'cum_cases_r':'mean',
                                                                    'current_undetected_r':'mean',
                                                                    }).reset_index()



# heatmap 1: cumulative undetected
heatdata = usedata.pivot(index='initial_detections', columns='n_zeros', values='cum_cases_r')

fig, axs = plt.subplots(1,1, figsize=(10,5))

sns.heatmap(heatdata,
            cmap = 'crest',
            ax=axs,
            cbar_kws={'label': 'Cumulative resistant infections'})
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Number of initial detected cases')
axs.invert_yaxis()
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.savefig(plot_folder+'heatmap_initial-detections_vs_n-zeros_cumulative-undetected-r.pdf', dpi=300)
plt.close()



# heatmap 2: current undetected
heatdata = usedata.pivot(index='initial_detections', columns='n_zeros', values='current_undetected_r')

fig, axs = plt.subplots(1,1, figsize=(10,5))

sns.heatmap(heatdata,
            cmap = 'crest',
            ax=axs,
            cbar_kws={'label': 'Undetected infections'})
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Number of initial detected cases')
axs.invert_yaxis()
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.savefig(plot_folder+'heatmap_initial-detections_vs_n-zeros_active-undetected-r.pdf', dpi=300)
plt.close()





# --- Heatmap 3: Confidence in elimination


# average over groups and parameters
usedata = elimdf.groupby(['initial_detections', 'n_zeros']).agg({'zero_incidence_after_detection':'mean'}).reset_index()

heatdata = usedata.pivot(index='initial_detections', columns='n_zeros', values='zero_incidence_after_detection')


fig, axs = plt.subplots(1,1, figsize=(10,5))

sns.heatmap(heatdata,
            cmap = 'crest',
            ax=axs,
            cbar_kws={'label': 'Likelihood of strain elimination'})
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Number of initial detected cases')
axs.invert_yaxis()
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.savefig(plot_folder+'heatmap_initial-detections_vs_n-zeros_confidence-in-elimination.pdf', dpi=300)
plt.close()




# --- lineplots

# individual lineplots
for initial_detections in np.arange(1,max_initial_detections):

    subdf = elimdf[elimdf.initial_detections==initial_detections]

        
    fig, axs = plt.subplots(1,1, figsize=(10,5))
       
    sns.lineplot(x = 'n_zeros', y ='zero_incidence_after_detection',
                 color = '#363A5D',
                 data = subdf, 
                 ax=axs,
                 alpha=0.9,
                 errorbar=(('pi', 95)))

    plt.legend().remove()    
    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xlabel('Days (without additional detection)')
    axs.set_ylabel('Proportion of simulations without new infections')
    plt.savefig(plot_folder+'confidence-in-elimination_all-intro-groups_all-abc-params_simint95_initial-detections-'+ str(initial_detections) +'.pdf', dpi=300)
    plt.close()



# compare multiple detection levels in one lineplot

fig, axs = plt.subplots(1,1, figsize=(10,5))

for i,initial_detections in enumerate([1,5,10]):

    subdf = elimdf[elimdf.initial_detections==initial_detections]

       
    sns.lineplot(x = 'n_zeros', y ='zero_incidence_after_detection',
                 color = colors_groups[i],
                 data = subdf, 
                 label=initial_detections,
                 ax=axs,
                 alpha=0.9,
                 errorbar=(('pi', 95)))

plt.legend(title='Initial detected cases') 
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Proportion of simulations without new infections')
plt.savefig(plot_folder+'confidence-in-elimination_all-intro-groups_all-abc-params_simint95_compare-initial-detections.pdf', dpi=300)
plt.close()



# statistics
# burden
stats_out_avg = burdendf.groupby(['n_zeros', 'initial_detections']).agg({'current_undetected_r':'mean','current_cases_r':'mean', 'cum_cases_r':'mean'}).reset_index()
stats_out_quantiles = burdendf.drop(['intro_group', 'abc_par_iter'], axis=1).groupby(['n_zeros', 'initial_detections']).quantile([0.025, 0.975]).reset_index()
# save
stats_out_avg.to_csv(plot_folder+'burdendf_statistics_avg.csv', index=False)
stats_out_quantiles.to_csv(plot_folder+'burdendf_statistics_percentiles.csv', index=False)
# elim
stats_out_avg = elimdf.groupby(['n_zeros', 'initial_detections']).agg({'zero_incidence_after_detection':'mean'}).reset_index()
stats_out_quantiles = elimdf.loc[:,['n_zeros', 'initial_detections','zero_incidence_after_detection']].groupby(['n_zeros', 'initial_detections']).quantile([0.025, 0.975]).reset_index()
# save
stats_out_avg.to_csv(plot_folder+'elimdf_statistics_avg.csv', index=False)
stats_out_quantiles.to_csv(plot_folder+'elimdf_statistics_percentiles.csv', index=False)


