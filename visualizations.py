
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



in_folder_1 = 'calibration/'
in_folder_2 = 'outputs/sim_results_post_abc/'
plot_folder = 'figures/main_analysis/'






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


max_initial_detections = 3


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
# MAIN ANALYSIS: LINEPLOTS - x DETECTIONS WITH INCREASING TIME WITHOUT DETECTION
# -------------------------------------------------------------------------------


"""
# --- Lineplots 1 & 2: Undetected disease burden (cumulative and active infections, respectively)

#for initial_detections in np.arange(1,50):
for initial_detections in np.arange(1,max_initial_detections):
      
    subdf = burdendf[burdendf.initial_detections==initial_detections]
    # --- Lineplot 1: Cumulative infections

    fig, axs = plt.subplots(1,1, figsize=(10,4))

    sns.lineplot(x='n_zeros', y='cum_cases_r', 
                data=subdf,
                ax=axs, #color='lightgray',
                errorbar=(('pi', 95)),
                color='#6870AE',
               )
    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xlabel('Days without additional detection')
    axs.set_ylabel('Cumulative cases (resistant strain)')
    plt.savefig(plot_folder+'lineplot-95-simint_cum-cases-r_increasing-zeros_all-intro-groups_all-abc-params_initial-detections-'+ str(initial_detections) +'.pdf', dpi=300)
    plt.close()


    # --- Lineplot 2: Active infections

    fig, axs = plt.subplots(1,1, figsize=(10,4))

    sns.lineplot(x='n_zeros', y='current_undetected_r', 
                data=subdf,
                ax=axs, #color='lightgray',
                errorbar=(('pi', 95)),
                color='#66B2B2'
               )
    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xlabel('Days without new detected cases')
    axs.set_ylabel('Undetected cases (resistant strain)')
    plt.savefig(plot_folder+'lineplot-95-simint_active-undetected-r_increasing-zeros_all-intro-groups_all-abc-params_initial-detections-'+ str(initial_detections) +'.pdf', dpi=300)
    plt.close()


    # -- corresponding statistics
    stats_out_avg = subdf.groupby(['n_zeros']).agg({'current_undetected_r':'mean','current_cases_r':'mean', 'cum_cases_r':'mean'}).reset_index()
    stats_out_quantiles = subdf.loc[:,['n_zeros','current_undetected_r', 'current_cases_r', 'cum_cases_r']].groupby(['n_zeros']).quantile([0.025, 0.975]).reset_index()
    # save
    stats_out_avg.to_csv(plot_folder+'burdendf_statistics_avg_'+str(initial_detections)+'.csv', index=False)
    stats_out_quantiles.to_csv(plot_folder+'burdendf_statistics_percentiles_'+str(initial_detections)+'.csv', index=False)

# --- Lineplot 3: Confidence in elimination

#for initial_detections in np.arange(1,50):
for initial_detections in np.arange(1,max_initial_detections):

    subdf = elimdf[elimdf.initial_detections==initial_detections]


        
    fig, axs = plt.subplots(1,1, figsize=(10,5))
       
    sns.lineplot(x = 'n_zeros', y ='zero_incidence_after_detection',
                 color = '#363A5D',
                 data = subdf, 
                 #label=group_name,
                 ax=axs,
                 alpha=0.9,
                 errorbar=(('pi', 95)))

    plt.legend().remove()    
    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xlabel('Days (without new detected cases)')
    axs.set_ylabel('Proportion of simulations without new cases')
    plt.savefig(plot_folder+'confidence-in-elimination_all-intro-groups_all-abc-params_simint95_initial-detections-'+ str(initial_detections) +'.pdf', dpi=300)
    plt.close()

    # -- corresponding statistics
    stats_out_avg = subdf.groupby(['n_zeros']).agg({'zero_incidence_after_detection':'mean'}).reset_index()
    stats_out_quantiles = subdf.loc[:,['n_zeros', 'zero_incidence_after_detection']].groupby(['n_zeros']).quantile([0.025, 0.975]).reset_index()
    # save
    stats_out_avg.to_csv(plot_folder+'elimdf_statistics_avg_'+str(initial_detections)+'.csv', index=False)
    stats_out_quantiles.to_csv(plot_folder+'elimdf_statistics_percentiles_'+str(initial_detections)+'.csv', index=False)
"""

# --- ANNOTATED Lineplots 1 & 2: Undetected disease burden (cumulative and active infections, respectively)

for initial_detections in np.arange(1,max_initial_detections):
      
    subdf = burdendf[burdendf.initial_detections==initial_detections]


    # get points for annotations
    line0_mean = subdf[subdf.n_zeros==0].cum_cases_r.mean()
    line60_mean = subdf[subdf.n_zeros==60].cum_cases_r.mean()
    line180_mean = subdf[subdf.n_zeros==180].cum_cases_r.mean()
    line0_top = subdf[subdf.n_zeros==0].cum_cases_r.quantile(0.975)
    line60_top = subdf[subdf.n_zeros==60].cum_cases_r.quantile(0.975)
    line180_top = subdf[subdf.n_zeros==180].cum_cases_r.quantile(0.975)
    line0_bottom = subdf[subdf.n_zeros==0].cum_cases_r.quantile(0.025)
    line60_bottom = subdf[subdf.n_zeros==60].cum_cases_r.quantile(0.025)
    line180_bottom = subdf[subdf.n_zeros==180].cum_cases_r.quantile(0.025)
    

    # --- Lineplot 1: Cumulative infections

    fig, axs = plt.subplots(1,1, figsize=(10,4))
    # annotations
    # at t=180
    axs.vlines(x=180, ymin=line180_bottom, ymax=line180_top, color='#9093B1', linestyle='--')
    # at t=0
    axs.vlines(x=0, ymin=line0_bottom, ymax=line0_top, color='#9093B1', linestyle='--')
    # at t=60
    axs.vlines(x=60, ymin=line60_bottom, ymax=line60_top, color='#9093B1', linestyle='--')
    
    # lineplot
    sns.lineplot(x='n_zeros', y='cum_cases_r', 
                data=subdf,
                ax=axs, #color='lightgray',
                errorbar=(('pi', 95)),
                color='#6870AE',
               )
    axs.spines[['right', 'top']].set_visible(False)
    # annotation
    sns.scatterplot(x=[180], y=[line180_mean], color='#6870AE', ax=axs)
    sns.scatterplot(x=[0], y=[line0_mean], color='#6870AE', ax=axs)
    sns.scatterplot(x=[60], y=[line60_mean], color='#6870AE', ax=axs)
    # text
    plt.text(x=181, y=line180_top+0.3, s=str(line180_top), color='#9093B1')
    plt.text(x=181, y=0.3, s='t=180', color='#9093B1')
    plt.text(x=181, y=line180_mean+0.3, s=str(round(line180_mean,1)), color='#9093B1')
    plt.text(x=1, y=line0_top+0.3, s=str(line0_top), color='#9093B1')
    plt.text(x=1, y=0.3, s='t=0', color='#9093B1')
    plt.text(x=1, y=line0_mean+0.3, s=str(round(line0_mean, 1)), color='#9093B1')
    plt.text(x=61, y=line60_top+0.3, s=str(line60_top), color='#9093B1')
    plt.text(x=61, y=0.3, s='t=60', color='#9093B1')
    plt.text(x=61, y=line60_mean+0.3, s=str(round(line60_mean, 1)), color='#9093B1')
    
    axs.set_xlabel('Days (without additional detection)')
    axs.set_ylabel('Cumulative infections (resistant strain)')
    plt.tight_layout()
    plt.savefig(plot_folder+'lineplot-95-simint_cum-cases-r_increasing-zeros_all-intro-groups_all-abc-params_initial-detections-'+ str(initial_detections) +'_annotated.pdf', dpi=300)
    plt.close()


for initial_detections in np.arange(1,max_initial_detections):

    subdf = elimdf[elimdf.initial_detections==initial_detections]

    # get points for annotations
    line0_mean = subdf[subdf.n_zeros==0].zero_incidence_after_detection.mean()
    line60_mean = subdf[subdf.n_zeros==60].zero_incidence_after_detection.mean()
    line180_mean = subdf[subdf.n_zeros==180].zero_incidence_after_detection.mean()
    line0_top = subdf[subdf.n_zeros==0].zero_incidence_after_detection.quantile(0.975)
    line60_top = subdf[subdf.n_zeros==60].zero_incidence_after_detection.quantile(0.975)
    line180_top = subdf[subdf.n_zeros==180].zero_incidence_after_detection.quantile(0.975)
    line0_bottom = subdf[subdf.n_zeros==0].zero_incidence_after_detection.quantile(0.025)
    line60_bottom = subdf[subdf.n_zeros==60].zero_incidence_after_detection.quantile(0.025)
    line180_bottom = subdf[subdf.n_zeros==180].zero_incidence_after_detection.quantile(0.025)

        
    fig, axs = plt.subplots(1,1, figsize=(10,5))
    # annotations
    # at t=180
    axs.vlines(x=180, ymin=line180_bottom, ymax=line180_top, color='#9093B1', linestyle='--')
    # at t=0
    axs.vlines(x=0, ymin=line0_bottom, ymax=line0_top, color='#9093B1', linestyle='--')
    # at t=60
    axs.vlines(x=60, ymin=line60_bottom, ymax=line60_top, color='#9093B1', linestyle='--')
       
    sns.lineplot(x = 'n_zeros', y ='zero_incidence_after_detection',
                 color = '#363A5D',
                 data = subdf, 
                 #label=group_name,
                 ax=axs,
                 alpha=0.9,
                 errorbar=(('pi', 95)))

    plt.legend().remove()    
    axs.spines[['right', 'top']].set_visible(False)
    # annotation
    sns.scatterplot(x=[180], y=[line180_mean], color='#6870AE', ax=axs)
    sns.scatterplot(x=[0], y=[line0_mean], color='#6870AE', ax=axs)
    sns.scatterplot(x=[60], y=[line60_mean], color='#6870AE', ax=axs)
    # text
    plt.text(x=181, y=line180_top+0.01, s=str(round(line180_top,2)), color='#9093B1')
    plt.text(x=181, y=line180_bottom+0.01, s=str(round(line180_bottom,2)), color='#9093B1')
    plt.text(x=181, y=line180_mean-0.03, s=str(round(line180_mean,2)), color='#9093B1')
    plt.text(x=181, y=line180_bottom-0.05, s='t=180', color='#9093B1')
    
    plt.text(x=1, y=line0_top+0.01, s=str(round(line0_top,2)), color='#9093B1')
    plt.text(x=1, y=line0_bottom+0.01, s=str(round(line0_bottom,2)), color='#9093B1')
    plt.text(x=1, y=line0_bottom-0.03, s='t=0', color='#9093B1')
    plt.text(x=1, y=line0_mean-0.03, s=str(round(line0_mean, 2)), color='#9093B1')
    
    plt.text(x=61, y=line60_top+0.01, s=str(round(line60_top,2)), color='#9093B1')
    plt.text(x=61, y=line60_bottom+0.01, s=str(round(line60_bottom,2)), color='#9093B1')
    plt.text(x=61, y=line60_bottom-0.05, s='t=60', color='#9093B1')
    plt.text(x=61, y=line60_mean-0.03, s=str(round(line60_mean, 2)), color='#9093B1')
    
    axs.set_xlabel('Days (without new detected cases)')
    axs.set_ylabel('Proportion of simulations without new infections')
    plt.tight_layout()
    plt.savefig(plot_folder+'confidence-in-elimination_all-intro-groups_all-abc-params_simint95_initial-detections-'+ str(initial_detections) +'_annotated.pdf', dpi=300)
    plt.close()

 























