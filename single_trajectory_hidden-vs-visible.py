#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

in_folder = 'outputs/sim_results_post_abc/'
plot_folder = 'figures/main_analysis/single_trajectory_plots/'


colors_groups = ['#363A5D', # dark space blue
          '#6870AE', # light space blue
          '#BB4430', # dark brick red
          '#D56F5D', # light brick red
          '#276864', # dark teal
          '#66B2B2', # light teal
          '#F39D2D', # yellow orange (dark yellow)
          '#F8C630', # saffron (yellow)
         ]



n_iter = 500


# --- read data



group_name = 'MSM_h'
abc_par_iter = 0
initial_detections = 1

data = pd.read_csv(in_folder+'simulation_data_with_abc_agg_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'_initial-detections'+ str(initial_detections)+'.csv')

data.cum_cases_r = data.cum_cases_r + 1

# -- plot a single trajectory 

# subset time
sub = data[data.time<2*365]


for randiter in np.arange(n_iter):
    sub2 = sub[sub.iteration==randiter]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9,5),
                            gridspec_kw={'height_ratios': [1, 1]},
                            sharex=False)
    # shaded background
    if sub[sub.cum_detected_r==2].shape[0]>0:
        signal2_start = sub2[sub2.cum_detected_r==2].time.min()
        signal2_end = sub2[sub2.cum_detected_r==2].time.max()
        ymaxval1 = sub2.current_undetected_r.max()
        ymaxval2 = sub2.cum_cases_r.max()
        axs[0].axvspan(xmin = signal2_start, xmax =signal2_end, 
                        ymin = 0, ymax=ymaxval1,
                        facecolor='lightgray', alpha=0.5)
        axs[1].axvspan(xmin = signal2_start, xmax =signal2_end, 
                        ymin = 0, ymax=ymaxval2,
                        facecolor='lightgray', alpha=0.5)

    # top plot: active cases
    sns.lineplot(data = sub2,
                 x='time', y='current_undetected_r',
                 linewidth=3, color='#6870AE',
                 ax=axs[0])
    # bottom plot: cumulative detected vs undetected
    sns.lineplot(data = sub2,
                 x='time', y='cum_detected_r',
                 ax=axs[1], color='#BB4430',
                 label='detected cases',
                 linewidth=3)
    sns.lineplot(data = sub2,
                 x='time', y='cum_cases_r', 
                 label='all cases',
                 linewidth=3, color='#6870AE',
                 ax=axs[1])

    axs[0].spines[['right', 'top']].set_visible(False)
    axs[1].spines[['right', 'top']].set_visible(False)
    axs[0].set_xlabel('Time (in days)')
    axs[1].set_xlabel('Time (in days)')
    axs[0].set_ylabel('Active infections')
    axs[1].set_ylabel('Cumulative infections')
    axs[0].set_title('Undetected infections')
    axs[1].set_title('Cumulative detected cases relative to all infections')
    plt.tight_layout()
    plt.savefig(plot_folder + 'single_trajectory_MSM_h_abc-params-0_cumulative-detected-vs-total_plus-active-undetected_iteration-'+str(randiter)+'.pdf', dpi=300)
    plt.close()



# single year
sub = data[data.time<1*365]


for randiter in np.arange(n_iter):
    sub2 = sub[sub.iteration==randiter]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9,5),
                            gridspec_kw={'height_ratios': [1, 1]},
                            sharex=False)
    # shaded background
    if sub[sub.cum_detected_r==2].shape[0]>0:
        signal2_start = sub2[sub2.cum_detected_r==2].time.min()
        signal2_end = sub2[sub2.cum_detected_r==2].time.max()
        ymaxval1 = sub2.current_undetected_r.max()
        ymaxval2 = sub2.cum_cases_r.max()
        axs[0].axvspan(xmin = signal2_start, xmax =signal2_end, 
                        ymin = 0, ymax=ymaxval1,
                        facecolor='lightgray', alpha=0.5)
        axs[1].axvspan(xmin = signal2_start, xmax =signal2_end, 
                        ymin = 0, ymax=ymaxval2,
                        facecolor='lightgray', alpha=0.5)

    # top plot: active cases
    sns.lineplot(data = sub2,
                 x='time', y='current_undetected_r',
                 linewidth=3, color='#6870AE',
                 ax=axs[0])
    # bottom plot: cumulative detected vs undetected
    sns.lineplot(data = sub2,
                 x='time', y='cum_detected_r',
                 ax=axs[1], color='#BB4430',
                 label='detected cases',
                 linewidth=3)
    sns.lineplot(data = sub2,
                 x='time', y='cum_cases_r', 
                 label='all infections',
                 linewidth=3, color='#6870AE',
                 ax=axs[1])

    axs[0].spines[['right', 'top']].set_visible(False)
    axs[1].spines[['right', 'top']].set_visible(False)
    axs[0].set_xlabel('Time (in days)')
    axs[1].set_xlabel('Time (in days)')
    axs[0].set_ylabel('Active infections')
    axs[1].set_ylabel('Cumulative infections')
    axs[0].set_title('Undetected infections')
    axs[1].set_title('Cumulative detected cases relative to all infections')
    plt.tight_layout()
    plt.savefig(plot_folder + 'single_trajectory_MSM_h_abc-params-0_cumulative-detected-vs-total_plus-active-undetected_1YEAR_iteration-'+str(randiter)+'.pdf', dpi=300)
    plt.close()


