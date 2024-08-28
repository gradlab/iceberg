


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




in_folder_1 = 'calibration/'
in_folder_2 = 'outputs/sim_results_varying_resources/'
plot_folder = 'figures/compare_resource_levels/'



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

# ranges considered for antibiotic susceptibility testing
for ast_rate in [0.01, 0.1, 0.2, 0.3, 0.4]:
    # ranges considered for screening
    for screen_multiplier in [0.5, 1., 1.5, 2.]:

	    for group_iter, group_name in enumerate(group_names_short):
	        
	        elimdf_agg = pd.read_csv(in_folder_2+'elimdf_agg_intro-group-'+group_name+'_detection-percent-'+str(int(ast_rate*100))+'_screening-multiplier-'+str(screen_multiplier)+'.csv')

	        elimdf_agg['ast_rate'] = ast_rate
	        elimdf_agg['screen_multiplier'] = screen_multiplier

	        elimdf = pd.concat((elimdf, elimdf_agg))

elimdf.to_csv(in_folder_2+'elimdf_all.csv', index=False)

# disease burden

burdendf = pd.DataFrame()

# ranges considered for antibiotic susceptibility testing
for ast_rate in [0.01, 0.1, 0.2, 0.3, 0.4]:
    # ranges considered for screening
    for screen_multiplier in [0.5, 1., 1.5, 2.]:

	    for group_iter, group_name in enumerate(group_names_short):

	        for abc_par_iter in np.arange(accepted.shape[0]):
	        	temp = pd.read_csv(in_folder_2+'simulation_data_with_abc_agg_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'_detection-percent-'+str(int(ast_rate*100))+'_screening-multiplier-'+str(screen_multiplier)+'.csv')
	        	temp['intro_group'] = group_name
	        	temp['abc_par_iter'] = abc_par_iter
	        	temp['initial_detections'] = initial_detections
	        	temp['ast_rate'] = ast_rate
	        	temp['screen_multiplier'] = screen_multiplier

	        	for n_zeros in set_of_zeros:
	        		temptemp = temp[temp['ABC_filtering_'+str(n_zeros)]==1].loc[:,['current_undetected_r','current_cases_r', 'cum_cases_r']]
	        		temptemp['n_zeros'] = n_zeros
	        		burdendf = pd.concat((burdendf,temptemp))

burdendf.to_csv(in_folder_2+'burdendf_all.csv', index=False)



# -------------------------------------------------------------------------------
# HEATMAPS OF SCREENING AND DETECTION RATES
# -------------------------------------------------------------------------------




# -- heatmaps 1 & 2: disease burden at varying months (1,2,3,4): active and cumulative cases, respectively
for n_zeros in [1, 30, 60, 90, 120, 150, 200]:
	usedata = burdendf[burdendf.n_zeros==n_zeros]
	usedata = usedata.groupby(['screen_multiplier', 'ast_rate']).agg('mean').reset_index()
	usedata1 = usedata.pivot(index='ast_rate', columns='screen_multiplier', values='cum_cases_r')
	# heatmap 1: cumulative undetected
	fig, axs = plt.subplots(1,1, figsize=(10,5))

	sns.heatmap(usedata1,palette = 'flare',ax=axs)
	axs.set_xlabel('Days (without additional detection)')
	axs.set_ylabel('Number of initial detected cases')
	plt.savefig(plot_folder+'heatmap_ast-rate_vs_screen_cumulative-undetected-r_n-zeros-'+str(n_zeros)+'.pdf', dpi=300)
	plt.close()



	# heatmap 2: current undetected
	usedata2 = usedata.pivot(index='ast_rate', columns='screen_multiplier', values='current_undetected_r')

	fig, axs = plt.subplots(1,1, figsize=(10,5))

	sns.heatmap(usedata2,palette = 'flare',ax=axs)
	axs.set_xlabel('Days (without additional detection)')
	axs.set_ylabel('Number of initial detected cases')
	plt.savefig(plot_folder+'heatmap_ast-rate_vs_screen_active-undetected-r_n-zeros-'+str(n_zeros)+'.pdf', dpi=300)
	plt.close()


# -- heatmap 3: confidence at varying months (1,2,3,4)
for n_zeros in [1, 30, 60, 90, 120, 150, 200]:
	usedata = elimdf[elimdf.n_zeros==n_zeros]
	usedata = usedata.groupby(['screen_multiplier', 'ast_rate']).agg('mean').reset_index()
	usedata3 = usedata.pivot(index='ast_rate', columns='screen_multiplier', values='zero_incidence_after_detection')

	fig, axs = plt.subplots(1,1, figsize=(10,5))

	sns.heatmap(usedata3,palette = 'flare',ax=axs)
	axs.set_xlabel('Days (without additional detection)')
	axs.set_ylabel('Number of initial detected cases')
	plt.savefig(plot_folder+'heatmap_ast-rate_vs_screen_confidence-in-elimination_n-zeros-'+str(n_zeros)+'.pdf', dpi=300)
	plt.close()




# -------------------------------------------------------------------------------
# LINEPLOTS WITH VARYING ANTIBIOTIC SUSCEPTIBILITY TESTING RATES
# -------------------------------------------------------------------------------

usedata = burdendf[burdendf.screen_multiplier==1.]

# cumulative cases
fig, axs = plt.subplots(1,1, figsize=(10,4))

sns.lineplot(x='n_zeros', y='cum_cases_r', 
			hue='ast_rate',
			palette=sns.color_palette("hls", 11),
			data=usedata,
			ax=axs, #color='lightgray',
			errorbar=(('pi', 95))
			)
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Cumulative infections (resistant strain)')
plt.savefig(plot_folder+'lineplot-95-simint_cum-cases-r_increasing-zeros_all-intro-groups_all-abc-params_compare-ast-rates.pdf', dpi=300)
plt.close()

# active cases
fig, axs = plt.subplots(1,1, figsize=(10,4))

sns.lineplot(x='n_zeros', y='current_undetected_r', 
			hue='ast_rate',
			palette=sns.color_palette("hls", 11),
			data=usedata, 
			ax=axs, #color='lightgray',
			errorbar=(('pi', 95)),)
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detectio)n')
axs.set_ylabel('Cumulative infections (resistant strain)')
plt.savefig(plot_folder+'lineplot-95-simint_active-cases-r_increasing-zeros_all-intro-groups_all-abc-params_compare-ast-rates.pdf', dpi=300)
plt.close()



# confidence in elimination
usedata = elimdf[elimdf.screen_multiplier==1.]

fig, axs = plt.subplots(1,1, figsize=(10,4))

sns.lineplot(x='n_zeros', y='zero_incidence_after_detection', 
			hue='ast_rate',
			palette=sns.color_palette("hls", 11),
			data=usedata,
			ax=axs, #color='lightgray',
			errorbar=(('pi', 95)))
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Proportion of simulations without new infections')
plt.savefig(plot_folder+'lineplot-95-simint_confidence-in-elimination_increasing-zeros_all-intro-groups_all-abc-params_compare-ast-rates.pdf', dpi=300)
plt.close()




# -------------------------------------------------------------------------------
# LINEPLOTS WITH VARYING SCREENING RATES
# -------------------------------------------------------------------------------


usedata = burdendf[burdendf.ast_rate==0.1]

# -- cumulative cases
fig, axs = plt.subplots(1,1, figsize=(10,4))

sns.lineplot(x='n_zeros', y='cum_cases_r', 
			hue='screen_multiplier',
			palette=sns.color_palette("hls", 11),
			data=usedata,
			ax=axs, 
			errorbar=(('pi', 95)))
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Cumulative infections (resistant strain)')
plt.savefig(plot_folder+'lineplot-95-simint_cum-cases-r_increasing-zeros_all-intro-groups_all-abc-params_compare-screen-multipliers.pdf', dpi=300)
plt.close()



# -- active cases
fig, axs = plt.subplots(1,1, figsize=(10,4))

sns.lineplot(x='n_zeros', y='current_undetected_r', 
			hue='screen_multiplier',
			palette=sns.color_palette("hls", 11),
			data=usedata,
			ax=axs, errorbar=(('pi', 95)))
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Cumulative infections (resistant strain)')
plt.savefig(plot_folder+'lineplot-95-simint_active-undetected-r_increasing-zeros_all-intro-groups_all-abc-params_compare-screen-multipliers.pdf', dpi=300)
plt.close()



# -- confidence in elimination
usedata = elimdf[elimdf.ast_rate==0.1]

sns.lineplot(x='n_zeros', y='zero_incidence_after_detection', 
			hue='screen_multiplier',
			palette=sns.color_palette("hls", 11),
			data=usedata, ax=axs, errorbar=(('pi', 95)))
axs.spines[['right', 'top']].set_visible(False)
axs.set_xlabel('Days (without additional detection)')
axs.set_ylabel('Proportion of simulations without new infections')
plt.savefig(plot_folder+'lineplot-95-simint_confidence-in-elimination_increasing-zeros_all-intro-groups_all-abc-params_compare-screen-multipliers.pdf', dpi=300)
plt.close()






