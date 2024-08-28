
import numpy as np
import pandas as pd
import os.path



from functions import loop_over_SOME_zeros_abc_sequence_timemask



out_folder = 'outputs/sim_results_post_abc/'
in_folder = 'calibration/'


# -----------------------------------------------------------------------------
# ABC filter trajectories with 1, 2, ... detections
# -----------------------------------------------------------------------------

# ABC filtering for multiple initial detection values
max_initial_detections = 3

# ABC filtering - maximum days without detection
#max_zeros = 121
# consider specific set of zeros
set_of_zeros = [0,1] + list(np.arange(5,250,5))


T = 365*5
n_groups = 8

accepted = pd.read_csv(in_folder+'accepted-parameters.csv')


group_names = ['MSM, high risk', 'MSM, low risk',
               'MSMW, high risk', 'MSMW, low risk',
               'MSW, high risk', 'MSW, low risk', 
               'WSM, high risk', 'WSM, low risk']

group_names_short = ['MSM_h', 'MSM_l',
               'MSMW_h', 'MSMW_l',
               'MSW_h', 'MSW_l', 
               'WSM_h', 'WSM_l']

n_iter = 500


for abc_par_iter in np.arange(accepted.shape[0]):
    for group_iter, group_name in enumerate(group_names_short):
        # check if any of the abc parameter - group iter combo exists
        fname = out_folder+'simulation_data_with_abc_agg_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'_initial-detections'+ str(1)+'.csv'    
        if os.path.isfile(fname)==False:
            # read data
            combodf = pd.read_csv(out_folder+'simulation_data_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'.csv')

            # reshape
            cum_detected_r_stack = combodf.cum_detected_r.values.reshape((n_iter, T, n_groups))

            # abc filter
            iterables=[np.arange(n_iter),np.arange(T)]
            index = pd.MultiIndex.from_product(iterables, names=['iteration','time'])
            

            # loop over initial detected cases
            for initial_detections in np.arange(1,max_initial_detections):

                # check if the specific initial detections file exists
                fname = out_folder+'simulation_data_with_abc_agg_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'_initial-detections'+ str(initial_detections)+'.csv'    
                if os.path.isfile(fname)==False:
                
                    # loop over zeros
                    temp_combo = loop_over_SOME_zeros_abc_sequence_timemask(cum_detected_r_stack, set_of_zeros, n_iter, T, initial_detections)
                    
                    combodf_abc = pd.DataFrame(temp_combo, 
                                               columns = ['ABC_filtering_'+str(x) for x in set_of_zeros],
                                               index=index).reset_index()
                        
                    # combine with simulation data
                    combodf_all = combodf.merge(combodf_abc, on=['iteration', 'time'], 
                                            how='left')
                    # aggregated across groups
                    combodf_agg = combodf.groupby(['iteration', 'time']).agg('sum').reset_index()
                    combodf_all_agg = combodf_agg.merge(combodf_abc, on=['iteration','time'], how='left')
                    
                    combodf_agg['initial_detections'] = initial_detections
                    combodf_all_agg['initial_detections'] = initial_detections

                    # save
                    # combodf_all.to_csv(out_folder+'simulation_data_with_abc_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'_initial-detections'+ str(initial_detections)+'.csv', index=False)
                    combodf_all_agg.to_csv(out_folder+'simulation_data_with_abc_agg_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'_initial-detections'+ str(initial_detections)+'.csv', index=False)
                    
                    # note: combodf_all_agg = number of cases summed over groups
                    #       combodf_all = number of cases by group



# -----------------------------------------------------------------------------
# compute confidence in elimination
# -----------------------------------------------------------------------------


for initial_detections in np.arange(1,max_initial_detections):

    # separate viz for each group & paramter combo
    for group_iter, group_name in enumerate(group_names_short):
        
        # check if any of the abc parameter - group iter combo exists
        fname = out_folder+'elimdf_agg_intro-group-'+group_name+'_initial-detections'+ str(initial_detections)+'.csv'
        if os.path.isfile(fname)==False:
            
            # first make elimdf for all parameter combos
            elimdf_all = pd.DataFrame()
            # loop over accepted parameter runs
            for abc_par_iter in np.arange(accepted.shape[0]):
                
                combodf_all_agg = pd.read_csv(out_folder+'simulation_data_with_abc_agg_index-group-'+group_name+'_abc-params-'+str(abc_par_iter)+'_initial-detections'+ str(initial_detections)+'.csv')

                combodf_all_agg['intro_group'] = group_name
                combodf_all_agg['abc_par_iter'] = abc_par_iter
                combodf_all_agg['initial_detections'] = initial_detections
                
                # add relative time
                
                for n_zeros in set_of_zeros:
                    combodf_all_agg['first_detection_time_'+str(n_zeros)] = 0
                    combodf_all_agg['ABC_iteration_filter_'+str(n_zeros)] = 0
                    for i in np.arange(n_iter):
                        if combodf_all_agg[(combodf_all_agg.iteration==i)]['ABC_filtering_'+str(n_zeros)].sum()>0:
                            combodf_all_agg.loc[(combodf_all_agg.iteration==i), 'ABC_iteration_filter_'+str(n_zeros)] = 1
                            dettime = combodf_all_agg[(combodf_all_agg.iteration==i)&(combodf_all_agg['ABC_filtering_'+str(n_zeros)]==1)].time.min()
                            combodf_all_agg.loc[(combodf_all_agg.iteration==i), 'first_detection_time_'+str(n_zeros)] = dettime
                            del dettime
                
                    
                # proportion of cases that are detected
                combodf_all_agg['proportion_detected_r'] = combodf_all_agg.cum_detected_r / (combodf_all_agg.cum_cases_r+1)
                combodf_all_agg['proportion_detected_nr'] = combodf_all_agg.cum_detected_nr / (combodf_all_agg.cum_cases_nr)
                
                
                # calculate confidence in elimination
                elimdf = pd.DataFrame()
                
                for ii in np.arange(combodf_all_agg.iteration.max()):
                    
                    sub = combodf_all_agg[(combodf_all_agg.iteration==ii)]
                    sub.reset_index(inplace=True,drop=True)
                    
                    for n_zeros in set_of_zeros:
                        
                        # if detected
                        if sub['ABC_iteration_filter_'+str(n_zeros)].max()>0:
                        
                            # difference between cumulative cases at detection time and end of simulation
                            temp1 = sub[sub['ABC_filtering_'+str(n_zeros)]==1]
                            diff1 = temp1.cum_cases_r - sub.cum_cases_r[sub.shape[0]-1]
            
            
                            # append:
                            tempdf = pd.DataFrame({'intro_group':group_name,
                                                   'abc_par_iter':abc_par_iter,
                                                   'iteration':ii,
                                                   'n_zeros':n_zeros,
                                                   'zero_incidence_after_detection':(diff1==0).astype(int),
                                                  })
                            elimdf = pd.concat((elimdf, tempdf))

                elimdf_all = pd.concat((elimdf_all,elimdf))
            
            # save 
            #elimdf_all.to_csv(out_folder+'elimdf_all_intro-group-'+group_name++'_initial-detections'+ str(initial_detections)+'.csv', index=False)
            
            # aggregate over iterations
            elimdf_agg = elimdf_all.groupby(['intro_group','abc_par_iter', 'n_zeros']).agg('mean').reset_index()
            # save
            elimdf_agg.to_csv(out_folder+'elimdf_agg_intro-group-'+group_name+'_initial-detections'+ str(initial_detections)+'.csv', index=False)
           




     
  