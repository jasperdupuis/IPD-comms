# -*- coding: utf-8 -*-
"""

Methods to analyze and visualize results for Q Learner and DQN in common,
in particular expected rewards v true rewards.

Works for conviction, trust, or base Q Learners.

Does not do anything for the Communicating Player results.

"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

directory = r'C:\Users\Jasper\Desktop\Machine Learning\6505\output\csvs\Base Q Learner'
result_csv = directory + r'\QL_N_1999.csv'

r_mean_str = 'reward (mean)'
r_std_str = 'reward (std)' 
q_mean_str = 'prediction (mean)' 
q_std_str = 'prediction (std)'

def generate_file_list(p_directory):
    all_files = os.listdir(p_directory)
    
    list_files = []
    for file in all_files:
        if not('.csv' in file):
            continue
        list_files.append(os.path.join(p_directory,file))
    return list_files

def generate_summary_results_q_v_r(list_full_paths):
    df = pd.DataFrame()    
    base_name = list_full_paths[0].split('_')[1]
    for file in list_full_paths:
        opponent_name = file.split('_')[2]
        data = np.genfromtxt(file, delimiter=',')
        rewards = data[::2,1:]
        predictions = data[1::2,1:]
        r_mean,r_std = np.mean(rewards,axis=0),np.std(rewards,axis=0)
        q_mean,q_std = np.mean(predictions,axis=0),np.std(predictions,axis=0)
        df[opponent_name + ' reward (mean)'] = r_mean
        df[opponent_name + ' reward (std)'] = r_std
        df[opponent_name + ' prediction (mean)'] = q_mean
        df[opponent_name + ' prediction (std)'] = q_std
    #save_name = base_name + '_N_'+str(len(r_mean)) + '.csv'
    save_name = 'QL_N_'+str(len(r_mean)) + '.csv'
    df.to_csv(os.path.join(directory,save_name))
    return

def plot_single_dimension(parent_fig,parent_ax,linestyle,p_df,p_query):
    #all list entries with passed p_query string in it
    strings =  list(filter(lambda x:p_query in x,p_df.columns)) 
    for r in strings:
        opponent = ''
        for s in r.split(' ')[:-2]:
            opponent = opponent + s + ' '
        curr_data = p_df[r].values
        parent_ax.plot(curr_data,label = opponent[:-1],linestyle = linestyle)    
    return parent_fig,parent_ax

#dont need this every time.

files = generate_file_list(directory)
generate_summary_results_q_v_r(files)

"""
df = pd.read_csv(result_csv)


fig, ax = plt.subplots(figsize=(10,7))
fig,ax = plot_single_dimension(fig,ax,'solid',df,r_mean_str)
fig,ax = plot_single_dimension(fig,ax,'dotted',df,q_mean_str)
fig.legend()
fig.show()
                  


fig, ax = plt.subplots(figsize=(10,7))
fig,ax = plot_single_dimension(fig,ax,'solid',df,r_mean_str)
fig.legend(loc=1)
fig.suptitle('True reward mean')
fig.show()
"""            

#if __name__ == '__main__':

