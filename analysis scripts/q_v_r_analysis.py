# -*- coding: utf-8 -*-
"""

Methods to analyze and visualize results for Q Learner and DQN in common,
in particular expected rewards v true rewards.

"""
import numpy as np
import pandas as pd
import os

directory = r'C:\Users\Jasper\Desktop\Machine Learning\6505\output\csvs\serial games'

result_csv = r'C:/Users/Jasper/Desktop/Machine Learning/6505/output/csvs/serial games/QL_N_999.csv'

def generate_file_list(p_directory):
    all_files = os.listdir(p_directory)
    
    list_files = []
    for file in all_files:
        if '.py' in file:
            continue
        list_files.append(os.path.join(p_directory,file))
    return list_files

def generate_summary_results(list_full_paths):
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
    save_name = base_name + '_N_'+str(len(r_mean)) + '.csv'
    df.to_csv(os.path.join(directory,save_name))

    ax.bar(label_coords-1.5*width_bar,ratios*dec_df[labels[0]],width=0.2,label=labels[0])
    ax.bar(label_coords-0.5*width_bar,ratios*dec_df[labels[1]],width=0.2,label=labels[1])
    ax.bar(label_coords+0.5*width_bar,ratios*dec_df[labels[2]],width=0.2,label=labels[2])
    ax.bar(label_coords+1.5*width_bar,ratios*dec_df[labels[3]],width=0.2,label=labels[3])
    ax.set_xticks(label_coords)
    ax.set_xlabel('Proportion of message bin scaled to total messages')
    ax.set_xlabel('One hot vector message index')
    ax.legend()
    ax.set_title('Summary message assessment results conditioned on total messages, N = ' + str(N))
    return fig

def plot_single_dimension(parent_fig,parent_ax,linestyle,p_df,p_query):
    strings =  list(filter(lambda x:p_query in x,p_df.columns))
    for r in strings:
        opponent = ''
        for s in r.split(' ')[:-2]:
            opponent = opponent + s + ' '
        curr_data = p_df[r].values
        ax.plot(curr_data,label = opponent[:-1],linestyle = linestyle)    
    return fig,ax


               

#need to know a priori that reward comes first, prediction comes second, and repeats every two rows
#plt.plot(r_mean,label='r mean');plt.plot(r_std,label='r std');plt.legend()
#plt.plot(q_mean,label='q mean');plt.plot(q_std,label='q std');plt.legend()
#plt.plot(q_mean,label='Predicted mean');plt.plot(r_mean,label='Reward mean');plt.legend()

df = pd.read_csv(result_csv)

r_mean_str = 'reward (mean)'
r_std_str = 'reward (std)' 
q_mean_str = 'prediction (mean)' 
q_std_str = 'prediction (std)'

# fig, ax = plt.subplots(figsize=(10,7))
# fig,ax = plot_single_dimension(fig,ax,'solid',df,r_mean_str)
# fig,ax = plot_single_dimension(fig,ax,'dotted',df,q_mean_str)
# fig.legend()
# fig.show()
                  


fig, ax = plt.subplots(figsize=(10,7))
fig,ax = plot_single_dimension(fig,ax,'solid',df,r_mean_str)
fig.legend(loc=1)
fig.suptitle('True reward mean')
fig.show()
                  

                  

#if __name__ == '__main__':

