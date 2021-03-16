# -*- coding: utf-8 -*-
"""

compare expected reward of two agents. (typically base and conviction)

@author: Jasper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


csv_QL = r'C:\Users\Jasper\Desktop\Machine Learning\6505\output\csvs\Base Q Learner\QL_N_999.csv'
csv_conviction = r'C:\Users\Jasper\Desktop\Machine Learning\6505\output\csvs\Conviction Q Learner\QL_N_998.csv'

players = [
     'AdaptorLong Paranoid Michael Scott',
     'Cautious QLearner Paranoid Michael Scott',
     'Cooperator Paranoid Michael Scott',
     'Defector Paranoid Michael Scott',
     'Evolved ANN 5 Noise 05 Paranoid Michael Scott',
     'Evolved ANN Paranoid Michael Scott',
     'Grudger Paranoid Michael Scott',
     'Random Paranoid Michael Scott',
     'Tit For Tat Paranoid Michael Scott',
     'Worse and Worse Paranoid Michael Scott']

r_mean_str = 'reward (mean)'
r_std_str = 'reward (std)' 
q_mean_str = 'prediction (mean)' 
q_std_str = 'prediction (std)'

def plot_single_dimension(parent_fig,parent_ax,linestyle,p_df,p_query,label_add=''):
    #all list entries with passed p_query string in it
    strings =  list(filter(lambda x:p_query in x,p_df.columns)) 
    for r in strings:
        opponent = ''
        for s in r.split(' ')[:-2]:
            opponent = opponent + s + ' '
        curr_data = p_df[r].values
        parent_ax.plot(curr_data,label = label_add + opponent[:-1] ,linestyle = linestyle)    
    return parent_fig,parent_ax

def plot_q_and_r_comparisons(csv_QL,csv_conviction,players)
    """
    Plots the average results of the 10000 game run against each other,
    of the base agent and conviction agent.
    """
    df_q = pd.read_csv(csv_QL) 
    df_c = pd.read_csv(csv_conviction)
    
    for p in players:
        fig, ax = plt.subplots(figsize=(10,7))
        fig,ax = plot_single_dimension(fig,ax,'solid',df_q,p + ' ' +q_mean_str, 'Base QL vs ')
        fig,ax = plot_single_dimension(fig,ax,'dashed',df_c,p + ' ' +q_mean_str, 'Conviction QL vs ')
        fig.legend(loc=(0.4,0.2))
        #fig.legend()
        fig.suptitle('Triple-Q Agent vs \n' + p + '\n' + q_mean_str + ', N = 10000' ,fontsize=14)
        fig.savefig('pdf\Triple-Q Agent vs ' + p + ' ' + q_mean_str +'.pdf',
                    dpi=300)
        fig.savefig('png\Triple-Q Agent vs ' + p + ' ' + q_mean_str +'.png',
                    dpi=300)
        
        fig, ax = plt.subplots(figsize=(10,7))
        fig,ax = plot_single_dimension(fig,ax,'solid',df_q,p + ' ' +r_mean_str, 'Base QL vs ')
        fig,ax = plot_single_dimension(fig,ax,'dashed',df_c,p + ' ' +r_mean_str, 'Conviction QL vs ')
        fig.legend(loc=(0.4,0.2))
        #fig.legend()
        fig.suptitle('Triple-Q Agent vs \n' + p + '\n' + r_mean_str + ', N = 10000',fontsize=14)
        fig.savefig('pdf\Triple-Q Agent vs ' + p + ' ' + r_mean_str +'.pdf',
                    dpi=300)
        fig.savefig('png\Triple-Q Agent vs ' + p + ' ' + r_mean_str +'.png',
                    dpi=300)
    
plot_q_and_r_comparisons(csv_QL,csv_conviction,players)