# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:38:22 2021

look at communicating player performance overall, by comparing
conviction output, base output, and opponent action
to determine which strategy was better at the conviction stage.

i.e. Demonstrate learning of the conviction box, looking directly
at reward differential.
base - (base+trust+conviction)

NOTE:
    
    The files have three row repeating patterns (from RL_func.write_base_opponent_decision)
    1)base action
    2)opponent action
    3)decision
    
In some (many) files, 2) and 3) are BOTH labelled "decision"

@author: Jasper
"""
import time
import os
import numpy as np
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt

from csv import reader

from axelrod.action import Action
C, D = Action.C, Action.D

directory = r'C:\Users\Jasper\Desktop\Machine Learning\6505\output\csvs\Communicating_Player'
result_csv = directory + r'\conviction_v_base.csv'

NUM_GAMES = 10000
NUM_TURNS = 996 #useable

def generate_file_list(p_directory):
    all_files = os.listdir(p_directory)
    
    list_files = []
    for file in all_files:
        if '.py' in file:
            continue
        list_files.append(os.path.join(p_directory,file))
    return list_files

def find_reward(own_action,
                opp_action):
    """
    I'm lazy
    """
    if own_action == 'C' and opp_action == 'C': return 3
    if own_action == 'C' and opp_action == 'D': return 0
    if own_action == 'D' and opp_action == 'C': return 5
    if own_action == 'D' and opp_action == 'D': return 1
    
def process_single_game_results(base_actions,decisions,opponent_actions):
    """
    must pass iterables
    """    
    base_reward = []
    for own,opp in zip(base_actions,opponent_actions):
        base_reward.append(find_reward(own,opp))    
    comm_reward = []
    for own,opp in zip(decisions,opponent_actions):
        comm_reward.append(find_reward(own,opp))
    return base_reward,comm_reward
        
def generate_single_opp_results(file):
    r_base = np.zeros((NUM_GAMES,NUM_TURNS))
    r_conv = np.zeros((NUM_GAMES,NUM_TURNS))
    counter = 0
    with open(file,'r') as r:
        while counter < NUM_GAMES-1:
            base_actions = r.readline().split(',')[2:-2]
            opp_actions = r.readline().split(',')[1:-2]
            decisions = r.readline().split(',')[1:-2]
            b, c = process_single_game_results(base_actions,decisions,opp_actions)
            r_base[counter,:] = b
            r_conv[counter,:] = c
            counter+=1
        return r_base,r_conv

def generate_summary_results_base_v_decision(list_full_paths):
    df = pd.DataFrame()    
    base_name = list_full_paths[0].split('_')[2]
    for file in list_full_paths:
        opponent_name = file.split('_')[3]
        r_b,r_c = generate_single_opp_results(file)   
        b_mean,b_std = np.mean(r_b,axis=0),np.std(r_b,axis=0)
        c_mean,c_std = np.mean(r_c,axis=0),np.std(r_c,axis=0)
        df[opponent_name + ' base reward (mean)'] = b_mean
        df[opponent_name + ' base reward (std)'] = b_std
        df[opponent_name + ' decision reward (mean)'] = c_mean
        df[opponent_name + ' decision reward (std)'] = c_std
    return df

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

def compare_results(p_df,parent_fig,parent_ax,opponent):
    """
    Compute and plot base - (Base+trust+conviction)    
    """
    b_data = p_df[opponent + ' base reward (mean)'].values
    c_data = p_df[opponent + ' decision reward (mean)'].values
    delta_sum = np.cumsum(b_data-c_data)
    parent_ax.plot(delta_sum,label = opponent[:-1])    
    return parent_fig,parent_ax
    

"""
file_list = generate_file_list(directory)
df = generate_summary_results_base_v_decision(file_list)
df.to_csv(os.path.join(directory,"QL_N_996"))
"""

"""
b_mean_str = 'base reward (mean)'
b_std_str = 'base reward (std)' 
c_mean_str = 'decision reward (mean)' 
c_std_str = 'decision reward (std)'

fig, ax = plt.subplots(figsize=(10,7))
fig,ax = plot_single_dimension(fig,ax,'solid',df,c_mean_str)
fig,ax = plot_single_dimension(fig,ax,'dashed',df,b_mean_str)
fig.legend(loc=1)
fig.suptitle('True reward mean')
fig.show()
"""

players = []
for f in file_list:
    players.append(f.split('_')[3])

fig, ax = plt.subplots(figsize=(10,7))    
for p in players:
    fig,ax = compare_results(df,fig,ax,p)
fig.legend(loc=1)
fig.suptitle('Reward delta: \n base - (base+trust+conviction)')
fig.show()



"""
"""











