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
    
In some files (especially <20210316 versions of competition),
 2) and 3) are BOTH labelled "decision"

@author: Jasper
"""
import time
import os
import numpy as np
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import imageio

from csv import reader

from axelrod.action import Action
C, D = Action.C, Action.D

filename = r'_M&J Q Learner Trust Q learner Conviction Q learner_Worse and Worse Paranoid Michael Scott_q versus r.csv'
directory = r'/'
result_csv = directory + r'/a.csv'

NUM_GAMES = 500
NUM_TURNS = 1995 #useable, N - 4 i think for all cases.

b_mean_str = 'base reward (mean)'
b_std_str = 'base reward (std)' 
c_mean_str = 'decision reward (mean)' 
c_std_str = 'decision reward (std)'

def generate_file_list(p_directory):
    all_files = os.listdir(p_directory)
    
    list_files = []
    for file in all_files:
        if not('.csv' in file):
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
            base_actions = r.readline().split(',')[3:-2]
            opp_actions = r.readline().split(',')[2:-2]
            decisions = r.readline().split(',')[2:-2]
            assessment = r.readline().split(',')[2:-2]
            b, c = process_single_game_results(base_actions,decisions,opp_actions)
            r_base[counter,:] = b
            r_conv[counter,:] = c
            counter+=1
        return r_base,r_conv
    


def generate_summary_results_base_v_decision(list_full_paths):
    df = pd.DataFrame()    
    base_name = list_full_paths[0].split('_')[1]
    for file in list_full_paths:
        opponent_name = file.split('_')[2]
        r_b,r_c = generate_single_opp_results(file)   
        b_mean,b_std = np.mean(r_b,axis=0),np.std(r_b,axis=0)
        c_mean,c_std = np.mean(r_c,axis=0),np.std(r_c,axis=0)
        df[opponent_name + ' base reward (mean)'] = b_mean
        df[opponent_name + ' base reward (std)'] = b_std
        df[opponent_name + ' decision reward (mean)'] = c_mean
        df[opponent_name + ' decision reward (std)'] = c_std
    return df


def conviction_and_base_agent_performance(file):
    is_coviction_better_than_base = np.zeros((NUM_GAMES,NUM_TURNS))
    counter = 0
    with open(file,'r') as r:
        while counter < NUM_GAMES-1:
            base_actions = r.readline().split(',')[3:-2]
            opp_actions = r.readline().split(',')[2:-2]
            decisions = r.readline().split(',')[2:-2]
            b, c = process_single_game_results(base_actions,decisions,opp_actions)
            for i in range(len(b)):  
                if b[i] > c[i]: 
                    is_coviction_better_than_base[counter,i]  = 1
                elif  b[i] == c[i]:
                    is_coviction_better_than_base[counter,i]  = 0
                elif  b[i] < c[i]:
                    is_coviction_better_than_base[counter,i]  = -1
            counter+=1
        return is_coviction_better_than_base

def compare_assessment_opponentaction(file):
    is_opponent_and_assessment_same = np.zeros((NUM_GAMES,NUM_TURNS))
    counter = 0
    with open(file,'r') as r:
        while counter < NUM_GAMES-1:
            base_actions = r.readline().split(',')[3:-2]
            opp_actions = r.readline().split(',')[2:-2]
            decisions = r.readline().split(',')[2:-2]
            assessment = r.readline().split(',')[2:-2]
            for i in range(len(assessment)):
                if assessment[i] == opp_actions[i]:
                    is_opponent_and_assessment_same[counter,i] = 1
                else:
                    is_opponent_and_assessment_same[counter,i] = 0  
            counter+=1
        return is_opponent_and_assessment_same

def mani_plot(list_full_paths):
    df = pd.DataFrame()    
    base_name = list_full_paths[0].split('_')[1]
    for file in list_full_paths:
        opponent_name = file.split('_')[2]
        is_opponent_and_assessment_same = compare_assessment_opponentaction(file)   
        filenames = []
        for is_coviction_better_than_base_per_game in is_coviction_better_than_base_all_games:
            plt.plot(range(len(is_coviction_better_than_base_per_game)),is_coviction_better_than_base_per_game)
            filename = f'haa{len(filenames)}.png'
            filenames.append(filename)
            plt.savefig(filename)
            plt.title(f'game = {len(filenames)}')
            plt.close()
        with imageio.get_writer('opponent.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        # Remove files
        for filename in set(filenames):
            os.remove(filename)  
        return df

def assessment_plot(list_full_paths):
    df = pd.DataFrame()    
    base_name = list_full_paths[0].split('_')[1]
    for file in list_full_paths:
        opponent_name = file.split('_')[2]
        is_opponent_and_assessment_same_all_games = compare_assessment_opponentaction(file)   
        filenames = []
        for is_opponent_and_assessment_same_per_game in is_opponent_and_assessment_same_all_games[:250]:
            plt.scatter(range(len(is_opponent_and_assessment_same_per_game)),is_opponent_and_assessment_same_per_game)
            filename = f'haa{len(filenames)}.png'
            filenames.append(filename)
            plt.savefig(filename)
            plt.text(2,3, "game number = "+ str(len(filenames)))
            plt.close()
        with imageio.get_writer('opponent.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        # Remove files
        for filename in set(filenames):
            os.remove(filename)  
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


def plot_single_dimension_with_errbars(parent_fig,parent_ax,linestyle,p_df,p_query):
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



#file_list = generate_file_list(directory)
file_list = [filename]
# df = generate_summary_results_base_v_decision(file_list)
# df.to_csv(result_csv)

# df2 = mani_plot(file_list)
# df2.to_csv(result_csv)

df3 = assessment_plot(file_list)
df3.to_csv(result_csv)



# df = pd.read_csv(result_csv)

# players = []
# for f in file_list:
#     if not('.csv' in f): continue
#     players.append(f.split('_')[2])

# fig, ax = plt.subplots(figsize=(10,7))    
# for p in players:
#     fig,ax = compare_results(df,fig,ax,p)
# fig.legend(loc=1)
# fig.suptitle('Mean reward delta for N = 2000 games: \n base - (base+trust+conviction)')
# fig.show()



"""
"""











