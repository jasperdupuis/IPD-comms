# -*- coding: utf-8 -*-
"""

Let's compare cooperation and/or defection rate of the base and conviction agents.

To make statistical have to go back to the raw CSVs.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

directory = r'C:\Users\Jasper\Desktop\Machine Learning\6505\output\csvs\Communicating_Player'
csv_target_dir = directory +  r'\summary tables'

NUM_GAMES = 10000
NUM_TURNS = 996 #useable

PLAYERS = [
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

def generate_file_list(p_directory):
    all_files = os.listdir(p_directory)
    
    list_files = []
    for file in all_files:
        if '.py' in file:
            continue
        if not ('.csv' in file): continue
        list_files.append(os.path.join(p_directory,file))
    return list_files

def process_single_game_results(base_actions,decisions):
    """
    must pass iterables
    """    
    base_cooperations = sum(x =='C' for x in base_actions)
    base_defections= sum(x =='D' for x in base_actions)
    
    conviction_cooperations = sum(x =='C' for x in decisions)
    conviction_defections= sum(x =='D' for x in decisions)
    
    N = base_cooperations + base_defections
    
    return base_cooperations,base_defections,conviction_cooperations,conviction_defections,N
        
def generate_single_opp_results(file):
    result = np.zeros((NUM_GAMES,9))
    counter = 0
    with open(file,'r') as r:
        while counter < NUM_GAMES-1:
            base_actions = r.readline().split(',')[2:-2]
            decisions = r.readline().split(',')[1:-2]
            bc,bd,cc,cd,N = process_single_game_results(base_actions,decisions)
            result[counter,0] = N
            result[counter,1] = bc
            result[counter,2] = bc/N
            result[counter,3] = bd
            result[counter,4] = bd/N
            result[counter,5] = cc
            result[counter,6] = cc/N
            result[counter,7] = cd
            result[counter,8] = cd/N
            counter+=1
        return result

def generate_base_v_decision_coop_defec_table(file):
    df = pd.DataFrame()    
    result = generate_single_opp_results(file)   
    df['Number of turns'] = result[:,0]
    df['Base cooperations'] = result[:,1]
    df['Base cooperations rate'] = result[:,2]
    df['Base defections'] = result[:,3]
    df['Base defections rate'] = result[:,4]
    df['Conviction cooperations'] = result[:,5]
    df['Conviction cooperations rate'] = result[:,6]
    df['Conviction defections'] = result[:,7]
    df['Conviction defections rate'] = result[:,8]
    return df

def generate_results_csv(p_directory,target_directory):
    """
    Main function that makes the coop/defec CSV.
    """

    file_list = generate_file_list(p_directory)
    dfs = []
    for f in file_list:
        opponent = f.split('_')[3]
        opp_id = [opponent] * 10000
        df = generate_base_v_decision_coop_defec_table(f)
        df['Opponent'] = opp_id
        dfs.append(df)
        
    df_final = pd.concat(dfs)
    df_final.to_csv(target_directory+'\\Cooperation and Defection Summary.csv')

def create_summary_dictionary(
        csv_file_path = csv_target_dir+'\\Cooperation and Defection Summary.csv'):
    df = pd.read_csv(csv_file_path)
    result = dict()
    for p in PLAYERS:
        sub_df = df[df['Opponent'] == p]
        base_c = sub_df['Base cooperations rate'].values
        base_d = sub_df['Base defections rate'].values
        conv_c = sub_df['Conviction cooperations rate'].values
        conv_d = sub_df['Conviction defections rate'].values
        sub_res = {'Base cooperation rate mean' : np.mean(base_c),
                   'Base cooperation rate std' : np.std(base_c),
                   'Conviction cooperation rate mean' : np.mean(conv_c),
                   'Conviction cooperation rate std' : np.std(conv_c),
                   'Base defection rate mean' : np.mean(base_d),
                   'Base defection rate std' : np.std(base_d),
                   'Conviction defection rate mean' : np.mean(conv_d),
                   'Conviction defection rate std' : np.std(conv_d),
                   }
        sub_res = dict(sub_res)    
        result[p] = sub_res
    return result

result = create_summary_dictionary()
labels = ['Base Cooperations', 'Conviction Cooperations' , 'Base Defections' , 'Conviction Defections']
xpos = np.arange(len(labels))
for key,value in result.items():
    means = list(value.values())[0::2]
    std = list(value.values())[1::2]
    fig,ax = plt.subplots(figsize=(10,7))
    ax.bar(xpos,means,yerr=std,tick_label=labels)
    ax.set_ylabel('Action rate')
    fig.suptitle(key,fontsize=14)
    fig.show()    
    
    





