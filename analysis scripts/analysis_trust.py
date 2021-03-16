# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:38:22 2021

analyze communicating player trust results

There's  a bit more here than the other scripts.

One set of functions takes a class object and makes a communication plot,
typically of trust results given Alternator base opponent.

The other set is for plotting accuracy in judging intent over time.

#Note for trust there is no assymetric rewards, just 5 for accurate and 0 for not.

@author: Jasper
"""

from axelrod.action import Action
C, D = Action.C, Action.D

import numpy as np
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt

csv_trust= r'C:\Users\Jasper\Desktop\Machine Learning\6505\output\csvs\Trust Q Learner\QL_N_998.csv'

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

def generate_deception_table(MJ_Communicator):
    """
    For the communicating player, the three lists of interest are coherent
    i.e. they commands are all in player.strategy() as:
        list_intent_received.append(prev_intent)
        list_intent_assessment.append(prev_assessment)
        list_intent_true.append(opponent.history[-1])
    """
    intent_rx = MJ_Communicator.list_intent_received        #what was received on turn n-1 (one hot vector)
    intent_tx = MJ_Communicator.list_intent_sent            #what was sent on turn n-1 (one hot vector)
    intent_assess = MJ_Communicator.list_intent_assessment  #what was assessed value of intent on turn n-1 (C or D)
    intent_true = MJ_Communicator.list_intent_true          #what the opponent did on turn n-1 (C or D)
    
    result = []
    for index in range(10):
        specific_message= torch.zeros(10)
        specific_message[index] = 1
        num_C_true = 0
        num_D_true = 0
        num_C_assess = 0
        num_D_assess = 0
        
        for rx,assess,true in zip(intent_rx,intent_assess,intent_true):
            if torch.all(torch.eq(rx,specific_message)): #checks if all values are equal
                if true == C: num_C_true += 1
                if true == D: num_D_true += 1
                if assess == C: num_C_assess += 1
                if assess == D: num_D_assess += 1
        result.append(np.array([num_C_true,num_C_assess,num_D_true,num_D_assess]))
    labels = ['True C','Identified C','True D','Identified D']
    return result,labels

def express_deception_table_as_percentages(deception_table,labels):
    result = deception_table
    c_true = []
    c_assess = []
    d_true = []
    d_assess = []
    N = []
    for entry in result:
        total = entry[1]+entry[3]
        N.append(total)
        c_true.append(entry[0]/total)
        c_assess.append(entry[1]/total)
        d_true.append(entry[2]/total)
        d_assess.append(entry[3]/total)
    df = pd.DataFrame()
    df['Total'] = N
    df[labels[0]] = c_true
    df[labels[1]] = c_assess
    df[labels[2]] = d_true
    df[labels[3]] = d_assess
    return df
    
def plot_deception_summary(dec_df,deception_table_labels):
    nums = dec_df['Total']
    N = np.sum(nums)
    ratios = nums/N
    labels = deception_table_labels 
    message_labels = list(np.arange(1,11))
    label_coords = np.arange(len(message_labels))
    width_bar = 0.2
    fig,ax = plt.subplots(figsize=(10,8))
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

df_t = pd.read_csv(csv_trust) 

for p in players:
    fig, ax = plt.subplots(figsize=(10,7))
    fig,ax = plot_single_dimension(fig,ax,'solid',df_q,p + ' ' +r_mean_str, 'Trust QL Reward vs ')
    fig,ax = plot_single_dimension(fig,ax,'dashed',df_c,p + ' ' +q_mean_str, 'Trust QL Predicted vs ')
    fig.legend(loc=(0.4,0.2))
    #fig.legend()
    fig.suptitle('Trust Agent in Triple-Q Agent vs \n' + p + '\n' + q_mean_str + ', N = 10000' ,fontsize=14)
    fig.savefig('pdf\Trust Agent in Triple-Q Agent vs ' + p + ' ' + q_mean_str +'.pdf',
                dpi=300)
    fig.savefig('png\Trust Agent in Triple-Q Agent vs ' + p + ' ' + q_mean_str +'.png',
                dpi=300)
    


#Look at trust messages received v judged. Use alternator. Needs a base object.
"""
deception_table, deception_table_labels = generate_deception_table(MJ_Communicator)
dec_df = express_deception_table_as_percentages(deception_table,deception_table_labels)
plt_comm_results = plot_deception_summary(dec_df,deception_table_labels)
plt_comm_results.show()

rewards = np.asarray(MJ_Communicator.trust.list_reward)
basis = np.arange(len(rewards))
r = np.cumsum(rewards)

plt.plot(rewards)
plt.plot(r)
"""
