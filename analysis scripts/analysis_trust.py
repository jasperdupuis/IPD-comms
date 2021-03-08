# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:38:22 2021

analyze communicating player trust results

@author: Jasper
"""

from axelrod.action import Action
C, D = Action.C, Action.D

import numpy as np
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt

def generate_deception_table(MJ_Communicator):
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


deception_table, deception_table_labels = generate_deception_table(MJ_Communicator)
dec_df = express_deception_table_as_percentages(deception_table,deception_table_labels)
plt_result = plot_deception_summary(dec_df,deception_table_labels)
plt_result.show()


rewards = np.asarray(MJ_Communicator.trust.list_reward)
basis = np.arange(len(rewards))
r = np.cumsum(rewards)

plt.plot(rewards)
plt.plot(r)

firsthalf = scipy.stats.linregress(basis[5:5000],rewards[5:5000])
secondhalf = scipy.stats.linregress(basis[5000:-5],rewards[5000:-5])
firsthalf
secondhalf


plt.plot(r)

