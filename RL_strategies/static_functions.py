# -*- coding: utf-8 -*-
"""

Functions used by our RL agents that don't chage across implementation

"""

import numpy as np
import csv
from typing import Union
import torch

from axelrod.action import Action

Score = Union[int,float]
C, D = Action.C, Action.D
      
def write_base_opponent_decision(name1,name2,own_base_actions,opponent_actions,own_decisions):
        """
        Communicating Player only.
        
        Due to temporal difference need to make sure RECORDED own actions and
        opponent action line up in indices. Single state's tuple but refer to different
        true time steps.
        opponent_action: the result of turn n-1
        own_base_actions: the result for turn n
        own_decisionsL the result for turn n
        """
        if len(own_base_actions) < 1: return
        
        opponent_actions = opponent_actions[1:]
        own_base_actions = own_base_actions[:-1]
        own_decisions = own_decisions[:-1]

        with open(r'output\csvs\\_'+name1+'_' +name2+'_q versus r.csv', 'a',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([name1 + "_v_" + name2 +"_base action"]+own_base_actions)
            writer.writerow([name1 + "_v_" + name2 +"_decision"]+own_decisions)
            writer.writerow([name1 + "_v_" + name2 +"_opponent action"]+opponent_actions)
    
def write_actions(name1,name2,own_base_actions,opponent_actions):
        if len(own_base_actions) < 1: return
        
        opponent_actions = opponent_actions[1:]
        own_base_actions = own_base_actions[:-1]
        
        with open(r'output\csvs\\_'+name1+'_' +name2+'_actions log.csv', 'a',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([name1 + "_v_" + name2 +"own action"]+own_base_actions)
            writer.writerow([name1 + "_v_" + name2 +"_opponent action"]+opponent_actions)
    
    
def write_q_and_rewards(name1,name2,reward,predicted_reward = None):
        
        """
        Due to temporal difference need to make sure RECORDED reward and
        predicted reward line up in indices. The reward and predicted are
        for part of a single state's tuple but refer to different
        true time steps.
        reward: the result of turn n-1
        predicted_reward: the result for turn n
        """
        if len(reward) < 1: return
        
        reward = reward[1:]
        predicted_reward = predicted_reward[:-1]

        with open(r'output\csvs\\_'+name1+'_' +name2+'_q versus r.csv', 'a',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([name1 + "_v_" + name2 +"_reward"]+reward)
            writer.writerow([name1 + "_v_" + name2 +"_reward_predicted"]+predicted_reward)
            
            
def action_to_tensor(action):
    if action == C: return torch.tensor([1.,0.])
    if action == D: return torch.tensor([0.,1.])

def int_to_action(integer): #not used right now
    if integer==1: return C
    if integer==0: return D

def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)    

