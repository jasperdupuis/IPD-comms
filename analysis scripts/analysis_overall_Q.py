# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:01:24 2021


compare the track of two games (with communication v no communication)


@author: Jasper
"""
from axelrod.action import Action
C, D = Action.C, Action.D

import numpy as np
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt

PLAYER = MJ_Communicator
OPPONENT = opponent

def find_reward(own_action,
                opp_action):
        """
        I'm lazy
        """
        if own_action == C and opp_action == C: return 3
        if own_action == C and opp_action == D: return 0
        if own_action == D and opp_action == C: return 5
        if own_action == D and opp_action == D: return 1


MJ_base_actions = PLAYER.list_base_action[1:]
MJ_conv_actions = PLAYER.list_decision
opp_base_actions = OPPONENT.list_base_action[1:]
opp_conv_actions = OPPONENT.list_decision

base_reward_self = []
for own,opp in zip(MJ_base_actions,opp_base_actions):
    base_reward_self.append(find_reward(own,opp))

comm_reward_self = []
for own,opp in zip(MJ_conv_actions,opp_conv_actions):
    comm_reward_self.append(find_reward(own,opp))

#cumsum
plt.plot(np.cumsum(base_reward_self),label='Base');plt.plot(np.cumsum(comm_reward_self),label='Comm');plt.legend()
#delta
plt.plot(np.cumsum(base_reward_self)-np.cumsum(comm_reward_self),label='Cumulative reward: \nBase - (Base+Trust+Conviction)');plt.title('Communication improves game result \n Q-learner agents for all three decisions');plt.legend()



