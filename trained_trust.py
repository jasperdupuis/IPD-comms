# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 05:36:26 2021

@author: Mani Teja Varma
"""

  
from collections import OrderedDict
from typing import Dict, Union

import axelrod as axl
from axelrod.action import Action, actions_to_str
from axelrod.player import Player
import pickle
from trust_box import Trust_Box
import numpy as np

from RL_strategies.q_learner_n_memory import Q_Learner_6505

Score = Union[int, float]

C, D = Action.C, Action.D


class Trained_Trust_QLearner(Trust_Box,Q_Learner_6505):
    name = "Trained Trust Q Learner"
    
    def __init__(self):
        super().__init__()
        qs_and_vs = []
        with open(r"trust_learner_values", "rb") as input_file:
            qs_and_vs = pickle.load(input_file)
        
        self.Qs = qs_and_vs[0]
        self.Vs = qs_and_vs[1]
        

    def find_state(self, intent_received)-> str:
        """
        translate the received intent to a hashable state
        """
        intent = intent_received.numpy()
        return np.array2string(intent)
    
    def find_reward(self,
                        assessment_prev,
                        prev_nme_action):
        if assessment_prev == prev_nme_action: return 5
        else: return 0
    
    def select_action(self, state: str) -> Action:
        """
        Selects the action based on the epsilon-greedy policy
        """
        rnd_num = self._random.random()
        p = 1.0 - self.action_selection_parameter
        if rnd_num < p and (state in self.Qs):
            action = max(self.Qs[state], key=lambda x: self.Qs[state][x])
            self.list_predicted_reward.append(self.Qs[state][action])
            return action
        self.action_selection_parameter = self.action_selection_parameter*self.epsilon_decay
        self.list_predicted_reward.append(-1) #this means a random action was taken and we need to record that
        return self._random.random_choice()
    
    def strategy(self,
                intent_received,
                intent_received_prev,
                assessment_prev,
                opponent):
        """
        Runs a qlearn algorithm while the tournament is running.
        Reimplement the base class strategy to work with trust communication
        """
        if len(self.history) < 2:
            self.prev_action = C
            self.original_prev_action = C
        
        state = self.find_state(intent_received)
        reward = self.find_reward(assessment_prev,
                                  opponent.history[-1])
        
        
        action = self.select_action(state) #this also appends q values to list_predicted_rewards
        self.prev_state = state
        self.prev_action = action
        
        self.list_reward.append(reward)
        self.finished_opponent = opponent.name
        return action

