import numpy as np
import torch

#Imports to support axl implementations and carry overs.
import axelrod as axl
from axelrod.action import Action, actions_to_str
from axelrod.player import Player
from collections import OrderedDict
from typing import Dict, Union
Score = Union[int, float]
C, D = Action.C, Action.D


import RL_strategies.dqn_learner_intergame_memory
from RL_strategies.dqn_learner_intergame_memory import DQN_Learner_Intergame_Memory
from RL_strategies.q_learner_n_memory import Q_Learner_6505


class Trust_Box(axl.Player):
    """
    Meant to return action from axelrod, but using to return some measure
    of opponent's intent based on received information.
    
    C = Expect opponent to cooperate based on intel
    D = Expect opponent to defect based on intel
    
    """
    
    list_reward = []
    

    def strategy(self, opponent: Player,opponent_intent):
        #return trust
        return C

    def strategy(self,
                intent_received,
                intent_received_prev,
                assessment_prev,
                prev_nme_action):
        return C
    
    def append_reward(self,reward):
        self.list_reward.append(reward)
    
class Ned_Stark(Trust_Box):
    """
    Takes opponent at face value.
    (We can replace this with Cooperator)
    """
    name = 'Trusting'
    
    def strategy(self):
        return C
    
    def strategy(self,
                intent_received,
                intent_received_prev,
                assessment_prev,
                prev_nme_action):
        return C

class Tywin_Lannister(Trust_Box):
    """
    Assumes everyone is out to get him.
    (We can replace this with Defector)
    """
    
    name = 'Paranoid'
    
    def strategy(self):
        return D
    
    def strategy(self,
                intent_received,
                intent_received_prev,
                assessment_prev,
                prev_nme_action):
        return D

class Trust_Q_Learner(Trust_Box,axl.RiskyQLearner):
    """
    Re implement the Q Learner to use the variable intent vector
    (Had to change state and inputs)
    """
    name = "Trust Q learner"
    
    #Default values, change with set_params
    learning_rate = 0.5 #learns fast
    discount_rate = 0.5 # cares about the future
    action_selection_parameter = 0.1 #bias towards exploration to visit new states
    memory_length = 1 # number of turns recalled in state
        
    def __init__(self) -> None:
        """Initialises the player by picking a random strategy."""

        super().__init__()

        # Set this explicitly, since the constructor of super will not pick it up
        # for any subclasses that do not override methods using random calls.
        self.classifier["stochastic"] = True

        self.prev_action = C # type: Action
        self.original_prev_action = C# type: Action
        self.score = 0
        self.Qs = OrderedDict({"": OrderedDict(zip([C, D], [0, 0]))})
        self.Vs = OrderedDict({"": 0})
        self.prev_state = ""
    
    def set_params(self,
                   learning,
                   discount,
                   select_param,
                   memory_length=1):
        self.learning_rate = learning
        self.discount_rate = discount
        self.action_selection_parameter = select_param
        self.memory_length = memory_length
        
        
        
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
        
    
    def strategy(self,
                intent_received,
                intent_received_prev,
                assessment_prev,
                prev_nme_action):
        """
        Runs a qlearn algorithm while the tournament is running.
        Reimplement the base class strategy to work with trust communication
        """
        if len(self.history) == 0:
            self.prev_action = C
            self.original_prev_action = C
        
        state = self.find_state(intent_received)
        reward = self.find_reward(assessment_prev,
                                  prev_nme_action)
        if state not in self.Qs:
            self.Qs[state] = OrderedDict(zip([C, D], [0, 0]))
            self.Vs[state] = 0
        self.perform_q_learning(self.prev_state, state, self.prev_action, reward)
        if state not in self.Qs:
            action = random_choice()
        else:
            action = self.select_action(state)
        self.prev_state = state
        self.prev_action = action
        self.append_reward(reward)
        return action
    
#class Trust_DQN(Trust_Box):
"""
Wrap the normal DQN learner.
"""
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    