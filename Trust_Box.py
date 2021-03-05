import numpy as np
import torch

import axelrod as axl
from axelrod.player import Player

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
    
    def __init__(self):
        super.__init__(self)


    def strategy(self, opponent: Player,opponent_intent):
        #return trust
        return C

    def strategy(intent_received,
                assessment_prev,
                prev_nme_action):
        return C
    
    
class Ned_Stark(Trust_Box):
    """
    Takes opponent at face value.
    (We can replace this with Cooperator)
    """
    name = 'Trusting'
    def strategy():
        return C

class Tywin_Lannister(Trust_Box):
    """
    Assumes everyone is out to get him.
    (We can replace this with Defector)
    """
    
    name = 'Paranoid'
    
    def strategy():
        return D

class Trust_Q_Learner(Trust_Box,axl.RiskyQLearner):
    """
    Re implement the Q Learner to use the variable length intent vector
    """
    name = "Trust Q learner"
    
    #Default values, change with set_params
    learning_rate = 0.5 #learns fast
    discount_rate = 0.5 # cares about the future
    action_selection_parameter = 0.1 #bias towards exploration to visit new states
    memory_length = 1 # number of turns recalled in state
        
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
                        prev_nme_action)
        if assessment_prev == prev_nme_action: return 5
        else: return 0
        
    
    def strategy(intent_received,
                assessment_prev,
                prev_nme_action):
        """
        Runs a qlearn algorithm while the tournament is running.
        Reimplement the base class strategy to work with trust communication
        """
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
        return action
    
#class Trust_DQN(Trust_Box):
"""
Wrap the normal DQN learner.
"""
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
