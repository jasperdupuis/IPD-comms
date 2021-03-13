# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:23:15 2021

@author: Jasper

"""

import axelrod as axl
from axelrod.action import Action, actions_to_str
from axelrod.player import Player

import numpy as np

from collections import OrderedDict
from typing import Dict, Union

Score = Union[int, float]

C, D = Action.C, Action.D

OUTPUT_DIR = "./output/"
TOURNAMENT_RESULTS_FILE = "tournament_results.png"
TOURNAMENT_PAYOFFS_FILE = "tournament_payoffs.png"

Score = Union[int,float]
C, D = Action.C, Action.D
            
class Q_Learner_6505(axl.RiskyQLearner):
    """
    This extends the axl.RiskyQLearner class.
    
    The base class only considers the opponent's moves in state.
    When in reality value of a state should consider own actions.

    It has a memory by default of 3 previous turns, but this is a parameter
    set in set_params

    This also doesn't use the (what I think is) useless information
    which is total cooperations to date in the match, which was formerly
    also included in the state (and prevented revisiting states)
    """
    
    name = "M&J's Q learner"
    
    #Default values, change with set_params
    learning_rate = 0.9 #learns fast
    discount_rate = 0.9 # cares about the future
    action_selection_parameter = 0.1 #bias towards exploration to visit new states
    memory_length = 3 # number of turns recalled in state
        
    def set_params(self,learning,discount,select_param,memory_length):
        self.learning_rate = learning
        self.discount_rate = discount
        self.action_selection_parameter = select_param
        self.memory_length = memory_length
    
    
    def find_state(self, opponent: Player) -> str:
        """
        Finds the my_state (the opponents last n moves +
        its previous proportion of playing C) as a hashable state
        """
        prob = "{:.1f}".format(opponent.cooperations)
        action_str_opp = actions_to_str(opponent.history[-self.memory_length :])
        action_str_own = actions_to_str(self.history[-self.memory_length :])
        return action_str_own + action_str_opp
    
