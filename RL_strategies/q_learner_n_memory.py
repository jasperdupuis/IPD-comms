# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:23:15 2021

@author: Jasper

"""

import RL_strategies.static_functions as RL_func

import axelrod as axl
from axelrod.action import Action, actions_to_str
from axelrod.player import Player

import numpy as np
import csv

from collections import OrderedDict
from typing import Dict, Union
from copy import deepcopy, copy

Score = Union[int,float]
C, D = Action.C, Action.D

class Q_Learner_6505(Player):
    """
    This copies and extends the axl.RiskyQLearner class.
    """
    
    name = "M&J's Q learner"    
    classifier = {
        "memory_depth": float("inf"),  # Long memory
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    #Default static values, change with set_params
    learning_rate = 0.9 #learns fast
    discount_rate = 0.9 # cares about the future
    action_selection_parameter = 0.1 #bias towards exploration to visit new states
    memory_length = 3 # number of turns recalled in state


    def __init__(self) -> None:
        """Initialises the player by picking a random strategy."""

        super().__init__()

        # Set this explicitly, since the constructor of super will not pick it up
        # for any subclasses that do not override methods using random calls.
        self.classifier["stochastic"] = True

        self.prev_action = None  # type: Action
        self.original_prev_action = None  # type: Action
        self.score = 0
        self.Qs = OrderedDict({"": OrderedDict(zip([C, D], [0, 0]))})
        self.Vs = OrderedDict({"": 0})
        self.prev_state = ""
        
        self.list_reward = []
        self.list_predicted_reward = []
        self.finished_opponent = "none yet"

    def reset(self):
        """
        copy pasta from base for now
        """
        
        Qs = deepcopy(self.Qs)
        Vs = deepcopy(self.Vs)
        
        RL_func.write_q_and_rewards(
                            self.name,
                            self.finished_opponent,
                            self.list_reward,
                            self.list_predicted_reward)
        
        super().__init__(**self.init_kwargs)
        
        self.list_reward = [] 
        self.list_predicted_reward = []
        self.Qs = Qs
        self.Vs = Vs

    def clone(self):
        """
        Copy and extend the base Player.clone() function so that networks
        save across clone operations.
        """
        
        Q_save = deepcopy(self.Qs)
        V_save = deepcopy(self.Vs)
        
        
        cls = self.__class__
        new_player = cls(**self.init_kwargs)
        new_player.match_attributes = copy(self.match_attributes)
        
        learning = self.learning_rate
        discount = self.discount_rate 
        select_param = self.action_selection_parameter
        memory_length = self.memory_length
        
        new_player.set_params(learning,discount,select_param,memory_length)
        new_player.Qs = deepcopy(Q_save)
        new_player.Vs = deepcopy(V_save)
        
        return new_player

    def set_params(self,
                   learning=0.9,
                   discount=0.5,
                   select_param=0.05,
                   memory_length=3):
        self.learning_rate = learning
        self.discount_rate = discount
        self.action_selection_parameter = select_param
        self.memory_length = memory_length

    def receive_match_attributes(self):
        (R, P, S, T) = self.match_attributes["game"].RPST()
        self.payoff_matrix = {C: {C: R, D: S}, D: {C: T, D: P}}

    def strategy(self, opponent: Player) -> Action:
        """
        Exact same as the original strategy, except now it saves reward to the list
        """
        reward=-1
        if len(self.history) == 0:
            self.prev_action = self._random.random_choice()
            self.original_prev_action = self.prev_action
        state = self.find_state(opponent)
        reward = self.find_reward(opponent)
        if state not in self.Qs:
            self.Qs[state] = OrderedDict(zip([C, D], [0, 0]))
            self.Vs[state] = 0
        self.perform_q_learning(
            self.prev_state, state, self.prev_action, reward
        )
        action = self.select_action(state)
        self.prev_state = state
        self.prev_action = action
        self.list_reward.append(reward)
        self.finished_opponent = opponent.name
        return action
    
        
    def select_action(self, state: str) -> Action:
        """
        Selects the action based on the epsilon-greedy policy
        """
        rnd_num = self._random.random()
        p = 1.0 - self.action_selection_parameter
        if rnd_num < p:
            action = max(self.Qs[state], key=lambda x: self.Qs[state][x])
            self.list_predicted_reward.append(self.Qs[state][action])
            return action
        self.list_predicted_reward.append(-1) #this means a random action was taken and we need to record that
        return self._random.random_choice()
    
    def find_state(self, opponent: Player) -> str:
        """
        Finds the my_state (the opponents last n moves +
        its previous proportion of playing C) as a hashable state
        """
        prob = "{:.1f}".format(opponent.cooperations)
        action_str_opp = actions_to_str(opponent.history[-self.memory_length :])
        action_str_own = actions_to_str(self.history[-self.memory_length :])
        return action_str_own + action_str_opp

    def perform_q_learning(self, prev_state: str, state: str, action: Action, reward):
        """
        Performs the qlearning algorithm
        """
        try:
            self.Qs[prev_state][action] = (1.0 - self.learning_rate) * self.Qs[prev_state][action] \
                + self.learning_rate * (reward + self.discount_rate * self.Vs[state])
        except:
            x=1
        self.Vs[prev_state] = max(self.Qs[prev_state].values())

    def find_reward(self, opponent: Player) -> Dict[Action, Dict[Action, Score]]:
        """
        Finds the reward gained on the last iteration
        """

        if len(opponent.history) == 0:
            opp_prev_action = self._random.random_choice()
        else:
            opp_prev_action = opponent.history[-1]
        return self.payoff_matrix[self.prev_action][opp_prev_action]

