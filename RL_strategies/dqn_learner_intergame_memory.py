# -*- coding: utf-8 -*-
"""
Hacks the Player.clone() and Player.reset() functions
to allow DQN learner to remember results across games.

Increases tournament average score substantially
"""

#Axelrod imports
import axelrod as axl
from axelrod.action import Action, actions_to_str
from axelrod.player import Player
from typing import Dict, Union

#our imports
import numpy as np
import torch
from torch import nn
import csv
from copy import deepcopy, copy


OUTPUT_DIR = "./output/"
TOURNAMENT_RESULTS_FILE = "tournament_results.png"
TOURNAMENT_PAYOFFS_FILE = "tournament_payoffs.png"

Score = Union[int,float]
C, D = Action.C, Action.D


def action_to_tensor(action):
    if action == C: return torch.tensor([1.,0.])
    if action == D: return torch.tensor([0.,1.])

def int_to_action(integer): #not used right now
    if integer==1: return C
    if integer==0: return D

def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)    

class DQN_Learner_Intergame_Memory(axl.RiskyQLearner):
    """
    Implement a Deep-Q Learner. 
    Initial implementation borrows heavily from https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435
    
    This method replaces the Q_Learner lookup method with a network with at least
    one hidden layer.
    """
    
    name = "Mani and Jasper's DQN learner"
    update_period = 25 #how many turns to play before training
    r_expect = ['Expected']
    r_actual = ['Actual']
        

    
    #Structures needed for learning + target generation
    optimizer = 'not yet set'
    loss = 'not yet set'
    q_network = 'not yet set'
    target_network = 'not yet set'
    memory_as_list = []

    state = 'not yet set'
    old_state = 'not yet set'
    
    
    #Default values, change with set_params
    learning_rate = 0.9 #learns fast
    discount_rate = 0.9 # cares about the future
    action_selection_parameter = 0.1 #bias towards exploration to visit new states
    memory_length = 3 # number of turns recalled in state
    units_per_hidden = 16    
    target_sync_freq = 10
    
    sync_counter = 0
    n_input_features = 4 # state as vector of 1s and 0s, or C and D, represents 1 turn.
    n_outputs = 2 #action as one hot


    def reset(self):
        """
        write game reward results to file
        then reset
        """
        if len(self.r_actual) < 10:
            super().__init__(**self.init_kwargs)
            return
            
        with open(r'output\csvs\rewards.csv', 'a',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(self.r_expect)
            writer.writerow(self.r_actual)
            
        self.r_expect = ['Expected']
        self.r_actual = ['Actual']
        
        super().__init__(**self.init_kwargs)


    def clone(self):
        """
        Copy and extend the base Player.clone() function so that networks
        save across clone operations.
        """
        cls = self.__class__
        new_player = cls(**self.init_kwargs)
        new_player.match_attributes = copy(self.match_attributes)
        
        t_save = deepcopy(self.target_network)
        q_save = deepcopy(self.q_network)
        n_mem = self.memory_length
        n_out = self.n_outputs        
        new_player.set_params()
        new_player.init_net()
        new_player.target_network.load_state_dict(t_save.state_dict())
        new_player.q_network.load_state_dict(q_save.state_dict())
        return new_player

    def set_params(self,                   
                  memory_turns=3,
                  n_output=2,
                  learning = 0.9,
                  discount = 0.9,
                  select_param = 0.1,
                  units_per_hidden = 16,
                  epsilon=0.1,
                  target_sync_freq = 25):
        """
        Hyperparameter setter, decent defaults provided.
        """
        self.learning_rate = learning
        self.discount_rate = discount
        self.action_selection_parameter = select_param
        self.units_per_hidden = units_per_hidden
        self.epsilon = epsilon
        self.target_sync_freq = target_sync_freq
        self.sync_counter = 0 #hard coded to 0
        
        self.memory_length = memory_turns
        self.n_input_features = 4*memory_turns
        self.n_outputs = n_output
            
    def __init__(self)->None:
        """
        Calls set_params and init_net to ease interaction with the tournament.
        This means this model WILL NOT LEARN FROM GAME TO GAME!
        (NOT Hyperparameters)
        """
        Player.__init__(self)     

    def init_net(self):
        """        
        network instantiated from arguments from set_params
        """    
        self.q_network = nn.Sequential(
        nn.Linear(self.n_input_features,  self.units_per_hidden, bias=True),
        nn.ReLU(),
        nn.Linear( self.units_per_hidden,  self.units_per_hidden, bias=True),
        nn.ReLU(), 
        nn.Linear( self.units_per_hidden,  self.units_per_hidden, bias=True),
        nn.ReLU(),
        nn.Linear( self.units_per_hidden, self.n_outputs, bias=True))
    
        self.target_network = deepcopy(self.q_network)
    
        self.optimizer = torch.optim.Adam(self.q_network.parameters(),
                                          lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()     

    def get_action(self, state, action_space_len):
        """
        This performs epsilon-greedy action selection.
        """
        with torch.no_grad():
            Qp = self.q_network(state) #outputs n values
        Q,A = torch.max(Qp, axis=0) #value, index
        self.r_expect.append(Q.item())
        if torch.rand(1,).item() > self.epsilon:
            return int_to_action(A)
        else: #random action
            A = torch.randint(0,action_space_len,(1,))
            return int_to_action(A)
    
    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_network(state)
        q,_ = torch.max(qp, axis=0)    
        return q
   
    def train_params(self,s,a,rn,sn):
        """
        A version of train() that doesn't rely on the experience buffer.
        Simpler to implement... I think
        
        a is not needed? No, because info on which (a) was chosen is
        encoded in s and q_network. qp in this function refers to the 
        highest quality of the actions that are available.
        """
        if(self.sync_counter == self.target_sync_freq):
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.sync_counter = 0
        
        # predict expected return of current state using main network
        qp = self.q_network(s)
        pred_return, _ = torch.max(qp, axis=0)
        
        # get target return using target network
        q_next = self.get_q_next(sn)
        target_return = rn + self.discount_rate * q_next
        
        loss = self.loss(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        self.sync_counter += 1       
        #return loss.item()

    def update_state_history(self,state):
        """
        A function that manages the queue's quirks.
        i.e. don't add if it's full, don't pop if it's empty
        (these both cause a hang)
        """
        b = torch.tensor(12.)
        if len(self.memory_as_list) <  self.memory_length:
            self.memory_as_list.append(state) #FIFO, removes the oldest.
            torch.cat(self.memory_as_list,out=b)
            return b #one hot format now.
        self.memory_as_list.pop(0) # remove oldest entry
        self.memory_as_list.append(state)# add as new entry
        torch.cat(self.memory_as_list,out=b)
        return b #one hot format now.

    def strategy(self,opponent: Player) -> Action:
        """
        The hook in to the Axelrod tournament.

        Training must happen or be called sometimes from here.
        """
        if len(self.history) == 0:
            self.prev_action = C # Let's just start with cooperation. Hooray optimism!
            return C

        opponent_last_action = action_to_tensor(opponent.history[-1])
        own_last_action = action_to_tensor(self.history[-1])
        this_state = torch.hstack((opponent_last_action,own_last_action))
        
        self.old_state = self.state
        self.state = self.update_state_history(this_state) # this is the input to network.
        reward = self.find_reward(opponent)
        self.r_actual.append(reward)
        
        if len(self.history) < self.memory_length + 1:
            self.prev_action = C # Let's just start with cooperation. Hooray optimism!
            return C        
        
        self.train_params(self.old_state,self.prev_action,reward,self.state)
        
        action = self.get_action(self.state,self.n_outputs)
        self.prev_action = action
        return action
