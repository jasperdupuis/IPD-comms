# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:23:15 2021

@author: Jasper

"""


import numpy as np
import torch
import axelrod as axl
from axelrod.action import Action
from axelrod.player import Player

from typing import Dict, Union


OUTPUT_DIR = "./output/"
TOURNAMENT_RESULTS_FILE = "tournament_results.png"
TOURNAMENT_PAYOFFS_FILE = "tournament_payoffs.png"


Score = Union[int,float]
C, D = Action.C, Action.D


def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
            
class MLP_Learner(Player):
    """
    For N input features, trains a fully connected MLP with ReLU
    of M hidden layers of size 16 while playing.
    Outputs a single action, C or D.
    """
    
    name = "Mani and Jasper's first model"
    
    #storage for 10000 long games, don't go that long.
    opp_move = np.zeros(10000)
    own_move = np.zeros(10000)
    rewards = np.zeros(10000)
    outputs = np.zeros(10000)
    
    nnet = 'not yet set' # default string in case it breaks
    units_per_hidden = 16
    
    
    loss = 'not yet set'    # default string in case it breaks
    trainer = 'not yet set' # default string in case it breaks
    
    def action_to_int(self,action):
        """
        Maps C or D actions to an int.
        """
        if action == C: return torch.tensor(1.)
        if action == D: return torch.tensor(0.)
        
    def int_to_action(self,integer): #not used right now
        if integer>0.9: return C
        if integer<0.9: return D
        
    def state_to_one_hot(reward): #not used right now
        """
        Maps the IPD(n-1) rewaerd to a one-hot vector.
        """
        if reward == 0: return torch.tensor([0,0,0,1])
        if reward == 1: return torch.tensor([0,0,1,0])
        if reward == 2: return torch.tensor([0,1,0,0])
        if reward == 3: return torch.tensor([1,0,0,0])
        
    
    def __init__(self, 
        learning_rate,
        num_input_features: int,
        num_hidden: int) -> None:
        """
        Just a constructor, also instantiates the MLP according to arguments.
        """
        Player.__init__(self)
        self.learning_rate = learning_rate
        self.num_input_features = num_input_features
        self.num_hidden = num_hidden
        self.init_net()
    
    def init_net(self):
        layers = []

        layers.append(torch.nn.Linear(self.num_input_features, self.units_per_hidden))
        layers.append(torch.nn.ReLU())

        for count in range(self.num_hidden):
            layers.append(torch.nn.Linear(self.units_per_hidden, self.units_per_hidden))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(self.units_per_hidden,1))#self.num_input_features))
        
        self.nnet = torch.nn.Sequential(*layers)

        self.nnet.apply(init_weights);
        self.loss = torch.nn.MSELoss()
        self.trainer = torch.optim.SGD(self.nnet.parameters(), lr=self.learning_rate)


    def train_net(self,
                  last_action_tensor,
                  reward):
        """
        Given the two actions (as ints) and reward for n-1, train the network.
        last_action_tensor = torch.tensor(own,opp)
        """
        l = self.loss(self.nnet(last_action_tensor),torch.tensor(reward,dtype=torch.float))
        self.trainer.zero_grad()
        l.backward()
        self.trainer.step()
        
        
    def compute_net(self,
                    last_action_tensor
                    ):
        """
        last_action_tensor arguments must be as ints.
        last_action_tensor = torch.tensor(own,opp)
        """
        output = self.nnet(last_action_tensor)
        return output
        


    def strategy(self,opponent: Player) -> Action:
        """
        The tournament calls this function to play a match.
        """
        if len(self.history) == 0:
            self.prev_action = C # Let's just start with cooperation. Hooray optimism!
            return C
        opponent_last_action = self.action_to_int(opponent.history[-1])
        own_last_action = self.action_to_int(self.history[-1])
        X = torch.tensor([own_last_action,opponent_last_action],dtype=torch.float)
        reward = self.find_reward(opponent)
        self.train_net(X,reward)
        action_as_tensor = self.compute_net(X)
        
        #record for posterity:
        self.own_move[len(self.history)-1] = self.action_to_int(self.history[-1])
        self.opp_move[len(self.history)-1] = self.action_to_int(opponent.history[-1])
        self.rewards[len(self.history)-1] = reward
        self.outputs[len(self.history)-1] = action_as_tensor
        
        
        #Ladder logic  I guess
        if action_as_tensor[0]>2.5: 
            self.prev_action = D
            return D
        elif action_as_tensor[0]>1.5: 
            self.prev_action = C
            return C
        elif action_as_tensor[0]>0.5: 
            self.prev_action = D
            return D
        else: 
            self.prev_action = C
            return C



        self.prev_action=D
        return D
        
            
        
    def find_reward(self, opponent: Player) -> Dict[Action, Dict[Action, Score]]:
        """
        Finds the reward gained on the last iteration
        """

        if len(opponent.history) == 0:
            opp_prev_action = self._random.random_choice()
        else:
            opp_prev_action = opponent.history[-1]
        return self.payoff_matrix[self.prev_action][opp_prev_action]
        
    
    def receive_match_attributes(self):
        """
        The class Match calls this once per player at the start of each match.
        Could include other information (e.g. game length)
        """
        (R, P, S, T) = self.match_attributes["game"].RPST()
        self.payoff_matrix = {C: {C: R, D: S}, D: {C: T, D: P}}


            
#Testing area
if __name__ == '__main__':
    lr = 0.3
    num_in = 2
    num_hid = 1
    mlp = MLP_Learner(lr,num_in,num_hid)
    tft = axl.TitForTat()
    alt = axl.Alternator()
    rnd = axl.Random() 
    
    if(False):
        #Check to see if this really beats tit for tat, no tournament: 
        games = []
        num_games = 20
        turns = 200
        for _ in range(num_games):
            mlp = MLP_Learner(lr,num_in,num_hid)
            tft = axl.TitForTat()
            
            game = axl.Match([mlp,tft],turns=turns)
            game.set_seed(5) #same every time for RNGs
            for _ in range(turns):
                game.play()
                games.append(game)
    if(False):
        #Check to see how this does in a tournament
        TURNS_PER_MATCH = 200
        REPETITIONS = 10
        players = [MLP_Learner(lr,num_in,num_hid),
                   axl.TitForTat(),
                   axl.Alternator(),
                   axl.Random()
                   ]
        tournament = axl.Tournament(
                players=players,
                turns=TURNS_PER_MATCH,
                repetitions=REPETITIONS
                )
        results = tournament.play()
        winners = results.ranked_names
        
        results_plot = axl.Plot(results)
        plot = results_plot.boxplot()
        plot.show()
        plot.savefig(OUTPUT_DIR+TOURNAMENT_RESULTS_FILE)
        
        plot = results_plot.payoff()
        plot.show()
        plot.savefig(OUTPUT_DIR+TOURNAMENT_PAYOFFS_FILE)

