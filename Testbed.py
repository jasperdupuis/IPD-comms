#Testing scripts.

import axelrod as axl
from axelrod.action import Action, actions_to_str
from axelrod.player import Player
from q_learner_n_memory import Q_Learner_6505

import numpy as np

from collections import OrderedDict
from typing import Dict, Union

Score = Union[int, float]

C, D = Action.C, Action.D

OUTPUT_DIR = "./output/"
TOURNAMENT_RESULTS_FILE = "tournament_results.png"
TOURNAMENT_PAYOFFS_FILE = "tournament_payoffs.png"

#Testing area
if __name__ == '__main__':
    learning_rate = 0.9
    discount_rate = 0.1
    action_selection_parameter = 0.1
    memory_length = 3
    q = Q_Learner_6505()
    q.set_params(learning_rate,discount_rate,action_selection_parameter,memory_length)
    
    tft = axl.TitForTat()
    alt = axl.Alternator()
    rnd = axl.Random() 
    
    if(False):#Play one matcha gainst a given opponent, chosen on next line.
        opponent = axl.Alternator()
        
        turns = 1000
        q = Q_Learner_6505()
        q.set_params(learning_rate,discount_rate,action_selection_parameter,memory_length)
        game = axl.Match([q,opponent],turns=turns)
        game.set_seed(5) #same every time for RNGs
        for _ in range(turns):
            game.play()
        
    if(False): # play a set of games versus tit for tat
        #Check to see if this really beats tit for tat, no tournament: 
        games = []
        num_games = 20
        turns = 1000
        for _ in range(num_games):
            q = Q_Learner_6505()
            q.set_params(learning_rate,discount_rate,action_selection_parameter,memory_length)
            tft = axl.TitForTat()
            
            game = axl.Match([q,tft],turns=turns)
            game.set_seed(5) #same every time for RNGs
            for _ in range(turns):
                game.play()
            games.append(game)

    if(False):
        #Check to see how this does in a tournament
        TURNS_PER_MATCH = 200
        REPETITIONS = 10
        q = Q_Learner_6505()
        q.set_params(learning_rate,discount_rate,action_selection_parameter,memory_length)
       
        players = [q,
                   axl.TitForTat(),
                   axl.Alternator(),
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

'''
result = game.scores()
q_rewards = np.zeros(len(result))
q_score = np.zeros(len(result))
opp_rewards = np.zeros(len(result))
opp_score = np.zeros(len(result))
index = 0
for qq,aa in result:
    q_rewards[index] = qq
    opp_rewards[index] =  aa
    q_score[index] = sum(q_rewards)
    opp_score[index] = sum(opp_rewards)
    index = index + 1
    
    
import matplotlib.pyplot as plt
import numpy as np  

plt.plot(q_score,label='Q_Learner fixed');plt.plot(opp_score,label='Alternator');plt.legend() 
'''