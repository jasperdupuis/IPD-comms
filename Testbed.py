#Testing zone.

import axelrod as axl
from q_learner_n_memory import Q_Learner_6505
from dqn_learner import DQN_Learner
from dqn_learner_intergame_memory import DQN_Learner_Intergame_Memory


import numpy as np
import matplotlib.pyplot as plt
import numpy as np  

OUTPUT_DIR = "./output/"
TOURNAMENT_RESULTS_FILE = "tournament_results.png"
TOURNAMENT_PAYOFFS_FILE = "tournament_payoffs.png"

def plot_game_result(game,mani_jasper,opponent):
    """
    Does not save the figure generated, could be added if we want that.
    """
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
    plt.plot(q_score,label=mani_jasper.name);plt.plot(opp_score,label=opponent.name);plt.legend()
    

#Testing area
if __name__ == '__main__':
    learning_rate = 0.9
    discount_rate = 0.1
    action_selection_parameter = 0.1
    memory_length = 10
    ql = Q_Learner_6505()
    ql.set_params(learning_rate,discount_rate,action_selection_parameter,memory_length)
    
    tft = axl.TitForTat()
    alt = axl.Alternator()
    rnd = axl.Random() 
    
    if(False):#Play one matcha gainst a given opponent, chosen on next line.
        #opponent = axl.Alternator() #wins easily    
        opponent = axl.TitForTat()     # wins by a hair
        #opponent = axl.CautiousQLearner() # wins easily
        #opponent = axl.EvolvedANNNoise05() # Can do well early, but loses over 1000 turns.
        dqn = DQN_Learner_Intergame_Memory() #just a shell awating commands
        dqn.set_params() #defaults in constructor.
        dqn.init_net()
    
        turns=10000
        repetitions = 1 # AKA num games
        game = axl.Match([dqn,opponent],turns=turns)
        #game.set_seed(5) #same every time for RNGs
        #game.set_seed()
        for _ in range(repetitions):
            game.play()
        print('Done single game, generating result plot.')
        plot_game_result(game,dqn,opponent)
        dqn.reset()
        
    if(False): # play a set of games versus tit for tat
        #Check to see if this really beats tit for tat, no tournament: 
        games = []
        repetitions = 20 #AKA num games
        turns = 100
        for _ in range(num_games):
            tft = axl.TitForTat()
            
            for _ in range(repetitions):
                dqn = DQN_Learner(memory_length,n_outputs)
                dqn.set_params()#default arguments are defined for function.
                dqn.init_net()
                game = axl.Match([q,tft],turns=turns)
                game.set_seed(5) #same every time for RNGs
                game.play()
                games.append(game)

    if(True):
        #Check to see how this does in a tournament
        TURNS_PER_MATCH = 1000
        REPETITIONS = 5
        
        dqn = DQN_Learner_Intergame_Memory() #just a shell awating commands
        dqn.set_params() #defaults in constructor.
        dqn.init_net()
        
        players = [axl.WorseAndWorse(),
                   axl.Cooperator(),
                   axl.Defector(),
                   axl.Grudger(),
                   axl.AdaptorBrief(),
                   axl.AdaptorLong(),
                   axl.Random(),
                   axl.TitForTat(),
                   axl.Alternator(),
                   axl.CautiousQLearner(),
                   dqn,
                   ql,
                   axl.EvolvedANN(),
                   axl.EvolvedANNNoise05()
                  ]
        tournament = axl.Tournament(
                players=players,
                turns=TURNS_PER_MATCH,
                repetitions=REPETITIONS
                )
        
        # tournament = axl.Tournament(
        #         players=players,
        #         turns=TURNS_PER_MATCH,
        #         repetitions=REPETITIONS
        #         )
        results = tournament.play()
        winners = results.ranked_names
        
        results_plot = axl.Plot(results)
        plot = results_plot.boxplot()
        plot.show()
        plot.savefig(OUTPUT_DIR+TOURNAMENT_RESULTS_FILE)
        
        plot = results_plot.payoff()
        plot.show()
        plot.savefig(OUTPUT_DIR+TOURNAMENT_PAYOFFS_FILE)
