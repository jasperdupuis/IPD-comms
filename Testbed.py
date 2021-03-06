#Testing zone.

import axelrod as axl
from tournament_6505 import Tournament_6505
from RL_strategies.q_learner_n_memory import Q_Learner_6505
from RL_strategies.dqn_learner import DQN_Learner
from RL_strategies.dqn_learner_intergame_memory import DQN_Learner_Intergame_Memory


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

    plt.plot(q_score,label=mani_jasper.name );plt.plot(opp_score,label=opponent.name);plt.legend();
def plot_multiple_games_result(game,mani_jasper,opponent):
    """
        Does not save the figure generated, could be added if we want that.
        """
    result = game.scores()
    q_rewards = np.zeros(len(result))
    q_score = np.zeros(len(result))
    opp_rewards = np.zeros(len(result))
    opp_score = np.zeros(len(result))
    index = 0
    for qq, aa in result:
        q_rewards[index] = qq
        opp_rewards[index] = aa
        q_score[index] = sum(q_rewards)
        opp_score[index] = sum(opp_rewards)
        index = index + 1

    plt.plot(q_score,label=(mani_jasper.name, ' D ' , game.players[0].defections , ' C ', game.players[0].cooperations));
    plt.plot(opp_score, 'r--', label=(opponent.name , ' D ' , game.players[1].defections , 'C ', game.players[1].cooperations));
    plt.legend();
    plt.savefig(mani_jasper.name + opponent.name + ".png");
    plt.show();
    plt.clf();


#Testing area
if __name__ == '__main__':
    learning_rate = 0.9
    discount_rate = 0.9
    action_selection_parameter = 0.1
    memory_length = 3
    ql = Q_Learner_6505()
    ql.set_params(learning_rate,discount_rate,action_selection_parameter,memory_length)
    
    tft = axl.TitForTat()
    alt = axl.Alternator()
    rnd = axl.Random() 
    
    
    if(False):#Play one matcha gainst a given opponent, chosen on next line.
        #opponent = axl.Alternator() #wins easily    
        #opponent = axl.TitForTat()     # wins by a hair
        #opponent = axl.CautiousQLearner() # wins easily
        #opponent = axl.Cooperator()
        #opponent = axl.Defector()
        opponent = axl.EvolvedANNNoise05() # Can do well early, but loses over 1000 turns.
        dqn = DQN_Learner_Intergame_Memory() #just a shell awating commands
        dqn.set_params() #defaults in the function definition.
        dqn.init_net()
        #dqn=ql # for q learning testing
    
        turns=2500
        repetitions = 1 # AKA num games
        game = axl.Match([ql,opponent],turns=turns)
        #game.set_seed(5) #same every time for RNGs
        #game.set_seed()
        for _ in range(repetitions):
            game.play()
        print('Done single game, generating result plot.')
        plot_game_result(game,dqn,opponent)
        nets = dqn.list_of_nets
        dqn.write_q_and_rewards()

    if (True):  # Play one matcha gainst a given opponent, chosen on next line.
        # opponent = axl.Alternator() #wins easily
        # opponent = axl.TitForTat()     # wins by a hair
        # opponent = axl.CautiousQLearner() # wins easily
        # opponent = axl.Cooperator()
        # opponent = axl.Defector()
        opponents = [axl.TitForTat(),axl.Alternator(),axl.Cooperator(),ql,axl.EvolvedANNNoise05()]
        #opponent = axl.EvolvedANNNoise05()  # Can do well early, but loses over 1000 turns.
        dqn = DQN_Learner_Intergame_Memory()  # just a shell awating commands
        dqn.set_params()  # defaults in the function definition.
        dqn.init_net()
        # dqn=ql # for q learning testing

        turns = 2500
        repetitions = 1  # AKA num games
        for opponent in opponents:
            game = axl.Match([ql, opponent], turns=turns)
            # game.set_seed(5) #same every time for RNGs
            # game.set_seed()
            for _ in range(repetitions):
                game.play()
            print('Match played between ' + dqn.name + " and " + opponent.name + ". Generating plot!")
            plot_multiple_games_result(game, dqn, opponent)
            # nets = dqn.list_of_nets
            # dqn.write_q_and_rewards()
            dqn.reset()
            dqn.set_params()
            dqn.init_net()

    if(False): # DEFAULT AXL TOURNAMENT
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
        tournament = Tournament_6505(
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
