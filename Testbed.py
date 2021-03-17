#Testing zone.

import axelrod as axl
from tournament_6505 import Tournament_6505
from RL_strategies.q_learner_n_memory import Q_Learner_6505
from RL_strategies.dqn_learner import DQN_Learner
from RL_strategies.dqn_learner_intergame_memory import DQN_Learner_Intergame_Memory
from benchmark_score import Benchmark_Score

import numpy as np
import matplotlib.pyplot as plt
import numpy as np  

OUTPUT_DIR = "./output/"
TOURNAMENT_RESULTS_FILE = "tournament_results.png"
TOURNAMENT_PAYOFFS_FILE = "tournament_payoffs.png"

SEED = 1000 # tournament is repeatable if seed is constant.

Q_LEARN_MEMORY_LENGTH = 5
Q_LEARN_EPSILON = 0.05 # random action 10% of the time
Q_LEARN_DISCOUNT = 0.4 #high value on next future
Q_LEARN_LEARNING = 0.8 #learn fast

#Tournament settings.
TURNS_PER_MATCH = 2000
REPETITIONS = 1

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
    learning_rate = Q_LEARN_LEARNING
    discount_rate = Q_LEARN_DISCOUNT
    action_selection_parameter = Q_LEARN_EPSILON
    memory_length = Q_LEARN_MEMORY_LENGTH
    ql = Q_Learner_6505()
    ql.set_params(learning_rate,discount_rate,action_selection_parameter,memory_length)
    
    dqn = DQN_Learner_Intergame_Memory() #just a shell awating commands
    dqn.set_params() #defaults in the function definition.
    dqn.init_net()
    
    tft = axl.TitForTat()
    alt = axl.Alternator()
    rnd = axl.Random() 
    
    
    if(True):#Play a single of match gainst a given opponent
        #opponent = axl.Alternator() #wins easily    
        #opponent = axl.TitForTat()     # wins by a hair
        #opponent = axl.CautiousQLearner() # wins easily
        #opponent = axl.Cooperator()
        #opponent = axl.Defector()
        opponent = axl.EvolvedANNNoise05() # Can do well early, but loses over 1000 turns.
        our_agent = ql
        
        benchmark = Benchmark_Score()
        
        turns=10000
        repetitions = 1 # AKA num games
        benchmark.set_benchmark_score(turns) #set benchmark score given number of turns
        for _ in range(repetitions):
            game = axl.Match([our_agent,opponent],
                             turns=turns,
                             seed = SEED)
            game.play()
            benchmark.add_player_score(sum(np.asarray(game.scores())[int(turns/10):,0]))
        print('Done game, plotting result.')
        print('Player reached  %.2f of benchmark' % (benchmark.get_benchmark_percentage()))
        plot_game_result(game,our_agent,opponent)
    
    if(False):#Play a series of matches gainst a series of opponents, with new agent every game.
        turns=1000
        repetitions = 10000 # AKA num games for each opponent type
        game = axl.Match([ql,opponent],turns=turns)

        players = [axl.Random(),
                           axl.TitForTat(),
                           axl.EvolvedANN(),
                           axl.EvolvedANNNoise05(),
                           axl.WorseAndWorse(),
                           axl.Cooperator(),
                           axl.Defector(),
                           axl.Grudger(),
                           axl.AdaptorLong(),
                           axl.CautiousQLearner()]
        
        for opponent in players:
            for _ in range(repetitions): #reinstantiate our own learner every game, delete it for certainty.
                ql = Q_Learner_6505()
                ql.set_params(learning_rate,discount_rate,action_selection_parameter,memory_length)
                our_agent = ql
                game = axl.Match([our_agent,opponent],
                                 turns=turns
                                 )
                game.play()
                ql.reset()
                opponent.reset()
                del(our_agent)
                del(ql)
        print('Done game(s)')

    if(False): # DEFAULT AXL TOURNAMENT
        #Check to see how this does in a tournament
        
        dqn = DQN_Learner_Intergame_Memory() #just a shell awating commands
        dqn.set_params() #defaults in constructor.
        dqn.init_net()
        
        players = [axl.Random(),
                   axl.TitForTat(),
                   axl.EvolvedANN(),
                   axl.EvolvedANNNoise05(),
                   axl.WorseAndWorse(),
                   axl.Cooperator(),
                   axl.Defector(),
                   axl.Grudger(),
                   axl.AdaptorBrief(),
                   axl.AdaptorLong(),
                   axl.Alternator(),
                   axl.CautiousQLearner(),
                   dqn,
                   ql
                  ]
        tournament = Tournament_6505(
                players=players,
                turns=TURNS_PER_MATCH,
                repetitions=REPETITIONS,
                seed= SEED
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
