# Testing zone.

#Axelrod imports
import axelrod as axl
from tournament_6505 import Tournament_6505,Tournament_Communicating_6505
from axelrod.action import Action
C, D = Action.C, Action.D


# M&J's RL strategies
from RL_strategies.q_learner_n_memory import Q_Learner_6505
from RL_strategies.dqn_learner import DQN_Learner
from RL_strategies.dqn_learner_intergame_memory import DQN_Learner_Intergame_Memory

# our stuff
from communicating_match import Match_6505
from communicating_player import Communicating_Player
from trust_box import Trust_Q_Learner,Ned_Stark,Tywin_Lannister
from conviction_box import Conviction_Q_Learner,Michael_Scott,Vizzini
from trained_trust import Trained_Trust_QLearner

import numpy as np
import matplotlib.pyplot as plt
import numpy as np  
import torch

SEED = 1000 # tournament is repeatable if seed is constant.

Q_LEARN_MEMORY_LENGTH = 5
Q_LEARN_EPSILON = 0.4 # random action 50% of the time, decays at rate 0.98.
Q_LEARN_EPSILON_DECAY = 0.98
Q_LEARN_DISCOUNT = 0.9 # high value on next future
Q_LEARN_LEARNING = 0.8 #learn fast


TRUST_Q_LEARN_DISCOUNT = 0.9

#Leave these as 1, breaks trust and not necessary to show a delta hopefully.
TRUST_MEMORY = 1
CONVICTION_MEMORY = 1

#Tournament settings.
TURNS_PER_MATCH = 3000
REPETITIONS = 200
OUTPUT_DIR = "./output/"
TOURNAMENT_RESULTS_FILE = "tournament_results.png"
TOURNAMENT_PAYOFFS_FILE = "tournament_payoffs.png"

def plot_game_result(game,mani_jasper,opponent):
    """
    Does not save the figure generated, could be added if we want that.
    """
    result = np.asarray(game.scores())
    q_score= np.cumsum(result[:,0])
    opp_score= np.cumsum(result[:,1])
    plt.plot(q_score,label=mani_jasper.name);plt.plot(opp_score,label=opponent.name);plt.legend()
    return q_score,opp_score

#Testing area
if __name__ == '__main__':
    
    dqn = DQN_Learner_Intergame_Memory() #just a shell awating commands
    dqn.set_params() #defaults in the function definition.
    dqn.init_net()

    #set up "our" communicating agent.
    ql = Q_Learner_6505()
    ql.set_params(
        Q_LEARN_LEARNING,
        Q_LEARN_DISCOUNT,
        Q_LEARN_EPSILON,
        Q_LEARN_EPSILON_DECAY,
        Q_LEARN_MEMORY_LENGTH)
    trust = Trust_Q_Learner()
    trust.set_params(        
        Q_LEARN_LEARNING,
        0,
        Q_LEARN_EPSILON,
        TRUST_MEMORY) #default values in func
    conviction = Conviction_Q_Learner() 
    conviction.set_params(
        Q_LEARN_LEARNING,
        Q_LEARN_DISCOUNT,
        Q_LEARN_EPSILON,
        CONVICTION_MEMORY) #default values in func

    MJ_Communicator = Communicating_Player()
    MJ_Communicator.set_agents(ql,
                               trust,
                               conviction
                               )

    #base = axl.Alternator() #wins easily    
    #base = axl.TitForTat()     # wins by a hair
    #base = axl.CautiousQLearner() # wins easily
    #base = axl.Cooperator()
    #base = axl.Defector()
    #base = axl.EvolvedANNNoise05()
    base = axl.EvolvedANN()
    
    opponent = Communicating_Player()
    opponent.set_agents(base,
                        Tywin_Lannister(),
                        Michael_Scott()
                        )    
    
    if(False):#Play one match against a given opponent, chosen on next line.
        turns=TURNS_PER_MATCH
        repetitions = 1 # AKA num games
        game = Match_6505([MJ_Communicator,opponent],
                          turns=turns,
                          seed=SEED)
        for _ in range(repetitions):
            game.play()
        print('Done single game, generating result plot.')
        own_score, opp_score = plot_game_result(game,MJ_Communicator,opponent)
        
    #
    # give both our_agent and opponent the Q learner for trust and conviction.
    # Custom tournament using our own reset() and del() methods 
    if(True):#Play a series of matches gainst a series of opponents, with new agent every game.
        turns=TURNS_PER_MATCH
        repetitions = REPETITIONS # AKA num games for each opponent type
        
        base_players = [#axl.Random(),
                    axl.TitForTat(),
                    axl.EvolvedANN(),
                    axl.EvolvedANNNoise05(),
                    axl.WorseAndWorse(),
                    axl.Cooperator(),
                    axl.Defector(),
                    axl.AdaptorLong(),
                    axl.CautiousQLearner()
                    ]
        
        for player in base_players:
            for _ in range(repetitions): #reinstantiate agents learner every game, delete it for certainty.
                o_trust = Tywin_Lannister()
                o_conviction = Michael_Scott() #just stick with base action.
                """
                o_trust = Trust_Q_Learner()
                o_conviction = Conviction_Q_Learner()
                o_trust.set_params(        
                    Q_LEARN_LEARNING,
                    Q_LEARN_DISCOUNT,
                    Q_LEARN_EPSILON,
                    TRUST_MEMORY) #default values in func
                o_conviction.set_params(
                    Q_LEARN_LEARNING,
                    Q_LEARN_DISCOUNT,
                    Q_LEARN_EPSILON,
                    CONVICTION_MEMORY) #default values in func   
                """
                opponent = Communicating_Player()
                opponent.set_agents(player,
                        o_trust,
                        o_conviction
                        )    

                ql = Q_Learner_6505()
                ql.set_params(
                    Q_LEARN_LEARNING,
                    Q_LEARN_DISCOUNT,
                    Q_LEARN_EPSILON,
                    Q_LEARN_EPSILON_DECAY,
                    Q_LEARN_MEMORY_LENGTH)
                #trust = Ned_Stark()
                trust = Trained_Trust_QLearner()
                conviction = Conviction_Q_Learner() 
                conviction.set_params(
                    Q_LEARN_LEARNING,
                    Q_LEARN_DISCOUNT,
                    Q_LEARN_EPSILON,
                    CONVICTION_MEMORY) #default values in func 
                our_agent = Communicating_Player()
                our_agent.set_agents(ql,
                               trust,
                               conviction
                               )       
            
                game = Match_6505([our_agent,opponent],
                                 turns=turns
                                 )
                game.play()
                our_agent.reset()
                opponent.reset()
                del(our_agent)
                del(ql)
                del(trust)
                del(conviction)
                del(opponent)
                del(o_trust)
                del(o_conviction)
        print('Done game(s)')

    #MODIFIED tournament
    if(False): # 
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
                   axl.EvolvedANN(),
                   axl.EvolvedANNNoise05()
                  ]
        
        players.append(dqn)
        players.append(ql)
        
        comm_players = []
        for player in players:
            x=1
            #TODO
            #make a new communicating player and give it target base agent.
        
        tournament = Tournament_Communicating_6505(
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
    
    #Save Trust Learner
    if(False):
        turns=5000
        repetitions = 1 # AKA num games
        player = axl.WorseAndWorse();
        o_trust = Tywin_Lannister()
        o_conviction = Michael_Scott()
        opponent = Communicating_Player()
        opponent.set_agents(player,
                        o_trust,
                        o_conviction
                        )
        game = Match_6505([MJ_Communicator,opponent],
                          turns=turns,
                          seed=SEED)
        for _ in range(repetitions):
            game.play()
        print('Done single game, generating result plot.')
        own_score, opp_score = plot_game_result(game,MJ_Communicator,opponent)
    
    #with tranied trust qlearner
    if(False):
        turns = 5000
        #set up "our" communicating agent.
        ql = Q_Learner_6505()
        ql.set_params(
            Q_LEARN_LEARNING,
            Q_LEARN_DISCOUNT,
            Q_LEARN_EPSILON,
            Q_LEARN_EPSILON_DECAY,
            Q_LEARN_MEMORY_LENGTH)
        trust = Trained_Trust_QLearner()
        conviction = Conviction_Q_Learner() 
        conviction.set_params(
            Q_LEARN_LEARNING,
            Q_LEARN_DISCOUNT,
            Q_LEARN_EPSILON,
            CONVICTION_MEMORY) #default values in func

        MJ_Communicator = Communicating_Player()
        MJ_Communicator.set_agents(ql,
                                   trust,
                                   conviction
                                   )
        game = Match_6505([MJ_Communicator,opponent],
                          turns=turns,
                          seed=SEED)
        game.play()
        print('Done single game, generating result plot.')
        own_score, opp_score = plot_game_result(game,MJ_Communicator,opponent)