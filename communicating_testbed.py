#Testing zone.

import axelrod as axl
from tournament_6505 import Tournament_6505,Tournament_Communicating_6505

# M&J's RL strategies
from RL_strategies.q_learner_n_memory import Q_Learner_6505
from RL_strategies.dqn_learner import DQN_Learner
from RL_strategies.dqn_learner_intergame_memory import DQN_Learner_Intergame_Memory

# our stuff
from communicating_match import Match_6505
from communicating_player import Communicating_Player
from trust_box import Trust_Q_Learner


import numpy as np
import matplotlib.pyplot as plt
import numpy as np  
import torch

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

"""
intent_rx = MJ_Communicator.list_intent_received #what was received on turn n-1 (one hot vector)
intent_tx = MJ_Communicator.list_intent_sent #what was sent on turn n-1 (one hot vector)
intent_assess = MJ_Communicator.list_intent_assessment #what was assessed value of intent on turn n-1 (C or D)
intent_true = MJ_Communicator.list_intent_true  #what hte opponent did on turn n-1 (C or D)

result = []
for index in range(10):
    specific_message= torch.zeros(10)
    specific_message[index] = 1
    num_C_true = 0
    num_D_true = 0
    num_C_assess = 0
    num_D_assess = 0
    
    for rx,assess,true in zip(intent_rx,intent_assess,intent_true):
        if torch.all(torch.eq(rx,specific_message)): #checks if all values are equal
            if true == C: num_C_true += 1
            if true == D: num_D_true += 1
            if assess == C: num_C_assess += 1
            if assess == D: num_D_assess += 1
    result.append(np.array([num_C_true,num_C_assess,num_D_true,num_D_assess]))
"""      

#Testing area
if __name__ == '__main__':
    learning_rate = 0.9
    discount_rate = 0.1
    action_selection_parameter = 0.1
    memory_length = 3
    
    MJ_Communicator = Communicating_Player()
    
    dqn = DQN_Learner_Intergame_Memory() #just a shell awating commands
    dqn.set_params() #defaults in the function definition.
    dqn.init_net()
    
    ql = Q_Learner_6505()
    ql.set_params(learning_rate,discount_rate,action_selection_parameter,memory_length)

    trust = Trust_Q_Learner()

    MJ_Communicator.set_base_agent(ql)
    MJ_Communicator.set_trust_agent(trust)
    
    MJ_Communicator.name = 'M&J Communicator Q-Learn'
    
    opponent = Communicating_Player()    
    
    if(True):#Play one matcha gainst a given opponent, chosen on next line.
        base = axl.Alternator() #wins easily    
        #base = axl.TitForTat()     # wins by a hair
        #base = axl.CautiousQLearner() # wins easily
        #base = axl.Cooperator()
        #base = axl.Defector()
        #base = axl.EvolvedANNNoise05() # Can do well early, but loses over 1000 turns.
        #
        opponent.set_base_agent(base)
        opponent.name='M&J Communicator Alternator'
    
        turns=50
        repetitions = 1 # AKA num games
        game = Match_6505([MJ_Communicator,opponent],turns=turns)
        #game.set_seed(5) #same every time for RNGs
        #game.set_seed()
        for _ in range(repetitions):
            game.play()
        print('Done single game, generating result plot.')
        plot_game_result(game,MJ_Communicator,opponent)
        nets = dqn.list_of_nets
        dqn.write_q_and_rewards()

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
