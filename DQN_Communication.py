# -*- coding: utf-8 -*-
"""
Hacks the Player.clone() and Player.reset() functions
to allow DQN learner to remember results across games.

Increases tournament average score substantially
"""

# Axelrod imports
import axelrod as axl
from axelrod.player import Player
from dqn_learner_intergame_memory import DQN_Learner_Intergame_Memory

import Trust_Box
import Conviction_Box



class CommunicatingPlayer(axl.RiskyQLearner):
    axelrod_no_communication_player = DQN_Learner_Intergame_Memory()
    trust_box = Trust_Box()
    conviction_box = Conviction_Box()

    def __init__(self):
        super.__init__(self)


    def strategy(self, opponent: Player):
        self_action, opponent_action = self.axelrod_no_communication_player.strategy(opponent), opponent.strategy(self)

        # generate intent from actions produced by player and opponent
        self_intent = self.generate_intent(self_action);
        opponent_intent = self.generate_intent(opponent_action);

        #trust RL
        trust = self.trust_box.strategy(opponent, opponent_intent)

        #conviction RL
        action = self.conviction_box.strategy(opponent,trust)

        #Dont forget to modify history of axelrod_no_comm player as the decison might have changed.

        #return action
        pass

    def generate_intent(self):
        pass