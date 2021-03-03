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

import torch

class CommunicatingPlayer(axl.Player):
    axelrod_no_communication_player = DQN_Learner_Intergame_Memory()
    trust_box = Trust_Box()
    conviction_box = Conviction_Box()

    def __init__(self):
        super.__init__(self)

    def strategy(self,opponent:Player):
        return self.axelrod_no_communication_player.strategy(opponent)


    def strategy(self, opponent: Player,player_intent,coplayer_intent):

        # generate intent from actions produced by player and opponent
        self_message = self.get_message(player_intent)
        opponent_message = self.get_message(coplayer_intent)

        #trust RL
        trust = self.trust_box.strategy(opponent, opponent_message)

        #conviction RL
        action = self.conviction_box.strategy(opponent,trust)

        #self.action = action
        #return action
        pass

    def get_message(self,intent):
        if intent == 'C':
            message = torch.normal(0.8,0.1)
        if intent == 'D':
            message = torch.normal(0.6,0.1)
        else:
            raise Exception("Sorry, intent can't be other than C/D .")
        return message