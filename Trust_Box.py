import axelrod as axl
from axelrod.player import Player


#Meant to return action from axelrod, but using to return trust
class Trust_Box(axl.RiskyQLearner):

    def __init__(self):
        super.__init__(self)


    def strategy(self, opponent: Player,opponent_intent):
        #return trust
        pass

