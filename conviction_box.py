import axelrod as axl
from axelrod.player import Player


class Conviction_Box(axl.Player):

    def __init__(self):
        super.__init__(self)


    def strategy(self, opponent: Player,trust):
        #return action
        pass
    
class Michael_Scott(Conviction_Box):
    """
    "You miss all the shots you don't take" - Wayne Gretzky
    
    
            -Michael Scott
    
    Always sticks with their gut, i.e. what was decided previously
    
    """
    
    def strategy(self,
                 action_base,
                 assessment,
                 prev_nme_action):
        return action_base
    
    
    
class Vizzini(Conviction_Box):
    """
    "But it's so simple. All I have to do is divine from what I know of you: 
    are you the sort of man who would put the poison into his own goblet or his enemy's? 
    Now, a clever man would put the poison into his own goblet, because he would know 
    that only a great fool would reach for what he was given. I am not a great fool,
    so I can clearly not choose the wine in front of you. But you must have known I was
    not a great fool, you would have counted on it, so I can clearly not choose the wine
    in front of me."
    
    Always second guesses, i.e. acts on new intelligence.
    
    Since second guessing in this little game means always 
    "knowing" what is coming, this means always defect.
    
    """
    
    def strategy(self,
                 action_base,
                 assessment,
                 prev_nme_action):
        return D
