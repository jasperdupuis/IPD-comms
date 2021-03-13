import axelrod as axl
from axelrod.player import Player
from axelrod.action import Action, actions_to_str


from collections import OrderedDict
from typing import Dict, Union

C, D = Action.C, Action.D
Score = Union[int,float]

class Conviction_Box(axl.Player):

    def __init__(self):
        super().__init__()
        self.list_decisions = []

    def strategy(self, opponent: Player,trust):
        #return action
        pass
    
    def append_decision(self,decision):
        self.list_decisions.append(decision)
    
    def strategy(self,
                 action_base,
                 assessment,
                 prev_nme_action,
                 reward):
        self.append_decision(action_base)
        return action_base
    
class Michael_Scott(Conviction_Box):
    """
    "You miss all the shots you don't take" 
        - Wayne Gretzky    
            -Michael Scott
    
    Always sticks with their gut, i.e. what was decided previously
    
    """
    
    def strategy(self,
                 action_base,
                 assessment,
                 prev_nme_action,
                 reward):
        self.append_decision(action_base)
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
    
    Always second guesses, i.e. acts on new intelligence 
    (functionally this means always defect, since if we are "sure" the opponent
    is doing C or D, the best immediate action is D)
    
    Since second guessing in this little game means always 
    "knowing" what is coming, this means always defect.
    
    """
    
    def strategy(self,
                 action_base,
                 assessment,
                 prev_nme_action,
                 reward):
        self.append_decision(D)
        return D
    
    
class Conviction_Q_Learner(Conviction_Box,axl.RiskyQLearner):
    """
    Re implement the Q Learner to use the variable length intent vector
    
    """
    name = "Conviction Q learner"
    
        #Default values, change with set_params
    learning_rate = 0.5 #learns fast
    discount_rate = 0.5 # cares about the future
    action_selection_parameter = 0.1 #bias towards exploration to visit new states
    memory_length = 1 # number of turns recalled in state
        
    reward = 0
    decision = C
    decision_prev = C
    
    list_decisions = []

    def __init__(self) -> None:
        """Initialises the player by picking a random strategy."""

        super().__init__()

        # Set this explicitly, since the constructor of super will not pick it up
        # for any subclasses that do not override methods using random calls.
        self.classifier["stochastic"] = True

        self.prev_action = C # type: Action
        self.original_prev_action = C# type: Action
        self.score = 0
        self.Qs = OrderedDict({"": OrderedDict(zip([C, D], [0, 0]))})
        self.Vs = OrderedDict({"": 0})
        self.prev_state = ""
        
        self.list_decisions = []
    
    def set_params(self,
                   learning,
                   discount,
                   select_param,
                   memory_length=3):
        self.learning_rate = learning
        self.discount_rate = discount
        self.action_selection_parameter = select_param
        self.memory_length = memory_length        
            
    def find_state(self,action_base,assessment) -> str:
        """
        Finds the my_state (the opponents last n moves +
        its previous proportion of playing C) as a hashable state
        """
        state_list = [action_base,assessment]
        action_str = actions_to_str(state_list)
        state_str = self.prev_state + action_str
        state_str = state_str[-(2*self.memory_length):]
        return state_str
    
    def find_reward(self,
                    prev_nme_action):
        return self.payoff_matrix[self.prev_action][prev_nme_action]
        
    
    def strategy(self,
                 action_base,
                 assessment,
                 reward,
                 prev_nme_action):
        """
        Runs a qlearn algorithm while the tournament is running.
        Reimplement the base class strategy to work with trust communication
        """
        if len(self.history) == 0:
            self.prev_action = C
            self.original_prev_action = C
        
        state = self.find_state(action_base,assessment) #recursive
        
        if state not in self.Qs:
            self.Qs[state] = OrderedDict(zip([C, D], [0, 0]))
            self.Vs[state] = 0
        self.perform_q_learning(self.prev_state, state, self.prev_action, reward)
        if state not in self.Qs:
            action = random_choice()
        else:
            action = self.select_action(state)
        self.prev_state = state
        self.prev_action = action
        self.append_decision(action)
        return action
    
    
        action = self.strategy(self.action_base,
                                 self.assessment,
                                 self.reward,
                                 prev_nme_action)                
        return action