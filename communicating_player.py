"""
A class that enables communication between opponents.
"""

#Axelrod imports
import axelrod as axl
from axelrod.action import Action, actions_to_str
from axelrod.player import Player
from typing import Dict, Union

import numpy as np
import torch
from torch import nn
from copy import deepcopy, copy


Class Communicating_Player(axl.Player):
    """
    This class enables communication between two players,
    and also containerizes up to three RL agents, with names in brackets:
        1) base strategy from Axelrod    ("base")
        2) "Message" assessing block     ("trust")
        3) Comparison of results from (1) and (2) ("conviction")
    """
    
    generator = 'not yet set'
    
    base = 'not yet set'
    trust = 0. 
    conviction = 1.
    
    action_base = C
    
    intent_sent = torch.zeros(10)
    intent_received = torch.zeros(10)
    intent_sent_prev = torch.zeros(10)
    intent_received_prev = torch.zeros(10)

    assessment = C
    assessment_prev = C
    
    decision = C
    decision_prev = C
    
    DECEIVE_MEAN = torch.tensor([.4])
    DECEIVE_STD = torch.tensor([.1])
    TRUTH_MEAN = torch.tensor([0.8])
    TRUTH_STD = torch.tensor([0.1])
    
    def __init__(self,seed)->None:
        """
        Doesn't do anything other than call the base and set a generator to a passed seed.
        """
        Player.__init__(self)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def set_base_agent(self,agent):
        self.base=deepcopy(agent)

    def set_trust_agent(self,agent):
        self.trust=deepcopy(agent)

    def set_conviction_agent(self,agent):
        self.conviction=deepcopy(agent)

    def generate_message(self):
        """
        self.action_base must be available before calling this.
        """
        intent = torch.zeros(10)
        rand = 0
        if self.action_base == C:
            rand = torch.normal(mean=self.TRUTH_MEAN,std=self.TRUTH_STD,generator = self.generator)
        else #self.action_base == D:
            rand = torch.normal(mean=self.DECEIVE_MEAN,std=self.DECEIVE_STD,generator = self.generator)
        if rand < 0.1: intent[0] = 1
        elif rand < 0.2: intent[1] = 1
        elif rand < 0.3: intent[2] = 1
        elif rand < 0.4: intent[3] = 1
        elif rand < 0.5: intent[4] = 1
        elif rand < 0.6: intent[5] = 1
        elif rand < 0.7: intent[6] = 1
        elif rand < 0.8: intent[7] = 1
        elif rand < 0.9: intent[8] = 1
        elif rand < 1.0: intent[9] = 1
        return intent 

    def generate_base_intent_and_message(self,opponent:Player) -> str:
        """
        Generate naive intent for upcoming round, and generate
        a message one-hot-vector based on that intent.
        """
        self.action_base = self.base.strategy(opponent)
        self.intent_sent_prev = self.intent_sent
        self.intent_sent = self.generate_message(self.action_base)
        return self.message_own
    
    def assess_received_intent(self,prev_nme_action):
        self.trust.strategy(self.intent_received,
                            self.intent_received_prev,
                            self.assessment_prev,
                            prev_nme_action)
        return C
    
    def decide_based_on_new_intel(self,assessment):
        self.conviction.strategy(self.action_base,
                                 assessment,
                                 prev_nme_action)
        return C
    
    def strategy(self,
                 opponent:Player,
                 message=torch.zeros(10))->Action:
        """
        given message and best action determined from self.generate_base_intent_and_message,
        determine best action based on Trust and Conviction.
        """
        # assess perceived intent message in opponent.sent_message
        self.intent_received_prev = self.intent_received
        self.intent_received = opponent.intent_sent
        self.assessment_prev = self.assessment
        self.assessment = self.assess_received_intent(opponent.history[-1])

        # receive assessment and decide to stay with self.base_Action
        # OR change it to the other action.        
        self.old_decision = self.decision
        self.decision = self.decide_based_on_new_intel(assessment,
                                                       opponent.history[-1])
        return self.decision
    
    
    
    
    
