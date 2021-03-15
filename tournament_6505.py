"""

Try to make it so axl.Tournament passes the players themselves not
clones of them.

"""

import axelrod as axl

#Dependencies from base class
import csv
import logging
import os
import warnings
from collections import defaultdict
from multiprocessing import Process, Queue, cpu_count
from tempfile import mkstemp
from typing import List, Optional, Tuple

import axelrod.interaction_utils as iu
import tqdm
from axelrod import DEFAULT_TURNS
from axelrod.action import Action, actions_to_str
from axelrod.player import Player

from axelrod.game import Game
from axelrod.match import Match
from axelrod.match_generator import MatchGenerator
from axelrod.result_set import ResultSet

from communicating_match import Match_6505

class Tournament_6505(axl.Tournament):

    def _play_matches(self, chunk, build_results=True):
            """
            Play matches in a given chunk.
            Parameters
            ----------
            chunk : tuple (index pair, match_parameters, repetitions)
                match_parameters are also a tuple: (turns, game, noise)
            build_results : bool
                whether or not to build a results set
            Returns
            -------
            interactions : dictionary
                Mapping player index pairs to results of matches:
                    (0, 1) -> [(C, D), (D, C),...]
            """
            interactions = defaultdict(list)
            index_pair, match_params, repetitions, seed = chunk
            p1_index, p2_index = index_pair
            if "learner" in self.players[p1_index].name:
                player1 = self.players[p1_index]
            else:
                player1 = self.players[p1_index].clone()
            if "learner" in self.players[p2_index].name:
                player2 = self.players[p2_index]
            else:
                player2 = self.players[p2_index].clone()
            match_params["players"] = (player1, player2)
            match_params["seed"] = seed
            match = Match(**match_params)
            for _ in range(repetitions):
                match.play()
    
                if build_results:
                    results = self._calculate_results(match.result)
                else:
                    results = None
    
                interactions[index_pair].append([match.result, results])
            return interactions
        
class Tournament_Communicating_6505(axl.Tournament):

    def _play_matches(self, chunk, build_results=True):
            """
            Play matches in a given chunk.
            Parameters
            ----------
            chunk : tuple (index pair, match_parameters, repetitions)
                match_parameters are also a tuple: (turns, game, noise)
            build_results : bool
                whether or not to build a results set
            Returns
            -------
            interactions : dictionary
                Mapping player index pairs to results of matches:
                    (0, 1) -> [(C, D), (D, C),...]
            """
            interactions = defaultdict(list)
            index_pair, match_params, repetitions, seed = chunk
            p1_index, p2_index = index_pair
            if self.players[p1_index].name=="M&J DQN learner w/ memory":
                player1 = self.players[p1_index]
            else:
                player1 = self.players[p1_index].clone()
            if self.players[p2_index].name=="M&J DQN learner w/ memory":
                player2 = self.players[p2_index]
            else:
                player2 = self.players[p2_index].clone()
            match_params["players"] = (player1, player2)
            match_params["seed"] = seed
            match = Match_6505(**match_params)
            for _ in range(repetitions):
                match.play()
    
                if build_results:
                    results = self._calculate_results(match.result)
                else:
                    results = None
    
                interactions[index_pair].append([match.result, results])
            return interactions
