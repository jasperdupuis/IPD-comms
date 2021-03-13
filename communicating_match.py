from math import ceil, log

import axelrod as axl
import axelrod.interaction_utils as iu
from axelrod import match



class Match_6505(axl.Match):
    """
    A match class for communicating players.
    """
    def simultaneous_play(self, player, coplayer, noise=0):
        """This pits two players against each other."""
        player.generate_base_intent_and_message(coplayer)
        coplayer.generate_base_intent_and_message(player)
        
        s1, s2 = player.strategy(coplayer), coplayer.strategy(player)
        if noise:
            # Note this uses the Match classes random generator, not either
            # player's random generator. A player shouldn't be able to
            # predict the outcome of this noise flip.
            s1 = self._random.random_flip(s1, noise)
            s2 = self._random.random_flip(s2, noise)
        player.update_history(s1, s2)
        coplayer.update_history(s2, s1)
        return s1, s2

