# IPD-comms

Testbed: Single match, iterated match, and tournament functionality. NOTE: must choose which of our learners to implement in a given boolean block.

*** 

MLP learner: learns the reward from a state and action set, and then ladder logic to select action. This is a dumb model, doesn't accomplish real RL.

*** 

All q-based methods use epsilon-greedy:

q_learner_n_memory: Implements the tabular q-learning algorithm with a memory n turns long. For example 3 turns memory, the state table has 64 entries (4^3).

dqn_learner: Implements single DQN learner. State contains n turns of history as a list[], but no actual look up associated with it. Has q- and t- networks, but no replay buffer. Does NOT carry over memory from game to game during a tournament.

dqn_learner_intergame_memory: Same as dqn_learner, but overwrote axl.Player.reset() and axl.Player.clone(). The first now dumps expected and actual q values (rewards) for each turn, then resets that particular memory and base class attributes.  ax.Player.clone() now implements some deepcopy() functionality so the q- and t- network are preserved from game to game during a tournament.
