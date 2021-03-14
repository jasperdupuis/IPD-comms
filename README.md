# IPD-comms

Testbed: Single match, iterated match, and tournament functionality. NOTE: must choose which of our learners to implement in a given boolean block.

*** 

MLP learner: learns the reward from a state and action set, and then ladder logic to select action. This is a dumb model, doesn't accomplish real RL.

*** 

All q-based methods use epsilon-greedy:

q_learner_n_memory: Implements the tabular q-learning algorithm with a memory n turns long. For example 3 turns memory, the state table has 64 entries (4^3).

dqn_learner: Implements single DQN learner. State contains n turns of history as a list[], but no actual look up associated with it. Has q- and t- networks, but no replay buffer. Does NOT carry over memory from game to game during a tournament.

dqn_learner_intergame_memory: Same as dqn_learner, but overwrote axl.Player.reset() and axl.Player.clone(). The first now dumps expected and actual q values (rewards) for each turn, then resets that particular memory and base class attributes.  ax.Player.clone() now implements some deepcopy() functionality so the q- and t- network are preserved from game to game during a tournament.

This chart shows learning over a 14 player tournament, you can see evidence of trying to adapt to new players every game.

![Expected reward over 14 player tournament](https://user-images.githubusercontent.com/13178493/109394730-da094900-78fe-11eb-9a7b-2d3532738b20.png)

Here is a chart uploaded 20210514 showing Q Learner performance, averaged over 10000 games of 1000 turns each against each of the shown opponents.
For clarity, ideal strategy against each of the shown agents is given (if known):

AdaptorLong: ??

CautiousQLearner: ?? (depends, it will always D or always C)

Cooperator: always D (5 points)

Defector: always D (1 point)

ANN05: ??

ANN: ??

Random: ??

Tit for Tat: C

Worse and Worse: D (defects with a probability N/1000, where N is number of turns that have passed)



![QL_summary_reward](https://user-images.githubusercontent.com/13178493/111077842-6a37b880-84d1-11eb-93fe-2c2156bbba52.png)

