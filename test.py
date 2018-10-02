import numpy as np

from eps_greedy import EpsGreedy
from lin_ucb import LinUCB
from offline_eval import offlineEvaluate
from ucb import UCB

# P2

data = np.genfromtxt("data/dataset.txt", delimiter=" ")
arms = data[:, 0].astype(np.int64)
rewards = data[:, 1]
contexts = data[:, 2:]

mab = EpsGreedy(10, 0.05)
results_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('EpsGreedy average reward', np.mean(results_EpsGreedy))

mab = UCB(10, 1.0)
results_UCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('UCB average reward', np.mean(results_UCB))

# P3
mab = LinUCB(10, 10, 1.0)
results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('LinUCB average reward', np.mean(results_LinUCB))
