from lin_ucb import LinUCB
import numpy as np

from offline_eval import offlineEvaluate
from plot_eval import plot_mab_total_rewards_per_round
from ucb import UCB

data = np.genfromtxt("data/dataset.txt", delimiter=" ")
arms = data[:, 0].astype(np.int64)
rewards = data[:, 1]
contexts = data[:, 2:]

nrounds = 800
rewards_dict = dict()
# alpha = 1 + sqrt(ln(2/δ)/2)
# so alpha >= 1
for alpha in np.arange(1, 20, 1):
    print(alpha)
    mab = LinUCB(10, 10, alpha)
    rewards_dict[str(alpha)] = offlineEvaluate(mab, arms, rewards, contexts, nrounds)

plot_mab_total_rewards_per_round(rewards_dict, nrounds, xlim=[770, 800], ylim=[205, 220])
# plot_mab_total_rewards_per_round(rewards_dict, nrounds)
