import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from eps_greedy import EpsGreedy
from grid_search import linucb_mean_rewards_by_alpha
from kernel_ucb import KernelUCB
from lin_ucb import LinUCB
from offline_eval import offlineEvaluate
from plot_eval import plot_mab_rewards_per_round
from ucb import UCB

data = np.genfromtxt("data/dataset.txt", delimiter=" ")
arms = data[:, 0].astype(np.int64)
rewards = data[:, 1]
contexts = data[:, 2:]

# P2
mab = EpsGreedy(10, 0.05)
rewards_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('EpsGreedy average reward', np.mean(rewards_EpsGreedy))

mab = UCB(10, 1.0)
rewards_UCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('UCB average reward', np.mean(rewards_UCB))

# P3
mab = LinUCB(10, 10, 1.0)
rewards_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('LinUCB average reward', np.mean(rewards_LinUCB))

# P4(a)
plot_data = {
    'EpsGreedy': rewards_EpsGreedy,
    'UCB': rewards_UCB,
    'LinUCB': rewards_LinUCB
}
plot_mab_rewards_per_round(plot_data, 800)

# P4(b)
nrounds = 800
kw = dict(
    start=1, end=50, step=0.5,
    arms=arms, rewards=rewards, contexts=contexts, nrounds=nrounds
)
linucb_mean_rewards_by_alpha(**kw)

kw.step = 0.1
linucb_mean_rewards_by_alpha(**kw)

# P5
# mab = KernelUCB(10, 10, 0.1, 0.1, rbf_kernel)
# rewards_KernelUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
# plot_data['KernelUCB'] = rewards_KernelUCB
# plot_mab_rewards_per_round(plot_data, 800)
