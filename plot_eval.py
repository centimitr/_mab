from mab import MAB
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def plot_mab_rewards_per_round(data, nrounds):
    for name, rewards in data.items():
        avg_rewards = []
        rng = range(1, nrounds + 1)
        for n in rng:
            avg_rewards.append(sum(rewards[:n]) / n)
        plt.plot(rng, avg_rewards, label=name)
    plt.xlabel('round')
    plt.ylabel('cumulative reward')
    plt.legend()
    plt.show()
