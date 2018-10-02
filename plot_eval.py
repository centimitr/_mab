import matplotlib.pyplot as plt


def plot_mab_rewards_per_round(data, nrounds):
    for name, rewards in data.items():
        avg_rewards = []
        rng = range(1, nrounds + 1)
        for n in rng:
            avg_rewards.append(sum(rewards[:n]) / n)
        plt.plot(rng, avg_rewards, label=name)
    plt.xlabel('round')
    plt.ylabel('cumulative reward (average)')
    plt.legend()
    plt.show()


def plot_mab_total_rewards_per_round(data, nrounds, xlim=None, ylim=None):
    for name, rewards in data.items():
        avg_rewards = []
        rng = range(1, nrounds + 1)
        for n in rng:
            avg_rewards.append(sum(rewards[:n]))
        plt.plot(rng, avg_rewards, label='alpha=' + name)
    plt.xlabel('round')
    plt.ylabel('cumulative reward (sum)')
    axes = plt.gca()
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    plt.legend()
    plt.show()
