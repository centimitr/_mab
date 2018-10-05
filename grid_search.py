import numpy as np
import matplotlib.pyplot as plt

from lin_ucb import LinUCB
from offline_eval import offlineEvaluate


# data = np.genfromtxt("data/dataset.txt", delimiter=" ")
# arms = data[:, 0].astype(np.int64)
# rewards = data[:, 1]
# contexts = data[:, 2:]
#
# nrounds = 800


# alpha = 1 + sqrt(ln(2/δ)/2)
# so alpha >= 1

def linucb_mean_rewards_by_alpha(start, end, step, arms, rewards, contexts, nrounds):
    mean_rewards_dict = []

    rng = np.arange(start, end, step)
    for alpha in rng:
        # print(alpha)
        mab = LinUCB(10, 10, alpha)
        result_rewards = offlineEvaluate(mab, arms, rewards, contexts, nrounds)
        mean_rewards_dict.append(np.mean(result_rewards))

    # max points
    y_max = np.max(mean_rewards_dict)
    x_max = rng[np.argmax(mean_rewards_dict)]
    print("max:", y_max, "at", x_max)

    plt.plot(rng, mean_rewards_dict)

    # annotate the highest value
    ax = plt.gca()
    text = "x={:.1f}, y={:.4f}".format(x_max, y_max)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrow_props = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrow_props, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(x_max, y_max), xytext=(0.7, 0.96), **kw)
    # to contain the annotation
    ax.set_ylim(top=ax.get_ylim()[1] + 0.001)

    plt.xlabel('alpha (step={:.1f})'.format(step))
    plt.ylabel('mean rewards')
    plt.title('LinUCB: mean rewards by alpha')
    plt.show()
