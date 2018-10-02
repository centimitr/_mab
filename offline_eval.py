import numpy as np
from eps_greedy import EpsGreedy
from ucb import UCB


def offlineEvaluate(mab, arms, rewards, contexts, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """

    history = []
    total_reward = 0

    # exit when 0 rounds
    if nrounds == 0:
        return 0

    tround = 1
    idx = 0
    while tround <= nrounds:
        while True:
            context = contexts[idx]
            arm = arms[idx]
            reward = rewards[idx]
            idx += 1

            chosen_arm = mab.play(tround, context)
            if chosen_arm == arm:
                break
        # when matching
        mab.update(arm, reward, context)
        history.append((context, arm, reward))
        total_reward += reward
        tround += 1

    return total_reward / nrounds
