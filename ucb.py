from mab import MAB
import numpy as np


class UCB(MAB):
    """
    Upper Confidence Bound (UCB) multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    rho : float
        positive real explore-exploit parameter

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, rho, Q0=np.inf):
        self.narms = narms
        self.rho = rho
        self.Qs = np.full(narms, Q0)  # Q value for each arm
        self.Ns = np.zeros(narms)  # Exploit times for each arm
        self.Rs = np.zeros(narms)  # accumulated rewards, r in [0, 1]

    def play(self, tround, context=None):
        def est_fn(rho, r, n):
            reward_mean = r / n
            bonus = np.sqrt((rho * np.log(tround)) / n)
            return reward_mean + bonus

        # use unused arms
        zero_indices = np.where(self.Ns == 0)[0]
        if len(zero_indices) > 0:
            idx = np.random.choice(zero_indices)
        else:
            # choose arm with max estimate value
            est_values = [est_fn(self.rho, r, n) for (r, n) in zip(self.Rs, self.Ns)]
            idx = np.argmax(est_values)
        arm = idx + 1
        return arm

    def update(self, arm, reward, context=None):
        # current values
        idx = arm - 1
        q = self.Qs[idx]
        n = self.Ns[idx]
        # update
        self.Qs[idx] = ((q * n) + reward) / (n + 1) if n > 0 else reward
        self.Ns[idx] = n + 1
        self.Rs[idx] += reward
