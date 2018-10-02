from mab import MAB
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


class EpsGreedy(MAB):
    """
    Epsilon-Greedy multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    epsilon : float
        explore probability

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, epsilon, Q0=np.inf):
        self.narms = narms
        self.epsilon = epsilon
        self.Qs = np.full(narms, Q0)  # Q value for each arm
        self.Ns = np.zeros(narms)  # Exploit times for each arm

    def play(self, tround, context=None):
        r = np.random.rand(0, 1)
        if r < self.epsilon:
            idx = np.random.randint(0, self.narms)
        else:
            idx = np.argmax(self.Qs)
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
