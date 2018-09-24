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
            k = np.random.randint(0, self.narms)
        else:
            k = np.argmax(self.Qs)
        arm = k + 1
        return arm

    def update(self, arm, reward, context=None):
        # current values
        k = arm - 1
        q = self.Qs[k]
        n = self.Ns[k]
        # update
        self.Qs[k] = ((q * n) + reward) / (n + 1)
        self.Ns[k] = n + 1
