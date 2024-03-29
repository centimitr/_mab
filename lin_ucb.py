from mab import MAB
import numpy as np
from numpy.linalg import inv


class LinUCB(MAB):
    """
    Contextual multi-armed bandit (LinUCB)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    alpha : float
        positive real explore-exploit parameter
    """

    def __init__(self, narms, ndims, alpha):
        self.narms = narms
        self.ndims = ndims
        self.alpha = alpha
        self.arm_range = range(1, narms + 1)
        self.Aa_dict = {arm: np.identity(ndims) for arm in self.arm_range}
        self.ba_dict = {arm: np.zeros(ndims) for arm in self.arm_range}
        return

    def _ctx_for_arm(self, context, arm):
        return context[self.ndims * (arm - 1):self.ndims * arm]

    def _p_fn(self, Aa, ba, Xa):
        Aa_inv = inv(Aa)
        theta = np.dot(Aa_inv, ba)
        p = np.dot(theta.T, Xa) + self.alpha * np.sqrt(np.dot(np.dot(Xa.T, Aa_inv), Xa))
        return p

    def play(self, tround, context):
        # choose arm with max p value
        p_values = [self._p_fn(self.Aa_dict[arm], self.ba_dict[arm], self._ctx_for_arm(context, arm))
                    for arm in self.arm_range]
        idx = np.argmax(p_values)
        arm = idx + 1
        return arm

    def update(self, arm, reward, context):
        Xa = self._ctx_for_arm(context, arm)
        # update
        self.Aa_dict[arm] += np.dot(Xa, Xa.T)
        self.ba_dict[arm] += reward * Xa
