import numpy as np
from numpy.linalg import inv

from mab import MAB


class KernelUCB(MAB):
    """
    Kernelised contextual multi-armed bandit (Kernelised LinUCB)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    gamma : float
        positive real explore-exploit parameter

    eta : float
        positive real explore-exploit parameter

    kern : callable
        a kernel function from sklearn.metrics.pairwise
    """

    def __init__(self, narms, ndims, gamma, eta, kern):
        self.narms = narms
        self.ndims = ndims
        self.gamma = gamma
        self.eta = eta
        self.kern = kern
        self.arm_range = range(1, narms + 1)
        # u = np.zeros(narms)
        # u[0] = 1
        self.u = [1 if idx == 0 else 0 for idx in range(narms)]
        # self.u = u.T
        # self.y = np.array([])
        self.history_rewards = [0.]
        self.tround = 1
        self.last_k_ctx = None
        self.last_K_inv = None
        return

    def _ctx_for_arm(self, context, arm):
        return context[self.ndims * (arm - 1):self.ndims * arm]

    def _k_ctx_for_arm(self, context, arm_range):
        ctx_for_last_arm = self._ctx_for_arm(context, arm_range[-1])
        return np.array([self.kern([ctx_for_last_arm], [self._ctx_for_arm(context, arm)]) for arm in
                         arm_range]).T

    def play(self, tround, context):
        idx = np.argmax(self.u)
        arm = idx + 1
        self.tround = tround
        return arm

    def update(self, arm, reward, context):
        self.history_rewards.append(reward)
        # self.y = np.append(self.y, reward).T
        y = np.array(self.history_rewards).T
        ctx = self._ctx_for_arm(context, arm)
        k_ctx = self._k_ctx_for_arm(context, self.arm_range)
        kern_ctx_ctx = self.kern([ctx], [ctx])

        if self.tround == 1:
            K_inv = (kern_ctx_ctx + self.gamma).T
        else:
            b = self.last_k_ctx
            K22 = inv(kern_ctx_ctx + self.gamma - np.dot(np.dot(b.T, self.last_K_inv), b))
            K11 = self.last_K_inv + np.dot(np.dot(np.dot(np.dot(K22, self.last_K_inv), b), b.T), self.last_K_inv)
            K12 = -1 * np.dot(np.dot(K22, self.last_K_inv), b)
            K21 = -1 * np.dot(np.dot(K22, b.T), self.last_K_inv)
            K_inv = np.block([[K11, K12], [K21, K22]])

        for arm in self.arm_range:
            cur_ctx = self._ctx_for_arm(context, arm)
            cur_k_ctx = self._k_ctx_for_arm(self._ctx_for_arm(context, arm), range(1, arm + 1))
            sigma = np.sqrt(
                self.kern([cur_ctx], [cur_ctx]) - np.dot(np.dot(cur_k_ctx.T, K_inv), cur_k_ctx))
            u = np.dot(np.dot(cur_k_ctx.T, K_inv), y) + self.eta / np.sqrt(
                self.gamma) * sigma

            self.u[arm] = u

        self.last_k_ctx = k_ctx
        self.last_K_inv = K_inv
