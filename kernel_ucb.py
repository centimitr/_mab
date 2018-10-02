import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
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
        u = np.zeros(narms)
        u[0] = 1
        self.u = u.T
        self.reward_history = []
        self.tround = 1
        self.last_k_ctx = None
        self.last_K_inv = None
        return

    def _ctx_for_arm(self, context, arm):
        return context[self.ndims * (arm - 1):self.ndims * arm]

    def _k_ctx_for_arm(self, ctx, arm_range):
        return [self.kern(ctx, self._ctx_for_arm(arm)) for arm in arm_range]

    def play(self, tround, context):
        idx = np.argmax(self.u)
        arm = idx + 1
        self.tround = tround
        return arm

    def update(self, arm, reward, context):
        self.reward_history.append(reward)
        ctx = self._ctx_for_arm(arm)
        k_ctx = self._k_ctx_for_arm(ctx, self.arm_range)
        kern_ctx_ctx = self.kern(ctx, ctx)

        if self.tround == 1:
            K_inv = (kern_ctx_ctx + self.gamma).T
        else:
            b = self.last_k_ctx
            K22 = (kern_ctx_ctx + self.gamma - np.dot(np.dot(b.T, self.last_K_inv), b)).T
            K11 = self.last_K_inv + np.dot(np.dot(np.dot(np.dot(K22, self.last_K_inv), b), b.T), self.last_K_inv)
            K12 = -1 * np.dot(np.dot(K22, self.last_K_inv), b)
            K21 = -1 * np.dot(np.dot(K22, b.T), self.last_K_inv)
            K_inv = np.block([[K11, K12], [K21, K22]])

        for arm in self.arm_range:
            cur_k_ctx = self._k_ctx_for_arm(self._ctx_for_arm(context, arm))
            cur_ctx = self._ctx_for_arm(arm)
            sigma = np.sqrt(
                self.kern(cur_ctx, cur_ctx) - np.dot(np.dot(cur_k_ctx.T, K_inv), cur_k_ctx))
            u = np.dot(np.dot(cur_k_ctx.T, K_inv), np.array(self.reward_history).T) + self.eta / np.sqrt(
                self.gamma) * sigma

            self.u[arm] = u

        self.last_k_ctx = k_ctx
        self.last_K_inv = K_inv
