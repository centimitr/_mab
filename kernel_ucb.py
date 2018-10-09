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
        self.history_context = []
        self.tround = 1

        # self.last_k_ctx = None
        self.last_Kinv = None
        return

    def _ctx_for_arm(self, context, arm):
        return context[self.ndims * (arm - 1):self.ndims * arm]

    def _k_ctx_for_arm(self, ctx, ctx_list):
        return np.array([self.kern([ctx, ctx_item]) for ctx_item in ctx_list]).T

    def _inv(self, a):
        phalanx = a.shape[0] == a.shape[1]
        invertible = phalanx and np.linalg.matrix_rank(a) == a.shape[0]
        return inv(a) if invertible else a

    def play(self, tround, context):
        if tround == 1:
            self.u = [1 if idx == 0 else 0 for idx in range(self.narms)]
        else:
            for arm in self.arm_range:
                cur_ctx = self._ctx_for_arm(context, arm)
                cur_k_ctx = self._k_ctx_for_arm(cur_ctx, self.history_context)
                y = np.array(self.history_rewards).T

                sigma = np.sqrt(self.kern([cur_ctx, cur_ctx]) - np.dot(np.dot(cur_k_ctx.T, self.last_Kinv),
                                                                       cur_k_ctx))
                u = np.dot(np.dot(cur_k_ctx.T, self.last_Kinv), y) + (self.eta / np.sqrt(self.gamma)) * sigma

                idx = arm - 1
                self.u[idx] = u

        idx = np.argmax(self.u)
        arm = idx + 1
        self.tround = tround
        return arm

    def update(self, arm, reward, context):
        ctx = self._ctx_for_arm(context, arm)
        self.history_context.append(ctx)
        self.history_rewards.append(reward)
        # y = np.array(self.history_rewards).T
        # k_ctx = self._k_ctx_for_arm(context, self.arm_range)
        kern_ctx_ctx = self.kern([ctx, ctx])

        if self.tround == 1:
            Kinv = 1. / (kern_ctx_ctx + self.gamma)
        else:
            b = self._k_ctx_for_arm(ctx, self.history_context)
            btKinv = np.dot(b.T, self.last_Kinv)
            Kinvb = np.dot(self.last_Kinv, b)

            K22 = self._inv((kern_ctx_ctx + self.gamma) - np.dot(btKinv, b))
            K11 = self.last_Kinv + np.dot(np.dot(K22, Kinvb), btKinv)
            K12 = np.dot(-K22, Kinvb)
            K21 = np.dot(-K22, btKinv)
            #
            Kinv = np.block([[K11, K12], [K21, K22]])

        self.last_Kinv = Kinv
