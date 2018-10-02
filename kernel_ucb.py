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
        return

    def play(self, tround, context):
        return

    def update(self, arm, reward, context):
        return
