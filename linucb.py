import numpy as np
from mab import MAB


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
        return

    def play(self, tround, context):
        return

    def update(self, arm, reward, context):
        return
