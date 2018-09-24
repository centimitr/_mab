from mab import MAB
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


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

    def play(self, tround, context=None):

    def update(self, arm, reward, context=None):
