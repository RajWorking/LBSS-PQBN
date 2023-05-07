import numpy as np
import math
from sympy import *
from utils import *
from config import *

def SampleZ(s, c) -> int:
    """
    Samples from the discrete Gaussian D_Z,s,c over the one-dimensional integer lattice Z, using rejection sampling.

    Args:
        n (int): (implicit) security parameter
        s (float): Standard deviation of the Gaussian distribution
        c (np.ndarray): Center of the Gaussian distribution.

    Returns:
        int : Sampled integer from the discrete Gaussian distribution over integers.
    """
    while True:
        x = np.random.randint(math.ceil(c - s * math.log(security_param)),
                              math.ceil(c + s * math.log(security_param)))
        p = (math.e ** (-math.pi * ((x - c)*(x - c)) / (s*s)))
        if np.random.rand() < p:
            return x


def SampleD(B, s, c):
    """
    A nearest-plane algorithm that samples from a discrete Gaussian D_Λ,s,c over any lattice Λ.
    In each iteration, the algorithm simply chooses a plane at random by sampling from an appropriate discrete Gaussian over the integers Z.
    Assuming scalar operations take unit time, the running time of the algorithm is O(n^2) plus the running time of the n oracle calls.

    Args:
        B (np.ndarray): an (ordered) basis of an n-dimensional lattice Λ
        s (float): Standard deviation of the Gaussian distribution
        c (np.ndarray): Center of the Gaussian distribution.

    Returns:
        np.ndarray: Sampled vector from the discrete Gaussian distribution.
    """
    m:int = len(c)
    v = np.zeros(m, dtype=int)
    B_gs = gram_schmidt(B).T # Gram-Schmidt Decomposition of basis B
    B_gs = B_gs.T
    for i in range(m-1, -1, -1):
        c_ = 0.01 * np.dot(c, B_gs[i]) / np.dot(B_gs[i], B_gs[i])  # float
        s_ = s / np.linalg.norm(B_gs[i])
        z = SampleZ(s_, c_)
        c -= z * B.T[i]
        v += z * B.T[i]
    return v
