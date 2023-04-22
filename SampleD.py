import numpy as np
import math


def SampleZ(n, s, c):
    """
    Samples from the discrete Gaussian D_Z,s,c over the one-dimensional integer lattice Z, using rejection sampling.

    Args:
        n (int): (implicit) security parameter
        s (float): Standard deviation of the Gaussian distribution
        c (np.ndarray): Center of the Gaussian distribution.

    Returns:
        np.ndarray: Sampled integer from the discrete Gaussian distribution over integers.
    """
    while True:
        x = np.random.randint(math.ceil(c - s * math.log(n)),
                              math.ceil(c + s * math.log(n)))
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

    n = len(c)
    v = np.zeros(n)

    print(B.shape)
    # Gram-Schmidt Decomposition of basis B
    B_gs, _ = np.linalg.qr(B)

    for i in range(n-1, -1, -1):
        c_ = np.dot(c, B_gs[i])
        z = SampleZ(n, s, c_)
        c -= z * B[i]
        v += z * B[i]

    return v
