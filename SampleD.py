import numpy as np
import math


def SampleZ(s, c, n) -> int:
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
        # print(c, s * math.log(n))
        x = np.random.randint(math.ceil(- s * math.log(n)),
                              math.ceil( s * math.log(n)))
        x += int(c)
        p = (math.e ** (-math.pi * ((x - c)*(x - c)) / (s*s)))
        if np.random.rand() < p:
            return x


def SampleD(B, s, c, n):
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

    # print(c)
    # Gram-Schmidt Decomposition of basis B
    B_gs, _ = np.linalg.qr(B)

    for i in range(m-1, -1, -1):
        c_ = np.dot(c, B_gs[i]) # float
        z = SampleZ(s, c_, n)
        # print(z, max(B[i]))
        # input()
        # print(type(z), B.dtype)
        c -= z * B[i]
        v += z * B[i]

    print(v)

    return v
