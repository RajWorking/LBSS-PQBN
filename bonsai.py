import numpy as np

from SampleD import SampleD, SampleZ
from utils import HNF, solve_lineareqn
from config import *


# The bonsai tree form the paper: Bonsai Trees, or How to Delegate a Lattice Basis
# The bonsai tree is a hierarchy of trapdoor functions.


def ToBasis(S: np.matrix, B: np.matrix):
    """ToBasis
    Deterministic poly-time algorithm that, given a full-rank set
    (not necessarily a basis) S of lattice vectors in Λ = L(B),
    outputs a basis T of Λ such that |t~_i | ≤ |s~_i| for all i.
    """
    l = sorted(S.T.tolist(), key=lambda x: np.linalg.norm(x))
    S = np.array(l)    
    # for s in S:
    #     print(np.linalg.norm(s))
    Q = (np.linalg.inv(B) @ S).astype(int)
    # find an unimodular matrix U such that T = UQ is upper triangular
    Uinv, T = np.linalg.qr(Q)
    return B @ Uinv


def RandBasis(S: np.matrix, s: int) -> np.matrix:
    """RandBasis Algorithm
    The probabilistic polynomial-time algorithm RandBasis(S, s) takes
    a basis S of an m-dimensional integer lattice Λ and
    a parameter s ≥ ||S~|| · ω(√log n), and outputs a basis S' of Λ,
    generated as follows.
    1. Let i ← 0. While i < m,
     (a) Choose v ← SampleD(S, s). If v is linearly independent of
       {v1, . . . , vi}, then let i ← i + 1 and let vi = v.
    2. Output S' = ToBasis(V, HNF(S))
    """
    m = S.shape[0]
    V = np.zeros((m, m))
    i = 0
    while i < m:
        c = np.zeros(m)
        v = SampleD(S, s, c)
        # check if v is linearly independent of vis
        Vprime = np.hstack((V, v[:, None]))
        if np.linalg.matrix_rank(Vprime) == i + 1:
            V[i] = v
            i += 1
    return ToBasis(V, HNF(S))


def ExtBasis(S: np.matrix, A: np.matrix, A_bar: np.matrix) -> np.matrix:
    """
    ExtBasis Algorithm
    The probabilistic polynomial-time algorithm ExtBasis(S, A, A') takes
    a basis S of an m-dimensional integer lattice Λ,
    a matrix A ∈ Z^{m × m_tilde}, and a matrix A' ∈ Z^{m × m_tilde}, and
    outputs a basis S' of Λ, generated as follows.
    There is a deterministic polynomial-time algorithm ExtBasis
    with the following properties:

    Args:
        S: A basis of an m-dimensional integer lattice Λ
        A: A matrix A ∈ Z^{n × m}
        A_bar: A matrix A' ∈ Z^{n × m_tilde}

    Returns:
    """
    m = S.shape[0]
    m_bar = A_bar.shape[1]
    m_prime = m + m_bar

    S_prime = np.zeros((m_prime, m_prime), dtype=int)
    S_prime[:m, :m] = S

    I = np.eye(m_bar)
    S_prime[m:, m:] = I
    W = solve_lineareqn(A, -A_bar, q)
    S_prime[:m, m:] = W
    return S_prime


def OptimizedGaussianSampling(S: np.matrix,
                              A: np.matrix,
                              A_bar: np.matrix,
                              q: int,
                              s: int) -> np.matrix:
    n, m = A.shape
    n, m_bar = A_bar.shape
    v_bar = np.array([SampleZ(s, 0, n) for _ in range(m_bar)])    
    y = (-A_bar @ v_bar) % q
    t = solve_lineareqn(A, y.reshape(n, 1), q) # center for sampleD
    e = SampleD(S, s, (-t) % q)
    v = e + t
    v_prime = np.concatenate((v, v_bar), axis=0) % q
    return v_prime


def OptimizedRandBasis(S: np.matrix,
                       A: np.matrix,
                       A_bar: np.matrix,
                       q: np.matrix,
                       s: int):
    n = A.shape[0]
    m = S.shape[0]
    m_bar = A_bar.shape[1]
    m_prime = m + m_bar
    V = np.zeros((m_prime, m_prime))
    i = 0
    while i < m_prime:
        c = np.zeros(n)
        v = OptimizedGaussianSampling(S, A, A_bar, c, q, s)
        # check if v is linearly independent of vis
        Vprime = np.hstack((V, v[:, None]))
        if np.linalg.matrix_rank(Vprime) == i + 1:
            V[i] = v
            i += 1
    return ToBasis(V, HNF(S))
