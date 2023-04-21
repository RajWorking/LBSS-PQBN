import numpy as np
import sympy
from sympy.matrices.normalforms import hermite_normal_form

from SampleD import SampleD
from utils import HNF


# The bonsai tree form the paper: Bonsai Trees, or How to Delegate a Lattice Basis
# The bonsai tree is a hierarchy of trapdoor functions.


def ToBasis(S:np.matrix, B:np.matrix):
    # deterministic poly-time algorithm that, given a full-rank set (not necessarily a basis) S
    # of lattice vectors in Λ = L(B), outputs a basis T of Λ such that |t~_i | ≤ |s~_i| for all i.
    # TODO: Verify this logic works
    Q = np.linalg.inv(B) @ S
    # find an unimodular matrix U such that T = UQ is upper triangular
    Uinv, T = np.linalg.qr(Q)
    return B @ Uinv

def RandBasis(S: np.matrix, s: int) -> np.matrix:
    """
     RandBasis Algorithm
     The probabilistic polynomial-time algorithm RandBasis(S, s) takes a basis S of an m-dimensional integer
     lattice Λ and a parameter s ≥ ||S~|| · ω(√log n), and outputs a basis S' of Λ, generated as follows.
     1. Let i ← 0. While i < m,
     (a) Choose v ← SampleD(S, s). If v is linearly independent of {v1, . . . , vi}, then let i ← i + 1 and let vi = v.
     2. Output S' = ToBasis(V, HNF(S))
    """
    m = S.shape[0]
    V = np.zeros((m, m))
    i = 0
    while i < m:
        # TODO: is c random?
        c = np.random.randint(0, 100, size=(m, 1))
        v = SampleD(S, s, c)
        # check if v is linearly independent of vis
        Vprime = np.concatenate((V, v), axis=0)
        if np.linalg.matrix_rank(Vprime) == i + 1:
            V[i] = v
            i += 1
    return ToBasis(V, HNF(S))


def ExtBasis(S: np.matrix, A: np.matrix, A_bar: np.matrix) -> np.matrix:
    """
    ExtBasis Algorithm
    The probabilistic polynomial-time algorithm ExtBasis(S, A, A') takes a basis S of an m-dimensional integer
    lattice Λ, a matrix A ∈ Z^{m × m_tilde}, and a matrix A' ∈ Z^{m × m_tilde}, and outputs a basis S' of Λ, generated as follows.
    There is a deterministic polynomial-time algorithm ExtBasis with the following properties:

    Args:
        S: A basis of an m-dimensional integer lattice Λ
        A: A matrix A ∈ Z^{n × m}
        A_bar: A matrix A' ∈ Z^{n × m_tilde}

    Returns:
    """
    m = S.shape[0]
    m_bar = A_bar.shape[1]
    m_prime = m + m_bar

    S_prime = np.zeros((m_prime, m_prime))
    S_prime[:m, :m] = S

    I = np.eye(m_bar)
    S_prime[m:, m:] = I
    
    # TODO: might need pseudoinverse here
    W = np.linalg.inv(A) @ (-A_bar)

    S_prime[0:m, m:] = W
    return S_prime


