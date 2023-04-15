import numpy as np


# The bonsai tree form the paper: Bonsai Trees, or How to Delegate a Lattice Basis
# The bonsai tree is a hierarchy of trapdoor functions.

# SampleZ Algorithm


def SampleZ(s, c):
    """
    # SampleZ, which samples from the discrete Gaussian DZ,s,c over the onedimensional integer lattice Z.
     Let t(n) ≥ ω(√log n) be some fixed function, say, t(n) = log n. SampleZ
     uses rejection sampling, and works as follows: on input (s, c) and (implicitly) the security parameter n,
     choose an integer x ← Z = Z ∩ [c − s · t(n), c + s · t(n)] uniformly at random. Then with probability
     ρ_s(x − c) ∈ (0, 1], output x, otherwise repeat.
    """
    pass


def SampleD(B, s, c):
    """
    # SampleD Algorithm
     A nearest-plane algorithm, called SampleD, that samples from a discrete
     Gaussian DΛ,s,c over any lattice Λ. In each iteration, the algorithm simply chooses a plane at random by
     sampling from an appropriate discrete Gaussian over the integers Z.
     The input to SampleD is an (ordered) basis B of an n-dimensional lattice Λ, a parameter s > 0, and a
     center c ∈ R^n.
     We describe the algorithm as if it has access to an oracle that samples exactly from D_{Z,s',c'}
     for any desired s' > 0 and c' ∈ R. (As long as s'
     is sufficiently large, the oracle can be implemented by the
     SampleZ algorithm described above.) SampleD proceeds as follows:
     1. Let v_n ← 0 and c_n ← c. For i ← n, . . . , 1, do:
       # (a) Let c'_i = <ci, b˜i> and s'_i = s/||b˜i|| > 0.
       # (b) Choose zi ∼ D_{Z,s',c'} (this is the only step that differs from the nearest-plane algorithm).
       # (c) Let c_{i−1} ← ci − z_i * b_i and let v_{i−1} ← v_i + z_i * b_i.
     2. Output v0.
     Assuming scalar operations take unit time, the running time of the algorithm is O(n^2) plus the running time of the n oracle calls.
    """
    pass


def ToBasis(V, S) -> np.matrix:
    # deterministic poly-time algorithm ToBasis(S, B)
    # that, given a full-rank set (not necessarily a basis) S of lattice vectors in Λ = L(B), outputs a basis T of Λ
    # such that |t~_i | ≤ |s~_i| for all i.
    pass


def HNF(S):
    # Compute Hermite Normal Form of a lattice basis using the LLL algorithm
    pass


def RandBasis(S: np.matrix, s: int) -> np.matrix:
    """
    # RandBasis Algorithm
     The probabilistic polynomial-time algorithm RandBasis(S, s) takes a basis S of an m-dimensional integer
     lattice Λ and a parameter s ≥ ||S~|| · ω(√log n), and outputs a basis S' of Λ, generated as follows.
     1. Let i ← 0. While i < m,
     (a) Choose v ← SampleD(S, s). If v is linearly independent of {v1, . . . , vi}, then let i ← i + 1 and let vi = v.
     2. Output S' = ToBasis(V, HNF(S))
    """
    m = S.shape[0]
    V = np.zeros((m, m))
    for i in range(m):
        while True:
            v = SampleD(S, s)
            # check if v is linearly independent of vis
            Vprime = np.concatenate((V, v), axis=0)
            if np.linalg.matrix_rank(Vprime) == i + 1:
                V[i] = v
                break
    return ToBasis(V, HNF(S))
