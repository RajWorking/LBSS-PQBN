import numpy as np
import scipy as sc
from config import *

# np.set_printoptions(threshold=np.inf, linewidth=100000)


def TrapGen():
    A1 = np.random.randint(q, size=(n, m1))
    return Algorithm_1(A1)


def Algorithm_1(A1):
    '''
    Refer Algorithm 1 on Page 10 of paper - Generating Shorter Bases for Hard Random Lattices
    Framework for constructing A ∈ Z_nxm q and basis S of orthogonal lattice to A

    Input:
        A1 (np.ndarray): integer (modulo q) matrix of dimensions n x m1
        m2 (int): dimension

    Output:
        A2 (np.ndarray): integer (modulo q) matrix of dimensions n x m2
        S (np.ndarray): integer (modulo q) matrix of dimensions m x m
    '''
    # TODO: compute H
    H = A1[:m1, :m1]  # HNF(A1)

    G = np.zeros((m1, m2))
    widths = np.ceil(np.log2(np.diagonal(H))).astype(int)

    col = 0
    e = np.identity(m1)
    for i, width in enumerate(widths):
        G[:, col: col+width] = e[i, None].T @ [2 ** np.arange(0, width)]
        col += width

    w = 2 ** math.floor(math.log2(m2 - 2 * n * math.log2(q)))
    C_ = 1e4
    G[:, col: col+w] = sc.linalg.hadamard(w)[:m1]

    print(G)

    U = np.zeros((m2, m2))
    R = np.zeros((m1, m2))
    P = np.zeros((m2, m1))
    C = np.identity(m1).astype(int)

    A2 = (-(A1 @ (R + G))) % q

    S = np.zeros((m, m))
    S[:m1, :m2] = (G + R)@U
    S[m1:, :m2] = U
    S[:m1, m2:] = R@P - C
    S[m1:, m2:] = P
    S %= q

    A = np.append(A1, A2, axis=1)
    return A, S


TrapGen()
