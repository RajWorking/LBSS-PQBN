import numpy as np
import scipy as sc
from sympy import Matrix, lcm
from sympy.matrices.normalforms import hermite_normal_form
import config

def matrix_rational_to_integer(X, q):
    X = np.array(X)
    X_lcm = lcm([term.q for term in X.flatten()])
    X_lcm_inv = pow(X_lcm, -1, q)
    X = (X_lcm_inv * (X * X_lcm) % q) % q
    return X

def dual_lattice_basis(A, q):
    '''
    Find basis of lattice L = {x | Ax = 0 mod q}
    '''
    n, m = A.shape
    mat = Matrix(A)
    mat = np.array(mat.nullspace())[:, :, 0].T
    mat = matrix_rational_to_integer(mat, q)
    mat = np.append(mat, q*np.identity(m).astype(int), axis=1)
    assert np.sum((A @ mat) % q) == 0
    return HNF(mat)


def HNF(B):
    # Compute Hermite Normal Form of a lattice using basis B
    mat = Matrix(B)
    hnf = hermite_normal_form(mat)
    H = np.array(hnf.tolist()).astype(np.int32)
    return H

def solve_lineareqn(A, b, q):
    '''
    find x:
    A * x = b mod q
    '''
    nA, mA = A.shape
    _, mb = b.shape
    
    x = np.zeros(shape=(mA, mb), dtype=int)
    mat = Matrix(A[:, :nA])
    x[:nA] = matrix_rational_to_integer((mat ** -1) * b, q)

    assert np.all((A @ x) % q == b % q)
    return x

def gram_schmidt(A):
    '''
    Gram-Schmidt orthogonalization column-wise of matrix A
    '''
    n, m = A.shape
    B = np.zeros((n, m))
    B[:, 0] = A[:, 0]
    for i in range(1, m):
        B[:, i] = A[:, i]
        for j in range(i):
            L = np.dot(A[:, i], B[:, j]) / np.dot(B[:, j], B[:, j])
            B[:, i] -= L * B[:, j]
    return B

if __name__ == '__main__':
    n = 3
    m = 5
    r = 4
    A = np.random.randint(10, size=n*r).reshape(n, r)
    b = np.random.randint(10, size=n*m).reshape(n, m)
    x = solve_lineareqn(A, b, config.q)
