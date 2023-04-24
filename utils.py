import numpy as np
import scipy as sc
from sympy import *
from sympy.matrices.normalforms import hermite_normal_form
from config import q

def matrix_rational_to_integer(X):
    X = np.array(X)
    X_lcm = lcm([term.q for term in X.flatten()])
    X_lcm_inv = pow(X_lcm, -1, q)
    X = (X_lcm_inv * (X * X_lcm) % q) % q
    return X

def dual_lattice_basis(A):
    '''
    Find basis of lattice L = {x | Ax = 0 mod q}
    '''
    n, m = A.shape
    mat = Matrix(A)   
    mat = np.array(mat.nullspace())[:, :, 0].T
    mat = matrix_rational_to_integer(mat)
    mat = np.append(mat, q*np.identity(m).astype(int), axis=1)
    assert np.sum((A @ mat) % q) == 0
    return HNF(mat)
    

def HNF(B):
    # Compute Hermite Normal Form of a lattice using basis B
    mat = Matrix(B)
    hnf = hermite_normal_form(mat)
    H = np.array(hnf.tolist()).astype(np.int32)
    return H

def solve_lineareqn(A, b):
    '''
    find x:
    A * x = b mod q
    '''
    nA, mA = A.shape
    b = b.reshape(nA, 1)
    x = np.zeros(mA, dtype=int)
    mat = Matrix(A[:, :n])
    x[:nA] = matrix_rational_to_integer((mat ** -1) * b).flatten()
    
    assert np.all((A @ x) % q == b.flatten())
    return x

if __name__ == '__main__':
    n = 3
    m = 5
    A = np.random.randint(10, size=n*m).reshape(n, m)
    b = np.random.randint(10, size=n)
    x = solve_lineareqn(A, b)
    print(x)
