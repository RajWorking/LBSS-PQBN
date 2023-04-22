import numpy as np
from sympy import *
from sympy.matrices.normalforms import hermite_normal_form

def basis_from_parity_matrix(A):
    '''
    Find basis B of lattice L = {x | Ax = 0 mod q}
    '''
    pass
    # null = mat.nullspace()
    # null = np.array(null)[:, :, 0].T
    # print(null)

    # assert np.sum(B @ null) == 0

    # mat = Matrix(null)
    

def HNF(B):
    # Compute Hermite Normal Form of a lattice using basis B
    print(B)
    mat = Matrix(B)
    hnf = hermite_normal_form(mat)
    H = np.array(hnf.tolist()).astype(np.int32)
    return H

