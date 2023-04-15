import numpy as np
from lattice import HNF

def test_HNF():
	S = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	assert np.allclose(HNF(S), H)