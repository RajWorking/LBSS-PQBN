import numpy as np
from bonsai import HNF

def test_HNF():
	S = np.array([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
	H = np.array([[10, 0, 2], [0, 15, 3], [0, 0, 2]])
	assert np.allclose(HNF(S), H)