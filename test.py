import unittest
from SampleD import SampleD
from utils import *



class TestStringMethods(unittest.TestCase):

    # def test_trapdoor_sampling(self):
    #     B, t = trapdoor_sampling()

    def test_SampleD(self):

        B = np.arange(100, 100 + 7*7).reshape((7, 7))
        s = 6599879
        c = np.arange(13, 13 + 7)

        sample = SampleD(B, s, c)
        print("Sampled vector: ", sample)
    
    def test_HNF(self):
        S = np.array([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
        H = np.array([[10, 0, 2], [0, 15, 3], [0, 0, 2]])
        assert np.allclose(HNF(S), H)
        
    def test_dual_lattice_basis(self):
        nA = 3
        mA = 7
        A = np.random.randint(10, size=nA*mA).reshape(nA, mA)
        D = dual_lattice_basis(A)


if __name__ == '__main__':
    unittest.main()
