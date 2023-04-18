import unittest
from prelims import *
from SampleD import SampleD
from bonsai import HNF

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


if __name__ == '__main__':
    unittest.main()