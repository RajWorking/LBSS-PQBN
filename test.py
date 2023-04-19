import unittest
from prelims import *
from SampleD import SampleD


class TestStringMethods(unittest.TestCase):

    # def test_trapdoor_sampling(self):
    #     B, t = trapdoor_sampling()

    def test_SampleD(self):

        B = np.arange(100, 100 + 7*7).reshape((7, 7))
        s = 6599879
        c = np.arange(13, 13 + 7)

        sample = SampleD(B, s, c)
        print("Sampled vector: ", sample)


if __name__ == '__main__':
    unittest.main()
