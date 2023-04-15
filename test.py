import unittest
from prelims import *

class TestStringMethods(unittest.TestCase):

    def test_trapdoor_sampling(self):
        B, t = trapdoor_sampling()

    def test_SampleD(self):
        
        # Example usage
        # Define lattice basis Λ, standard deviation s, and center c
        Λ = np.array([[1, 2], [3, 4]])
        s = 1
        c = np.array([0.5, 0.5])

        # Sample from the discrete Gaussian distribution
        sample = sample_d(Λ, s, c)
        print("Sampled vector: ", sample)

        # self.assertTrue('FOO'.isupper())
        # self.assertFalse('Foo'.isupper())


if __name__ == '__main__':
    unittest.main()