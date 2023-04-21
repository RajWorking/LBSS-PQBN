from typing import Optional
import numpy as np
from .bonsai import RandBasis, ExtBasis
import hashlib


class LBSS:
    def __init__(self, n: int, q: int, s: float, hash_fn = hashlib.sha256):
        self.n = n
        self.q = q
        self.m = 2 * self.n * self.q
        self.hash_fn = hash_fn
        A0, T_A0 = Trapgen(self.n, self.q)
        self.A0 = A0
        self.T_A0 = T_A0
        self.s = s
        self.keys = self.gen()

    def gen(self):
        A = [np.random.rand(self.n, self.m) for _ in range(self.n)]
        A_prime = [A[i] + self.A0 for i in range(self.n)]
        T_A_prime = [RandBasis(ExtBasis(self.T_A0, self.A0, A_prime[i]), self.s)
                      for i in range(self.n)]
        return [(x, y) for x, y in zip(A_prime, T_A_prime)]

    def sign(self, x: bytes):
        h = self.hash_fn(x).digest()
        # convert the hash to a numpy array of bits
        h = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
        l = len(h)
        hamming_weight = np.sum(h)
        B = [np.random.rand(self.n, self.m) for _ in range(l)]
        B_m: Optional[np.array] = None
        for i in range(l):
            if h[i] == 1:
                B_m = np.vstack([A[i]] + [B[i][j] for j in range(hamming_weight)])
                