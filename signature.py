import hashlib
from typing import List, Optional

import numpy as np

from bonsai import ExtBasis, OptimizedRandBasis, OptimizedGaussianSampling, RandBasis
from TrapGen import TrapGen
from SampleD import SampleD


class LBSS:
    def __init__(self, n: int,
                 q: int,
                 s: int,
                 m: int,
                 trapgen_delta: int = 1,
                 hash_fn=hashlib.sha256):
        self.n: int = n
        self.q: int = q
        self.m: int = m
        assert (m >= n)
        self.hash_fn = hash_fn
        self.A0, self.T_A0 = TrapGen(self.n, self.q, trapgen_delta)
        assert self.A0.shape == (n, m)
        assert self.T_A0.shape == (m, m)
        self.s: int = s
        self.maxl: int = 256

    def gen(self):
        A = [np.random.randint(0, self.q, (self.n, self.m))
             for _ in range(self.n)]
        A_prime: List[np.array] = [
            np.hstack([self.A0, A[i]]) % self.q for i in range(self.n)]

        self.B = [np.random.randint(0, self.q, (self.n, self.m))
                  for _ in range(self.maxl)]

        # T_A_prime: List[np.array] = [RandBasis(ExtBasis(self.T_A0, self.A0, A[i]), self.s)
        #                              for i in range(self.n)]
        T_A_prime = [OptimizedRandBasis(self.T_A0,
                                        self.A0,
                                        A[i],
                                        self.q,
                                        self.s) for i in range(self.n)]
        return [(x, y) for x, y in zip(A_prime, T_A_prime)]

    def _get_B_m(self, x: bytes) -> np.matrix:
        h = self.hash_fn(x).digest()
        # convert the hash to a numpy array of bits
        h = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
        indexes = np.where(h == 1)[0]
        B_m = np.hstack([self.B[j] for j in indexes])
        return B_m

    def sign(self, x: bytes, pk: np.matrix, sk: np.matrix) -> np.matrix:
        B_m = self._get_B_m(x)
        # S_prime = ExtBasis(sk, pk, B_m)
        # B_m = np.hstack([pk, B_m])
        # m_prime = B_m.shape[1]
        # v = SampleD(S_prime, self.s, c=np.zeros(m_prime))
        v = OptimizedGaussianSampling(sk, pk, B_m, self.q, self.s)
        return v

    def vrfy(self, x: bytes, v: np.matrix, pk: np.matrix) -> bool:
        B_m = self._get_B_m(x)
        B_m = np.hstack([pk, B_m])
        m_prime = B_m.shape[1]
        if v.shape[0] != m_prime:
            print('Wrong size of signature')
            return False
        if np.sum((B_m @ v) % self.q) == 0:
            print('Does not belong to lattice')
            max = np.max([np.linalg.norm(column) for column in v.T])
            if max <= self.s * np.sqrt(m_prime):
                print('Too large signature')
                return True
        return False
