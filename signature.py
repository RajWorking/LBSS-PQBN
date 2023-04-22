import hashlib
from typing import Optional

import numpy as np

from bonsai import ExtBasis, RandBasis
from TrapGen import TrapGen
from SampleD import SampleD


class LBSS:
    def __init__(self, n: int,
                 q: int,
                 s: float,
                 m: int,
                 trapgen_delta: int = 1,
                 hash_fn=hashlib.sha256):
        self.n = n
        self.q = q
        self.m = m
        self.hash_fn = hash_fn
        self.A0, self.T_A0 = TrapGen(self.n, self.q, trapgen_delta)
        assert self.A0.shape == (n, m)
        assert self.T_A0.shape == (m, m)
        self.s = s

    def gen(self):
        A = [np.random.randint(0, self.q, (self.n, self.m))
             for _ in range(self.n)]
        A_prime = [np.hstack([self.A0, A[i]]) % self.q for i in range(self.n)]
        T_A_prime = [RandBasis(ExtBasis(self.T_A0, self.A0, A_prime[i]), self.s)
                     for i in range(self.n)]
        return [(x, y) for x, y in zip(A_prime, T_A_prime)]

    def _get_B_m(self, x: bytes) -> np.matrix:
        h = self.hash_fn(x).digest()
        # convert the hash to a numpy array of bits
        h = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
        l = len(h)
        indexes = np.where(h == 1)[0]
        B = [np.random.rand(self.n, self.m) for _ in range(l)]
        B_m = np.hstack([B[j] for j in indexes])
        return B_m

    def sign(self, x: bytes, pk: np.matrix, sk: np.matrix) -> np.matrix:
        B_m = self._get_B_m(x)
        S_prime = ExtBasis(sk, pk, B_m)
        m_prime = B_m.shape[1] + self.m
        v = SampleD(S_prime, self.s, c=np.zeros((m_prime, 1)))
        return v

    def vrfy(self, x: bytes, v: np.matrix, pk: np.matrix) -> bool:
        if len(v) != self.n:
            return False
        B_m = self._get_B_m(x)
        B_m = np.hstack([pk, B_m])
        m_prime = B_m.shape[1]
        Temp = (B_m @ v) % self.q
        if np.allclose(Temp, 0):
            max = np.max([np.linalg.norm(column) for column in v.T])
            if max <= self.s * np.sqrt(m_prime):
                return True
        return False
