'''
Define all security parameters here
'''
import math
from signature import LBSS

n = 32

# prime number
q = 101  # 822220415899

delta = 1
m1 = math.ceil((1 + delta) * n * math.log2(q))
m2 = math.ceil((4 + 2*delta) * n * math.log2(q))
m = m1 + m2

# Gaussian parameter
s = int(math.sqrt(n * math.log2(n) * math.log2(q)))

print(f"{n=}")
print(f"{q=}")
print(f"{m=}")
print(f"{s=}")

if __name__ == '__main__':
    lbss = LBSS(n, q, s, m, delta)
    print("Generating Keys")
    keys = lbss.gen()
    pk, sk = keys[0]
    print("Signing")
    sgn = lbss.sign(b'hello world', pk, sk)
    print("Verifying")
    res = lbss.vrfy(b'hello world', sgn, pk)
    print(f"{res=}")
