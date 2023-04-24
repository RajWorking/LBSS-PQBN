'''
Define all security parameters here
'''
import math

n = 32

# prime number
q = 40237 # 822220415899

delta = 1
m1 = math.ceil((1 + delta) * n * math.log2(q))
m2 = math.ceil((4 + 2*delta) * n * math.log2(q))
m = m1 + m2

# Gaussian parameter
s = int(math.sqrt(n * math.log2(n) * math.log2(q)))

# print(n, m, m1, m2, s, q)
