'''
Define all security parameters here
'''
import math

n = 512

# large prime number
q = 28468961266298161708283942383236250722126064247639

m = int(2 * n * math.log2(q))

# Gaussian parameter
s = int(math.sqrt(n * math.log2(n) * math.log2(q)))

# print(n, q, m, s)

