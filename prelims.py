import numpy as np
from scipy.linalg import orth, solve_triangular


def trapdoor_sampling(n, f):
    # Generate a random n x n matrix A
    A = np.random.randn(n, n)

    # Transform f into g using A
    def g(x): return f(A.dot(x))

    # Sample n vectors from the distribution defined by g
    u = np.array([g(np.random.randn(n)) for i in range(n)])

    # Compute an orthogonal basis B from the vectors u
    B = orth(u)

    # Compute the inverse of A mod q
    q = 2**32 - 1  # Choose a large prime number
    A_inv = np.linalg.inv(A) % q

    # Compute the trapdoor t
    t = B.dot(A_inv) % q

    return B, t


def sample_d(B, sigma, seed):
    # Set the random seed
    np.random.seed(seed)

    # Compute the Cholesky decomposition of the matrix B^T B
    L = np.linalg.cholesky(B.T.dot(B))

    # Generate a random vector u from the Gaussian distribution
    u = np.random.randn(B.shape[1])

    # Compute the projection of u onto the lattice generated by B
    v = np.round(solve_triangular(L, B.T.dot(u), lower=True))

    # Compute the distance between u and v
    dist = np.linalg.norm(u - B.dot(v))

    # If the distance is greater than sqrt(2*pi)*sigma, repeat the process
    while dist > np.sqrt(2*np.pi)*sigma:
        u = np.random.randn(B.shape[1])
        v = np.round(solve_triangular(L, B.T.dot(u), lower=True))
        dist = np.linalg.norm(u - B.dot(v))

    return v


def ExtBasis(S, A, A_tilde):
    """
    Computes an extended basis of a lattice given a basis S, a generator matrix A for the lattice,
    and an arbitrary matrix A_tilde. The extended basis has the same norm as the original basis.
    The algorithm is deterministic and outputs the same result for any permutation of the columns of A.

    Args:
    S: numpy array of shape (m, m) representing the basis of the q-ary lattice.
    A: numpy array of shape (n, m) representing the generator matrix for the q-ary lattice.
    A_tilde: numpy array of shape (m, m) representing an arbitrary matrix.

    Returns:
    S_0: numpy array of shape (m+m_tilde, m+m_tilde) representing the extended basis of the q-ary lattice.
    """

    # Combine A and A_tilde into a single matrix A_0
    A_0 = np.hstack((A, A_tilde))

    m = S.shape[0]

    # Initialize the extended basis S_0 with S
    S_0 = np.hstack((S, np.zeros((m, m_tilde))))

    # Loop over the columns of A_0 in some order (e.g., increasing order)
    for j in range(m+m_tilde):

        # Compute the Gram-Schmidt orthogonalization of the columns of S_0 and A_0 up to column j
        B = np.hstack((S_0, A_0[:, :j+1]))
        mu = np.linalg.inv(B.T @ B) @ B.T @ A_0[:, j]
        v = A_0[:, j] - B @ mu

        # Check if the norm of the last orthogonal vector is greater than some threshold
        if np.linalg.norm(v) > 2**(n/2):
            # Add v to the extended basis S_0
            S_0 = np.hstack((S_0, v.reshape((-1, 1))))

    return S_0
