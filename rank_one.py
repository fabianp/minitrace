import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as splinalg

def rank_one(X, y, alpha, shape_B, u0=None, v0=None, maxiter=1000):
    """

    min over u and v:
         || y - X vec(u v.T)||_2 ^2 + alpha ||v - 1||_2 ^2


    Parameters
    ----------
    """
    #X = splinalg.aslinearoperator(X)
    if u0 is None:
        u0 = np.random.randn(shape_B[0])
    if v0 is None:
        v0 = np.random.randn(shape_B[1])
    for _ in range(maxiter):
        u0 = u0.reshape((shape_B[0], 1))
        v0 = v0.reshape((shape_B[1], 1))

        # update v
        V = sparse.kron(v0, sparse.eye(shape_B[0], shape_B[0]))
        H = X.dot(V)
        def K_matvec(z):
            return H.T.dot(H.dot(z)) + alpha * z
        K = splinalg.LinearOperator((H.shape[1], H.shape[1]), matvec=K_matvec, dtype=H.dtype)
        Ky = H.T.dot(y) + alpha * np.ones(H.shape[0])
        v0, info = splinalg.cg(K, Ky)

        # update u
        U = sparse.kron(sparse.eye(shape_B[1], shape_B[1]), u0)
        G = X.dot(U)
        def K2_matvec(z):
            return G.T.dot(G.dot(z))
        K = splinalg.LinearOperator((G.shape[1], G.shape[1]), matvec=K2_matvec, dtype=G.dtype)
        Ky = G.T.dot(y)
        u0, info = splinalg.cg(K, Ky)
    return u0.reshape((shape_B[0], 1)), v0.reshape((shape_B[1], 1))

if __name__ == '__main__':
    X = sparse.csr_matrix(np.random.randn(10, 100))
    B = np.random.randn(10, 10)
    U, s, Vt = linalg.svd(B)
    s[1:] = 0.
    B_lowrank = np.dot(U, np.dot(np.diag(s), Vt))
    y = X.dot(B_lowrank.ravel('F'))
    u, v = rank_one(X, y, 100., (10, 10))

    import pylab as plt
    plt.matshow(B_lowrank)
    plt.colorbar()
    plt.show()
    plt.matshow(np.dot(u, v.T))
    plt.colorbar()
    plt.show()