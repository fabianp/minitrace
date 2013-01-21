import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as splinalg

def rank_one(X, y, alpha, shape_B, u0=None, v0=None, rtol=1e-6, maxiter=1000, verbose=False):
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
        v0 = np.ones(shape_B[1]) # np.random.randn(shape_B[1])
    for _ in range(maxiter):

        # update v
        v0 = v0.reshape((shape_B[1], 1))
        H = X.dot(sparse.kron(v0, sparse.eye(shape_B[1], shape_B[1])))
        def K_matvec(z):
            return H.T.dot(H.dot(z))
        K = splinalg.LinearOperator((H.shape[1], H.shape[1]), matvec=K_matvec, dtype=H.dtype)
        Ky = H.T.dot(y)
        u0, info = splinalg.cg(K, Ky, x0=u0, tol=rtol)

        # update u
        u0 = u0.reshape((shape_B[0], 1))
        G = X.dot(sparse.kron(sparse.eye(shape_B[0], shape_B[0]), u0))
        def K2_matvec(z):
            return G.T.dot(G.dot(z)) + alpha * z
        K = splinalg.LinearOperator((G.shape[1], G.shape[1]), matvec=K2_matvec, dtype=G.dtype)
        Ky = G.T.dot(y) + alpha * np.ones(H.shape[1])
        v0, info = splinalg.cg(K, Ky, x0=v0, tol=rtol)

        if verbose:
            v0 = v0.reshape((shape_B[1], 1))
            pobj = np.linalg.norm(y - X.dot(np.dot(u0, v0.T).ravel('F'))) ** 2 + alpha * linalg.norm(v0 - 1) ** 2
            print('POBJ: %s' % pobj)
    return u0.reshape((shape_B[0], 1)), v0.reshape((shape_B[1], 1))

if __name__ == '__main__':
    X = sparse.csr_matrix(np.random.randn(1000, 100))
    u_true, v_true = np.random.rand(10, 1), 1 + .3 * np.random.randn(10, 1)
    B_lowrank = np.dot(u_true, v_true.T)
    y = X.dot(B_lowrank.ravel('F')) + .5 * np.random.randn(X.shape[0])
    u, v = rank_one(X, y, 1e-3, (10, 10), maxiter=200, verbose=True)

    import pylab as plt
    plt.matshow(B_lowrank)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u, v.T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()