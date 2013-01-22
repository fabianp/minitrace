import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as splinalg

def rank_one(X, y, alpha, size_u, size_v, Z=None, u0=None, v0=None, rtol=1e-6, maxiter=100, verbose=False):
    """

    min over u and v:
         || y - X vec(u v.T) - Z w||_2 ^2 + alpha ||v - 1||_2 ^2


    Parameters
    ----------
    """
    #X = splinalg.aslinearoperator(X)
    if u0 is None:
        u0 = np.random.randn(size_u)
    if v0 is None:
        v0 = np.ones(size_v) # np.random.randn(shape_B[1])
    w0 = None
    if Z is not None:
        w0 = np.zeros(Z.shape[1])

    for _ in range(maxiter):

        # update v
        v0 = v0.reshape((size_v, 1))
        Kron_v = sparse.kron(v0, sparse.eye(size_u, size_u))
        def K_matvec(z):
            return Kron_v.T.dot(X.T.dot(X.dot(Kron_v.dot(z))))
        K = splinalg.LinearOperator((Kron_v.shape[1], Kron_v.shape[1]), matvec=K_matvec, dtype=X.dtype)
        if Z is None:
            Ky = Kron_v.T.dot(X.T.dot(y))
        else:
            Ky = Kron_v.T.dot(X.T.dot(y - np.dot(Z, w0)))
        u0, info = splinalg.cg(K, Ky, x0=u0, tol=rtol)
        u0 = u0.reshape((size_u, 1))

        # update u
        Kron_u = sparse.kron(sparse.eye(size_v, size_v), u0)
        def K2_matvec(z):
            return Kron_u.T.dot(X.T.dot(X.dot(Kron_u.dot(z)))) + alpha * z
        K = splinalg.LinearOperator((Kron_u.shape[1], Kron_u.shape[1]), matvec=K2_matvec, dtype=X.dtype)
        if Z is None:
            Ky = Kron_u.T.dot(X.T.dot(y)) + alpha * np.ones(Kron_u.shape[1])
        else:
            Ky = Kron_u.T.dot(X.T.dot(y - np.dot(Z, w0))) + alpha * np.ones(Kron_u.shape[1])
        v0, info = splinalg.cg(K, Ky, x0=v0, tol=rtol)
        v0 = v0.reshape((size_v, 1))

        # update w
        if Z is not None:
            w0 = linalg.lstsq(Z, y - X.dot(np.dot(u0, v0.T).ravel('F')))[0]

        if verbose:
            v0 = v0.reshape((size_v, 1))
            if Z is None:
                pobj = np.linalg.norm(y - X.dot(np.dot(u0, v0.T).ravel('F'))) ** 2 + alpha * linalg.norm(v0 - 1) ** 2
            else:
                pobj = np.linalg.norm(y - X.dot(np.dot(u0, v0.T).ravel('F')) - np.dot(Z, w0)) ** 2 + alpha * linalg.norm(v0 - 1) ** 2

            print('POBJ: %s' % pobj)
    if Z is None:
        return u0.reshape((size_u, 1)), v0.reshape((size_v, 1))
    else:
        return u0.reshape((size_u, 1)), v0.reshape((size_v, 1)), w0

if __name__ == '__main__':
    size_u, size_v = 10, 8
    X = sparse.csr_matrix(np.random.randn(1000, size_u * size_v))
    Z = np.random.randn(1000, 4)
    u_true, v_true = np.random.rand(size_u, 1), 1 + .3 * np.random.randn(size_v, 1)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .5 * np.random.randn(X.shape[0])
    u, v, w = rank_one(X, y, 1e-3, size_u, size_v, Z=Z, maxiter=200, verbose=True)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u, v.T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()