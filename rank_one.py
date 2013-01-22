import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as splinalg

def rank_one(X, y, alpha, shape_B, Z=None, u0=None, v0=None, rtol=1e-6, maxiter=1000, verbose=False):
    """

    min over u and v:
         || y - X vec(u v.T) - Z w||_2 ^2 + alpha ||v - 1||_2 ^2


    Parameters
    ----------
    """
    #X = splinalg.aslinearoperator(X)
    if u0 is None:
        u0 = np.random.randn(shape_B[0])
    if v0 is None:
        v0 = np.ones(shape_B[1]) # np.random.randn(shape_B[1])
    w0 = None
    if Z is not None:
        w0 = np.zeros(Z.shape[1])

    for _ in range(maxiter):

        # update v
        v0 = v0.reshape((shape_B[1], 1))
        Kron_v = sparse.kron(v0, sparse.eye(shape_B[1], shape_B[1]))
        def K_matvec(z):
            return Kron_v.T.dot(X.T.dot(X.dot(Kron_v.dot(z))))
        K = splinalg.LinearOperator((Kron_v.shape[1], Kron_v.shape[1]), matvec=K_matvec, dtype=X.dtype)
        if Z is None:
            Ky = Kron_v.T.dot(X.T.dot(y))
        else:
            Ky = Kron_v.T.dot(X.T.dot(y - np.dot(Z, w0)))
        u0, info = splinalg.cg(K, Ky, x0=u0, tol=rtol)

        # update u
        u0 = u0.reshape((shape_B[0], 1))
        Kron_u = sparse.kron(sparse.eye(shape_B[0], shape_B[0]), u0)
        def K2_matvec(z):
            return Kron_u.T.dot(X.T.dot(X.dot(Kron_u.dot(z)))) + alpha * z
        K = splinalg.LinearOperator((Kron_u.shape[1], Kron_u.shape[1]), matvec=K2_matvec, dtype=X.dtype)
        if Z is None:
            Ky = Kron_u.T.dot(X.T.dot(y)) + alpha * np.ones(Kron_u.shape[1])
        else:
            Ky = Kron_u.T.dot(X.T.dot(y - np.dot(Z, w0))) + alpha * np.ones(Kron_u.shape[1])
        v0, info = splinalg.cg(K, Ky, x0=v0, tol=rtol)

        # update w
        if Z is not None:
            w0 = linalg.lstsq(Z, y - X.dot(Kron_u.dot(v0)))[0]

        if verbose:
            v0 = v0.reshape((shape_B[1], 1))
            pobj = np.linalg.norm(y - X.dot(np.dot(u0, v0.T).ravel('F'))) ** 2 + alpha * linalg.norm(v0 - 1) ** 2
            print('POBJ: %s' % pobj)
    if Z is None:
        return u0.reshape((shape_B[0], 1)), v0.reshape((shape_B[1], 1))
    else:
        return u0.reshape((shape_B[0], 1)), v0.reshape((shape_B[1], 1)), w0

if __name__ == '__main__':
    X = sparse.csr_matrix(np.random.randn(1000, 100))
    Z = np.random.randn(1000, 4)
    u_true, v_true = np.random.rand(10, 1), 1 + .3 * np.random.randn(10, 1)
    B_lowrank = np.dot(u_true, v_true.T)
    y = X.dot(B_lowrank.ravel('F')) + .5 * np.random.randn(X.shape[0])
    u, v, w = rank_one(X, y, 1e-3, (10, 10), Z=Z, maxiter=200, verbose=True)

    import pylab as plt
    plt.matshow(B_lowrank)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u, v.T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()