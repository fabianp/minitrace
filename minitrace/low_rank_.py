import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as splinalg


def low_rank(X, y, alpha, shape_u, Z=None, u0=None, v0=None, rtol=1e-6, maxiter=100, verbose=False):
    """

    min over u and v:
         || y - X vec(u v.T) - Z w||_2 ^2 + alpha ||v - 1||_2 ^2


    Parameters
    ----------
    """
    X = splinalg.aslinearoperator(X)
    assert len(shape_u) == 2
    shape_v = (X.shape[1] / shape_u[0], shape_u[1]) # TODO: check first dimension is integer
    if u0 is None:
        u0 = np.random.randn(*shape_u)
    if v0 is None:
        v0 = np.ones(shape_v) # np.random.randn(shape_B[1])
    w0 = None
    if Z is not None:
        w0 = np.zeros(Z.shape[1])

    for _ in range(maxiter):

        # update v
        v0 = v0.reshape(shape_v)
        size_id = X.shape[1] / shape_v[0]
        Kron_v = sparse.kron(v0, sparse.eye(size_id, shape_u[0] * shape_u[1] / shape_v[1]))
        def K_matvec(z):
            return Kron_v.T.dot(X.rmatvec(X.matvec(Kron_v.dot(z))))
        K = splinalg.LinearOperator((Kron_v.shape[1], Kron_v.shape[1]), matvec=K_matvec, dtype=X.dtype)
        if Z is None:
            Ky = Kron_v.T.dot(X.rmatvec(y))
        else:
            Ky = Kron_v.T.dot(X.rmatvec(y - np.dot(Z, w0)))
        u0, info = splinalg.cg(K, Ky, x0=u0.ravel(), tol=rtol)
        u0 = u0.reshape(shape_u, order='F')

        # update u
        Kron_u = sparse.kron(sparse.eye(X.shape[1] / shape_u[0], shape_v[1] * shape_v[0] / shape_u[1]), u0)
        def K2_matvec(z):
            return Kron_u.T.dot(X.rmatvec(X.matvec(Kron_u.dot(z)))) + alpha * z
        K = splinalg.LinearOperator((Kron_u.shape[1], Kron_u.shape[1]), matvec=K2_matvec, dtype=X.dtype)
        if Z is None:
            Ky = Kron_u.T.dot(X.rmatvec(y)) + alpha * np.ones(Kron_u.shape[1])
        else:
            Ky = Kron_u.T.dot(X.rmatvec(y - np.dot(Z, w0))) + alpha * np.ones(Kron_u.shape[1])
        vt0, info = splinalg.cg(K, Ky, x0=v0.T.ravel(), tol=rtol)
        vt0 = vt0.reshape((shape_v[1], shape_v[0]), order='F')
        v0 = vt0.T

        # update w
        if Z is not None:
            # TODO: cache SVD(Z)
            w0 = linalg.lstsq(Z, y - X.matvec(np.dot(u0, v0.T).ravel('F')))[0]

        if verbose:
            v0 = v0.reshape(shape_v)
            if Z is None:
                pobj = np.linalg.norm(y - X.matvec(np.dot(u0, v0.T).ravel('F'))) ** 2 + alpha * linalg.norm(v0 - 1) ** 2
            else:
                pobj = np.linalg.norm(y - X.matvec(np.dot(u0, v0.T).ravel('F')) - np.dot(Z, w0)) ** 2 + alpha * linalg.norm(v0 - 1) ** 2

            print('POBJ: %s' % pobj)
    if Z is None:
        return u0.reshape(shape_u), v0.reshape(shape_v)
    else:
        return u0.reshape(shape_u), v0.reshape(shape_v), w0

if __name__ == '__main__':
    size_u, size_v = 10, 8
    X = sparse.csr_matrix(np.random.randn(1000, size_u * size_v))
    Z = np.random.randn(1000, 20)
    u_true, v_true = np.random.rand(size_u, 2), 1 + .1 * np.random.randn(size_v, 2)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .3 * np.random.randn(X.shape[0])
    u, v = low_rank(X, y, .1, (size_u, 2), Z=None, maxiter=500, verbose=True)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u, v.T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()