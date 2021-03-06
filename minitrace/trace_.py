import numpy as np
from scipy import linalg
from scipy.sparse import linalg as splinalg

def prox_l1(a, b):
    return np.sign(a) * np.fmax(np.abs(a) - b, 0)

def prox(X, t, v0, n_nonzero=1000, n=0, algo='dense', n_svals=10):
    """prox operator for trace norm
    Algo: {sparse, dense}
    """

    if algo=='sparse':
        k = min(np.min(X.shape), n_nonzero)
        u, s, vt = splinalg.svds(X, k=k, v0=v0[:, 0], maxiter=50, tol=1e-3)
    else:
        u, s, vt = linalg.svd(X, full_matrices=False)
        #u, s, vt = randomized_svd(X, n_svals)
    s[n:] = np.sign(s[n:]) * np.fmax(np.abs(s[n:]) - t, 0)
    low_rank = np.dot(u, np.dot(np.diag(s), vt))
    return low_rank, s, u, vt


def conj_loss(X, y, Xy, M, epsilon, sol0):
    # conjugate of the loss function
    n_features = X.shape[1]
    matvec = lambda z: X.rmatvec((X.matvec(z))) + epsilon * z
    K = splinalg.LinearOperator((n_features, n_features), matvec, dtype=X.dtype)
    sol = splinalg.cg(K, M.ravel(order='F') + Xy, maxiter=20, x0=sol0)[0]
    p = np.dot(sol, M.ravel(order='F')) - .5 * (linalg.norm(y - X.matvec(sol)) ** 2)
    p -= 0.5 * epsilon * (linalg.norm(sol) ** 2)
    return p, sol

def trace_pobj(X, y, B, alpha, epsilon, s_vals):
    n_samples, _ = X.shape
    bt = B.ravel(order='F')
    #s_vals = linalg.svdvals(B)
    return  0.5 * (linalg.norm(y - X.matvec(bt)) ** 2) + \
            0.5 * epsilon * (linalg.norm(bt) ** 2) + \
            alpha * linalg.norm(s_vals, 1)


def trace(X, y, alpha, beta, shape_B, rtol=1e-3, max_iter=1000, verbose=False, warm_start=None,
          n_svals=10, L=None):
    """
    solve the model:

        ||y - X vec(B)||_2 ^2 + alpha ||B||_* + beta ||B||_F

    where vec = B.ravel('F')

    Parameters
    ----------
    X : LinearOperator

    L : None

    shape_B : tuple
    """
    X = splinalg.aslinearoperator(X)
    n_samples = X.shape[0]
    #alpha = alpha * n_samples
    beta = beta * n_samples

    if warm_start is None:
        # fortran ordered !!important when reshaping !!
        B = np.asfortranarray(np.random.randn(*shape_B))
    else:
        B = warm_start
    gap = []

    if L is None:
        def K_matvec(v):
            return X.rmatvec(X.matvec(v)) + beta * v
        K = splinalg.LinearOperator((X.shape[1], X.shape[1]), matvec=K_matvec, dtype=X.dtype)
        L = splinalg.eigsh(K, 1, return_eigenvectors=False)[0]

    step_size = 1. / L
    Xy = X.rmatvec(y)
    v0 = None
    t = 1.
    conj0 = None
    for n_iter in range(max_iter):
        b = B.ravel(order='F')
        grad_g = -Xy + X.rmatvec(X.matvec(b)) + beta * b
        tmp = (b - step_size * grad_g).reshape(*B.shape, order='F')
        xk, s_vals, u0, v0 = prox(tmp, step_size * alpha, v0, n_svals=n_svals)
        tk = (1 + np.sqrt(1 + 4 * t * t)) / 2.
        B = xk + ((t - 1.) / tk) * (xk - B)
        t = tk
        if n_iter % 200 == 199:
            tmp = grad_g.reshape(*B.shape, order='F')
            tmp = splinalg.svds(tmp, 1, tol=.1)[1][0]
            scale = min(1., alpha / tmp)
            M = grad_g * scale
            M = M.reshape(*B.shape, order='F')
            #assert linalg.norm(M, 2) <= alpha + 1e-7 # dual feasible
            pobj = trace_pobj(X, y, B, alpha, beta, s_vals)
            p, conj0 = conj_loss(X, y, Xy, M, beta, conj0)
            dual_gap = pobj + p
            # because we computed conj_loss approximately, dual_gap might happen to be negative
            dual_gap = np.abs(dual_gap)
            if verbose:
                print('Dual gap: %s' % dual_gap)
            gap.append(dual_gap)
            if np.abs(dual_gap) <= rtol:
                break

    return B, gap
