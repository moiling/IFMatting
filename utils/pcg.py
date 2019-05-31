import numpy as np


def solve_cg(
    A,
    b,
    rtol,
    max_iter,
    atol=0.0,
    x0=None,
    precondition=None,
    callback=None,
    print_info=False,
):
    """
    Solve the linear system Ax = b for x using preconditioned conjugate
    gradient descent.

    A: np.ndarray of dtype np.float64
        Must be a square symmetric matrix
    b: np.ndarray of dtype np.float64
        Right-hand side of linear system
    rtol: float64
        Conjugate gradient descent will stop when
        norm(A x - b) < relative_tolerance norm(b)
    max_iter: int
        Maximum number of iterations
    atol: float64
        Conjugate gradient descent will stop when
        norm(A x - b) < absolute_tolerance
    x0: np.ndarray of dtype np.float64
        Initial guess of solution x
    precondition: func(r) -> r
        Improve solution of residual r, for example solve(M, r)
        where M is an easy-to-invert approximation of A.
    callback: func(A, x, b)
        callback to inspect temporary result after each iteration.
    print_info: bool
        If to print convergence information.

    Returns
    -------

    x: np.ndarray of dtype np.float64
        Solution to the linear system Ax = b.

    """

    x = np.zeros(A.shape[0]) if x0 is None else x0.copy()

    if callback is not None:
        callback(A, x, b)

    if precondition is None:
        def precondition(r):
            return r

    norm_b = np.linalg.norm(b)

    r = b - A.dot(x)
    z = precondition(r)
    p = z.copy()
    rz = np.inner(r, z)
    for iteration in range(max_iter):
        Ap = A.dot(p)
        alpha = rz / np.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        residual_error = np.linalg.norm(r)

        if print_info:
            print("iteration %05d - residual error %e" % (
                iteration,
                residual_error))

        if callback is not None:
            callback(A, x, b)

        if residual_error < atol or residual_error < rtol:
            break

        z = precondition(r)
        beta = 1.0 / rz
        rz = np.inner(r, z)
        beta *= rz
        p *= beta
        p += z

    return x
