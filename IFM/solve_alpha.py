import inspect

import scipy
from scipy.sparse import csr_matrix as csr, diags
import numpy as np

from utils.utils import affinity_matrix_to_laplacian, solve_cg


def solve_alpha(trimap, w_cm, w_uu, w_l, h, a_k, w_f, params):
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)

    L = affinity_matrix_to_laplacian(w_cm)
    L = params['s_cm'] * L.T.dot(L)
    L = L + params['s_l'] * affinity_matrix_to_laplacian(w_l)
    L = L + params['s_uu'] * affinity_matrix_to_laplacian(w_uu)
    d = is_known.flatten().astype(np.float64)
    A = params['lambda'] * diags(d)

    if params['use_k_u']:
        A = A + params['s_ku'] * diags(h.flatten())
        b = A.dot(w_f.flatten())
    else:
        b = A.dot(a_k)

    A = A + L

    # use preconditioned conjugate gradient to solve the linear systems
    # solution = scipy.sparse.linalg.cg(A, b, x0=None, tol=1e-6, maxiter=2000, M=None, callback=None)
    # return solution[0]
    inv_diag = 1 / A.diagonal()

    def precondition(r):
        return r * inv_diag

    solution = solve_cg(
        A,
        b,
        max_iter=2000,
        rtol=1e-6,
        precondition=precondition)

    return solution


def report(x):
    frame = inspect.currentframe().f_back
    print('%4d: %e' % (frame.f_locals['iter_'], frame.f_locals['resid']))
