import inspect

import scipy
from scipy.sparse import csr_matrix as csr, diags
import numpy as np

from utils.utils import affinityMatrixToLaplacian


def solve_alpha(trimap, w_cm, w_uu, w_l, kToUconf, tau, a_k, kToU, params):
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    L = affinityMatrixToLaplacian(w_cm)
    L = params['s_cm'] * L.T.dot(L)
    L = L + params['s_l'] * affinityMatrixToLaplacian(w_l)
    L = L + params['s_uu'] * affinityMatrixToLaplacian(w_uu)
    d = is_known.flatten().astype(np.float64)
    A = params['lambda'] * diags(d)

    if params['use_k_u']:
        A = A + params['s_ku'] * diags(kToUconf.flatten())
        b = A.dot(kToU.flatten())
    else:
        b = A.dot(a_k)

    A = A + L

    # use preconditioned conjugate gradient to solve the linear systems
    solution = scipy.sparse.linalg.cg(A, b, x0=None, tol=1e-10, maxiter=2000, M=None, callback=report)
    return solution[0]


def report(x):
    frame = inspect.currentframe().f_back
    print('%4d: %e' % (frame.f_locals['iter_'], frame.f_locals['resid']))
