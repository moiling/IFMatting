import inspect

import scipy
from scipy.sparse import csr_matrix as csr, diags
import numpy as np

from utils.pcg import solve_cg
from utils.utils import affinityMatrixToLaplacian


def solve_alpha_2(w_cm, w_uu, w_l, n_p, tau, a_k, w_f, params):
    d_cm = diags(csr.sum(w_cm, axis=1).A.ravel(), format='csr')
    d_uu = diags(csr.sum(w_uu, axis=1).A.ravel(), format='csr')
    d_l = diags(csr.sum(w_l, axis=1).A.ravel(), format='csr')

    # (15)
    l_ifm = (d_cm - w_cm).T.dot(d_cm - w_cm) + params['s_uu'] * (d_uu - w_uu) + params['s_l'] * (d_l - w_l)

    if params['use_k_u']:
        # (16)
        A = l_ifm + params['lambda'] * tau + params['s_ku'] * n_p
        b = (params['lambda'] * tau + params['s_ku'] * n_p).dot(w_f)
    else:
        # (19)
        A = l_ifm + params['lambda'] * tau
        b = params['lambda'] * tau * a_k

    # use preconditioned conjugate gradient to solve the linear systems
    solution = scipy.sparse.linalg.cg(A, b, x0=None, tol=1e-10, maxiter=10000, M=None, callback=report)
    return solution[0]


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
    solution = solve_cg(A, b, x0=None, rtol=1e-7, max_iter=2000, print_info=True)
    return solution


def report(x):
    frame = inspect.currentframe().f_back
    print('%4d: %e' % (frame.f_locals['iter_'], frame.f_locals['resid']))
