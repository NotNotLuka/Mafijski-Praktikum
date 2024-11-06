#!/usr/bin/env python3

import cupy as np
import matplotlib.pyplot as plt
import sys

from cupy import linalg as LA

# A je mxn matrika
# Q je mxm ortogonalna
# R=rezultat, koncna A je 'zgornje' trikotna


def qr(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A


def make_householder(a):
    # rescaling to v and sign for numerical stability
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.outer(v, v)
    return H


def qr_givens(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for j in range(n - (m == n)):
        for i in range(j + 1, m):
            r = np.hypot(A[j, j], A[i, j])
            c = A[j, j] / r
            s = A[i, j] / r
            givensRot = np.array([[c, s], [-s, c]])
            A[[j, i], j:] = np.dot(givensRot, A[[j, i], j:])
            Q[[j, i], :] = np.dot(givensRot, Q[[j, i], :])
    return Q.T, A


def trid_householder(M):
    A = np.copy(M)
    m, n = A.shape
    if m != n:
        print("need quadratic symmetric matrix")
        sys.exit(1)
    Q = np.eye(m)
    for i in range(m - 2):
        H = np.eye(m)
        H[i + 1 :, i + 1 :] = make_householder(A[i + 1 :, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
        A = np.dot(A, H)
    return Q, A


def qlnr(d, e, z, tol=1.0e-9):
    # d - diagonal values
    # e - off-tridiag values
    # z - orthogonal matrix to process further
    n = len(d)
    e = np.roll(e, -1)  # reorder
    itmax = 1000
    for l in range(n):
        for iter in range(itmax):
            m = n - 1
            for mm in range(l, n - 1):
                dd = abs(d[mm]) + abs(d[mm + 1])
                if abs(e[mm]) + dd == dd:
                    m = mm
                    break
                if abs(e[mm]) < tol:
                    m = mm
                    break
            if iter == itmax - 1:
                print("too many iterations", iter)
                sys.exit(0)
            if m != l:
                g = (d[l + 1] - d[l]) / (2.0 * e[l])
                r = np.sqrt(g * g + 1.0)
                g = d[m] - d[l] + e[l] / (g + np.sign(g) * r)
                s = 1.0
                c = 1.0
                p = 0.0
                for i in range(m - 1, l - 1, -1):
                    f = s * e[i]
                    b = c * e[i]
                    if abs(f) > abs(g):
                        c = g / f
                        r = np.sqrt(c * c + 1.0)
                        e[i + 1] = f * r
                        s = 1.0 / r
                        c *= s
                    else:
                        s = f / g
                        r = np.sqrt(s * s + 1.0)
                        e[i + 1] = g * r
                        c = 1.0 / r
                        s *= c
                    g = d[i + 1] - p
                    r = (d[i] - g) * s + 2.0 * c * b
                    p = s * r
                    d[i + 1] = g + p
                    g = c * r - b
                    for k in range(n):
                        f = z[k, i + 1]
                        z[k, i + 1] = s * z[k, i] + c * f
                        z[k, i] = c * z[k, i] - s * f
                d[l] -= p
                e[l] = g
                e[m] = 0.0
            else:
                break
    return d, z


def get_eigenvalues(a):

    Q, Trid = trid_householder(a)

    n = Trid.shape[0]
    d = np.zeros(n)
    e = np.zeros(n)
    for i in range(n):
        d[i] = Trid[i, i]
    for i in range(n - 1):
        e[i + 1] = Trid[i + 1, i]

    lambda_d, Q_fin = qlnr(d, e, Q)
    return sorted(lambda_d), Q_fin
