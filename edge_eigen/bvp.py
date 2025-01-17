#!/usr/bin/env python

"""A variety of methods to solve ODE boundary value problems.

AUTHOR:
    Jonathan Senning <jonathan.senning@gordon.edu>
    Gordon College
    Based Octave functions written in the spring of 1999
    Python version: October-November 2008
"""

import numpy
from diffeq import rku4 as rk4

# -----------------------------------------------------------------------------


def shoot(f, a, b, z1, z2, t, tol):
    """Implements the shooting method to solve second order BVPs

    USAGE:
        y = shoot(f, a, b, z1, z2, t, tol)

    INPUT:
        f     - function dy/dt = f(y,t).  Since we are solving a second-
                order boundary-value problem that has been transformed
                into a first order system, this function should return a
                1x2 array with the first entry equal to y and the second
                entry equal to y'.
        a     - solution value at the left boundary: a = y(t[0]).
        b     - solution value at the right boundary: b = y(t[n-1]).
        z1    - first initial estimate of y'(t[0]).
        z1    - second initial estimate of y'(t[0]).
        t     - array of n time values to determine y at.
        tol   - allowable tolerance on right boundary: | b - y[n-1] | < tol

    OUTPUT:
        y     - array of solution function values corresponding to the
                values in the supplied array t.

    NOTE:
        This function assumes that the second order BVP has been converted to
        a first order system of two equations.  The secant method is used to
        refine the initial values of y' used for the initial value problems.
    """

    max_iter = 10  # Maximum number of shooting iterations

    n = len(t)  # Determine the size of the arrays we will generate

    # Compute solution to first initial value problem (IVP) with y'(a) = z1.
    # Because we are using the secant method to refine our estimates of z =
    # y', we don't really need all the solution of the IVP, just the last
    # point of it -- this is saved in w1.

    y = rk4(f, [a, z1], t)
    w1 = y[n - 1, 0]

    # print("%2d: z = %10.3e, error = %10.3e" % (0, z1, b - w1))

    # Begin the main loop.  We will compute the solution of a second IVP and
    # then use the both solutions to refine our estimate of y'(a).  This
    # second solution then replaces the first and a new "second" solution is
    # generated.  This process continues until we either solve the problem to
    # within the specified tolerance or we exceed the maximum number of
    # allowable iterations.

    for i in range(max_iter):

        # Solve second initial value problem, using y'(a) = z2.  We need to
        # retain the entire solution vector y since if y(t(n)) is close enough
        # to b for us to stop then the first column of y becomes our solution
        # vector.

        y = rk4(f, [a, z2], t)
        w2 = y[n - 1, 0]

        # print("%2d: z = %10.3e, error = %10.3e" % (i + 1, z2, b - w2))

        # Check to see if we are done...

        if numpy.abs(b - w2) < tol:
            break

        # Compute the new approximations to the initial value of the first
        # derivative.  We compute z2 using a linear fit through (z1,w1) and
        # (z2,w2) where w1 and w2 are the estimates at t=b of the initial
        # value problems solved above with y1'(a) = z1 and y2'(a) = z2.  The
        # new value for z1 is the old value of z2.

        # z1, z2 = (z2, z1 + (z2 - z1) / (w2 - w1) * (b - w1))
        z1, z2 = (z2, z2 + 0.1 * (z2 - z1) / (w2 - w1) * (b - w2))
        w1 = w2

    # All done.  Check to see if we really solved the problem, and then return
    # the solution.

    return y[:, 0]


# -----------------------------------------------------------------------------


def fd(u, v, w, t, a, b):
    """Implements the finite difference  method to solve linear second order BVPs

    Compute finite difference solution to the BVP

        x'' = u(t) + v(t) x + w(t) x'
        x(t[0]) = a, x(t[n-1]) = b

    t should be passed in as an n element array.   u, v, and w should be
    either n element arrays corresponding to u(t), v(t) and w(t) or
    scalars, in which case an n element array with the given value is
    generated for each of them.

    USAGE:
        x = fd(u, v, w, t, a, b)

    INPUT:
        u,v,w - arrays containing u(t), v(t), and w(t) values.  May be
                specified as Python lists, NumPy arrays, or scalars.  In
                each case they are converted to NumPy arrays.
        t     - array of n time values to determine x at
        a     - solution value at the left boundary: a = x(t[0])
        b     - solution value at the right boundary: b = x(t[n-1])

    OUTPUT:
        x     - array of solution function values corresponding to the
                values in the supplied array t.
    """

    # Get the dimension of t and make sure that t is an n-element vector

    if type(t) is not numpy.ndarray:
        if type(t) is list:
            t = numpy.array(t)
        else:
            t = numpy.array([float(t)])

    n = len(t)

    # Make sure that u, v, and w are either scalars or n-element vectors.
    # If they are scalars then we create vectors with the scalar value in
    # each position.

    if type(u) is int or type(u) is float:
        u = numpy.array([float(u)] * n)

    if type(v) is int or type(v) is float:
        v = numpy.array([float(v)] * n)

    if type(w) is int or type(w) is float:
        w = numpy.array([float(w)] * n)

    # Compute the stepsize.  It is assumed that all elements in t are
    # equally spaced.

    h = t[1] - t[0]

    # Construct tridiagonal system; boundary conditions appear as first and
    # last equations in system.

    A = -(1.0 + w[1:n] * h / 2.0)
    A[-1] = 0.0

    C = -(1.0 - w[0:n - 1] * h / 2.0)
    C[0] = 0.0

    D = 2.0 + h * h * v
    D[0] = D[n - 1] = 1.0

    B = -h * h * u
    B[0] = a
    B[n - 1] = b

    # Solve tridiagonal system

    for i in range(1, n):
        xmult = A[i - 1] / D[i - 1]
        D[i] = D[i] - xmult * C[i - 1]
        B[i] = B[i] - xmult * B[i - 1]

    x = numpy.zeros(n)
    x[n - 1] = B[n - 1] / D[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (B[i] - C[i] * x[i + 1]) / D[i]

    return x


def shoot_modified(f, a, b, z1, z2, t, tol):
    """Implements the shooting method to solve second order BVPs

    USAGE:
        y = shoot(f, a, b, z1, z2, t, tol)

    INPUT:
        f     - function dy/dt = f(y,t).  Since we are solving a second-
                order boundary-value problem that has been transformed
                into a first order system, this function should return a
                1x2 array with the first entry equal to y and the second
                entry equal to y'.
        a     - solution value at the left boundary: a = y(t[0]).
        b     - solution value at the right boundary: b = y(t[n-1]).
        z1    - first initial estimate of y'(t[0]).
        z1    - second initial estimate of y'(t[0]).
        t     - array of n time values to determine y at.
        tol   - allowable tolerance on right boundary: | b - y[n-1] | < tol

    OUTPUT:
        y     - array of solution function values corresponding to the
                values in the supplied array t.

    NOTE:
        This function assumes that the second order BVP has been converted to
        a first order system of two equations.  The secant method is used to
        refine the initial values of y' used for the initial value problems.
    """

    max_iter = 10  # Maximum number of shooting iterations

    n = len(t)  # Determine the size of the arrays we will generate

    # Compute solution to first initial value problem (IVP) with y'(a) = z1.
    # Because we are using the secant method to refine our estimates of z =
    # y', we don't really need all the solution of the IVP, just the last
    # point of it -- this is saved in w1.

    y = rk4(f, [a, z1], t)
    w1 = y[n - 1, 0]

    # print("%2d: z = %10.3e, error = %10.3e" % (0, z1, b - w1))

    # Begin the main loop.  We will compute the solution of a second IVP and
    # then use the both solutions to refine our estimate of y'(a).  This
    # second solution then replaces the first and a new "second" solution is
    # generated.  This process continues until we either solve the problem to
    # within the specified tolerance or we exceed the maximum number of
    # allowable iterations.

    for i in range(max_iter):

        # Solve second initial value problem, using y'(a) = z2.  We need to
        # retain the entire solution vector y since if y(t(n)) is close enough
        # to b for us to stop then the first column of y becomes our solution
        # vector.

        y = rk4(f, [a, z2], t)
        w2 = y[n - 1, 0]

        # print("%2d: z = %10.3e, error = %10.3e" % (i + 1, z2, b - w2))

        # Check to see if we are done...

        if numpy.abs(b - w2) < tol:
            break

        # Compute the new approximations to the initial value of the first
        # derivative.  We compute z2 using a linear fit through (z1,w1) and
        # (z2,w2) where w1 and w2 are the estimates at t=b of the initial
        # value problems solved above with y1'(a) = z1 and y2'(a) = z2.  The
        # new value for z1 is the old value of z2.

        # z1, z2 = (z2, z1 + (z2 - z1) / (w2 - w1) * (b - w1))
        z1, z2 = (z2, z2 + 0.1 * (z2 - z1) / (w2 - w1) * (b - w2))
        w1 = w2

    # All done.  Check to see if we really solved the problem, and then return
    # the solution.

    return y[:, 0]






