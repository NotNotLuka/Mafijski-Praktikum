#!/usr/bin/env python

"""A variety of methods to solve first order ordinary differential equations.

AUTHOR:
    Jonathan Senning <jonathan.senning@gordon.edu>
    Gordon College
    Based Octave functions written in the spring of 1999
    Python version: March 2008, October 2008
"""

import numpy

# -----------------------------------------------------------------------------


def rku4(f, x0, t):
    """Fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rku4(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len(t)
    x0 = numpy.array(x0, dtype=numpy.float64)
    x = numpy.zeros((n, x0.shape[0]))
    x[0] = x0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = h * f(x[i], t[i])
        k2 = h * f(x[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(x[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(x[i] + k3, t[i + 1])
        x[i + 1] = x[i] + (k1 + 2.0 * (k2 + k3) + k4) / 6.0

    return x
