#!/usr/bin/env python
# encoding: utf-8
"""
utilities.py

Created by Andrew Ning on 2013-05-31.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np


def cosd(value):
    """cosine of value where value is given in degrees"""

    return np.cos(np.radians(value))


def sind(value):
    """sine of value where value is given in degrees"""

    return np.sin(np.radians(value))


def tand(value):
    """tangent of value where value is given in degrees"""

    return np.tan(np.radians(value))


def hstack(vec):
    """stack arrays horizontally.  useful for assemblying Jacobian"""
    return np.vstack(vec).T


def vstack(vec):
    """for consistently also provide the vertical stack"""
    return np.vstack(vec)



class CubicSpline(object):

    def __init__(self, x1, x2, f1, f2, g1, g2):

        self.x1 = x1
        self.x2 = x2

        self.A = np.array([[x1**3, x1**2, x1, 1.0],
                  [x2**3, x2**2, x2, 1.0],
                  [3*x1**2, 2*x1, 1.0, 0.0],
                  [3*x2**2, 2*x2, 1.0, 0.0]])
        self.b = np.array([f1, f2, g1, g2])

        self.coeff = np.linalg.solve(self.A, self.b)

        self.poly = np.polynomial.Polynomial(self.coeff[::-1])


    def eval(self, x):
        return self.poly(x)


    def eval_deriv(self, x):
        polyd = self.poly.deriv()
        return polyd(x)


    def eval_deriv_params(self, xvec, dx1, dx2, df1, df2, dg1, dg2):

        x1 = self.x1
        x2 = self.x2
        dA_dx1 = np.matrix([[3*x1**2, 2*x1, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [6*x1, 2.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])
        dA_dx2 = np.matrix([[0.0, 0.0, 0.0, 0.0],
                  [3*x2**2, 2*x2, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [6*x2, 2.0, 0.0, 0.0]])
        df = np.array([df1, df2, dg1, dg2])
        c = np.matrix(self.coeff).T

        n = len(xvec)
        dF = np.zeros(n)
        for i in range(n):
            x = np.array([xvec[i]**3, xvec[i]**2, xvec[i], 1.0])
            d = np.linalg.solve(self.A.T, x)
            dF_dx1 = -d*dA_dx1*c
            dF_dx2 = -d*dA_dx2*c
            dF_df = np.linalg.solve(self.A.T, x)
            dF[i] = np.dot(dF_df, df) + dF_dx1[0]*dx1 + dF_dx2[0]*dx2

        return dF



def _getvar(comp, name):
    vars = name.split('.')
    base = comp
    for var in vars:
        base = getattr(base, var)

    return base


def check_gradient(comp, fd='central', step_size=1e-6, tol=1e-6, display=False):

    comp.run()
    comp.linearize()
    inputs, outputs, J = comp.provideJ()

    # compute size of Jacobian
    m = 0
    mvec = []  # size of each output
    cmvec = []  # cumulative size of outputs
    nvec = []  # size of each input
    cnvec = []  # cumulative size of inputs
    for out in outputs:
        f = _getvar(comp, out)
        if np.array(f).shape == ():
            msub = 1
        else:
            msub = len(f)
        m += msub
        mvec.append(msub)
        cmvec.append(m)
    n = 0
    for inp in inputs:
        x = _getvar(comp, inp)
        if np.array(x).shape == ():
            nsub = 1
        else:
            nsub = len(x)
        n += nsub
        nvec.append(nsub)
        cnvec.append(n)

    JFD = np.zeros((m, n))

    if J.shape != JFD.shape:
        raise TypeError('Incorrect Jacobian size. Your provided Jacobian is of shape {}, but it should be ({}, {})'.format(J.shape, m, n))


    # initialize start and end indices of where to insert into Jacobian
    m1 = 0
    m2 = 0


    for i, out in enumerate(outputs):

        # get function value at center
        f = _getvar(comp, out)
        if np.array(f).shape == ():
            lenf = 1
        else:
            f = np.copy(f)  # so not pointed to same memory address
            lenf = len(f)

        m2 += lenf

        n1 = 0

        for j, inp in enumerate(inputs):

            # get x value at center (save location)
            x = _getvar(comp, inp)
            if np.array(x).shape == ():
                x0 = x
                lenx = 1
            else:
                x = np.copy(x)  # so not pointing to same memory address
                x0 = np.copy(x)
                lenx = len(x)

            for k in range(lenx):

                # take a step
                if lenx == 1:
                    h = step_size*x
                    if h == 0:
                        h = step_size
                    x += h
                else:
                    h = step_size*x[k]
                    if h == 0:
                        h = step_size
                    x[k] += h
                setattr(comp, inp, x)
                comp.run()

                # fd
                fp = np.copy(_getvar(comp, out))

                if fd == 'central':

                    # step back
                    if lenx == 1:
                        x -= 2*h
                    else:
                        x[k] -= 2*h
                    setattr(comp, inp, x)
                    comp.run()

                    fm = np.copy(_getvar(comp, out))

                    deriv = (fp - fm)/(2*h)

                else:
                    deriv = (fp - f)/h


                JFD[m1:m2, n1+k] = deriv

                # reset state
                x = np.copy(x0)
                setattr(comp, inp, x0)
                comp.run()

            n1 += lenx

        m1 = m2

    # error checking
    namevec = []
    errorvec = []

    if display:
        print '{:<20} ({}) {:<10} ({}, {})'.format('error', 'errortype', 'name', 'analytic', 'fd')
        print

    for i in range(m):
        for j in range(n):

            # get corresonding variables names
            for ii in range(len(mvec)):
                if cmvec[ii] > i:
                    oname = 'd_' + outputs[ii]

                    if mvec[ii] > 1:  # need to print indices
                        subtract = 0
                        if ii > 0:
                            subtract = cmvec[ii-1]
                        idx = i - subtract
                        oname += '[' + str(idx) + ']'

                    break
            for jj in range(len(nvec)):
                if cnvec[jj] > j:
                    iname = 'd_' + inputs[jj]

                    if nvec[jj] > 1:  # need to print indices
                        subtract = 0
                        if jj > 0:
                            subtract = cnvec[jj-1]
                        idx = j - subtract
                        iname += '[' + str(idx) + ']'

                    break
            name = oname + ' / ' + iname

            # compute error
            if np.abs(J[i, j]) <= tol:
                errortype = 'absolute'
                error = J[i, j] - JFD[i, j]
            else:
                errortype = 'relative'
                error = 1.0 - JFD[i, j]/J[i, j]
            error = np.abs(error)

            # display
            if error > tol:
                star = ' ***** '
            else:
                star = ''

            if display:
                output = '{}{:<20} ({}) {}: ({}, {})'.format(star, error, errortype, name, J[i, j], JFD[i, j])
                print output

            # save
            namevec.append(name)
            errorvec.append(error)

    return namevec, errorvec



