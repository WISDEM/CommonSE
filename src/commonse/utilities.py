#!/usr/bin/env python
# encoding: utf-8
"""
utilities.py

Created by Andrew Ning on 2013-05-31.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
# from openmdao.main.api import VariableTree
# from openmdao.main.datatypes.api import Array

# from csystem import DirectionVector



def cosd(value):
    """cosine of value where value is given in degrees"""

    return np.cos(np.radians(value))


def sind(value):
    """sine of value where value is given in degrees"""

    return np.sin(np.radians(value))


def tand(value):
    """tangent of value where value is given in degrees"""

    return np.tan(np.radians(value))



# class Vector(VariableTree):

#     x = Array()
#     y = Array()
#     z = Array()

#     def toDirVec(self):
#         return DirectionVector(self.x, self.y, self.z)


def cubicSpline(x1, x2, f1, f2, g1, g2, xeval=None):


    A = np.array([[x1**3, x1**2, x1, 1.0],
                  [x2**3, x2**2, x2, 1.0],
                  [3*x1**2, 2*x1, 1.0, 0.0],
                  [3*x2**2, 2*x2, 1.0, 0.0]])
    b = np.array([f1, f2, g1, g2])

    coeff = np.linalg.solve(A, b)

    if xeval is None:
        return coeff
    else:
        return np.polyval(coeff, xeval)



# class MassMomentInertia(VariableTree):

#     xx = Float(units='kg*m**2', desc='mass moment of inertia about x-axis')
#     yy = Float(units='kg*m**2', desc='mass moment of inertia about y-axis')
#     zz = Float(units='kg*m**2', desc='mass moment of inertia about z-axis')
#     xy = Float(units='kg*m**2', desc='mass x-y product of inertia')
#     xz = Float(units='kg*m**2', desc='mass x-z product of inertia')
#     yz = Float(units='kg*m**2', desc='mass y-z product of inertia')


def check_gradient(comp, fd='central', step_size=1e-6, tol=1e-6, display=False):


    comp.run()
    comp.linearize()
    inputs, outputs, J = comp.provideJ()
    print J.shape

    # compute size of Jacobian
    m = 0
    mvec = []
    cmvec = []
    nvec = []
    cnvec = []
    for out in outputs:
        f = getattr(comp, out)
        if np.array(f).shape == ():
            msub = 1
        else:
            msub = len(f)
        m += msub
        mvec.append(msub)
        cmvec.append(m)
    n = 0
    for inp in inputs:
        x = getattr(comp, inp)
        if np.array(x).shape == ():
            nsub = 1
        else:
            nsub = len(x)
        n += nsub
        nvec.append(nsub)
        cnvec.append(n)

    JFD = np.zeros((m, n))
    print m, n
    exit()

    # initialize indices
    m1 = 0
    m2 = 0


    for i, out in enumerate(outputs):

        # get function value at center
        f = getattr(comp, out)
        if np.array(f).shape == ():
            lenf = 1
        else:
            f = np.copy(f)  # so not pointed to same memory address
            lenf = len(f)

        m2 += lenf

        n1 = 0

        for j, inp in enumerate(inputs):

            # get x value at center (save location)
            x = getattr(comp, inp)
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
                fp = np.copy(getattr(comp, out))

                if fd == 'central':

                    # step back
                    if lenx == 1:
                        x -= 2*h
                    else:
                        x[k] -= 2*h
                    setattr(comp, inp, x)
                    comp.run()

                    fm = np.copy(getattr(comp, out))

                    deriv = (fp - fm)/(2*h)

                else:
                    deriv = (fp - f)/h


                JFD[m1:m2, n1+k] = deriv

                # reset state
                setattr(comp, inp, x0)
                comp.run()

            n1 += lenx

        m1 = m2

    # error checking
    namevec = []
    errorvec = []

    if display:
        print '{:<20} ({}) {:<20} ({}, {})'.format('error', 'errortype', 'name', 'analytic', 'fd')
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
            print i, j
            print m, n
            print J.shape
            print J[i, j]
            if J[i, j] <= tol:
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



