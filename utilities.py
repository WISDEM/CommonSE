#!/usr/bin/env python
# encoding: utf-8
"""
utilities.py

Created by Andrew Ning on 2013-05-31.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from openmdao.main.api import VariableTree
from openmdao.main.datatypes.api import Array

from csystem import DirectionVector



def cosd(value):
    """cosine of value where value is given in degrees"""

    return np.cos(np.radians(value))


def sind(value):
    """sine of value where value is given in degrees"""

    return np.sin(np.radians(value))


def tand(value):
    """tangent of value where value is given in degrees"""

    return np.tan(np.radians(value))



class Vector(VariableTree):

    x = Array()
    y = Array()
    z = Array()

    def toDirVec(self):
        return DirectionVector(self.x, self.y, self.z)


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


