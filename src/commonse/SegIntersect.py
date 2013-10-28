#-------------------------------------------------------------------------
# Name:        SegIntersect.py
# Purpose: It Calculates Intersection of 2 segments in space
#
# Author:      RRD
#
# Created:     24/10/2012
# Copyright:   (c) rdamiani 2012
# Licence:     <your licence>
#-------------------------------------------------------------------------


import numpy as np


def SegIntersect(A1, A2, B1, B2):
    """The function returns the intersection or the points of closest approach if lines are skewed.
    If lines are parallel, NaN is returned.
    INPUT:
        A1  -float(3,n), [x,y,z;nsegments] cordinates of 1st point(s) of 1st segment(s)
        A2  -float(3,n), [x,y,z;nsegments] cordinates of 2nd point(s) of 1st segment(s)
        B1  -float(3,n), [x,y,z;nsegments] cordinates of 1st point(s) of 2nd segment(s)
        B2  -float(3,n), [x,y,z;nsegments] cordinates of 2nd point(s) of 2nd segment(s)
    OUTPUT:
        A0  -float(3,n), [x,y,z;nsegments] coordinates of intersection point (=B0) or closet point to 2nd line on 1st segment,
        B0  -float(3,n), [x,y,z;nsegments] coordinates of intersection point (=A0) or closet point to 2nd line on 1st segment,
        OR  -NaN
    """


    vec = np.cross(A2 - A1, B2 - B1, 0, 0, 0)
    nA = np.sum(np.cross(B2 - B1, A1 - B1, 0, 0, 0) * vec, axis=0)
    nB = np.sum(np.cross(A2 - A1, B1 - A1, 0, 0, 0) * vec, axis=0)
    d = np.sum(vec**2, axis=0)

    A0 = np.ones(A1.shape) * np.NaN
    B0 = A0.copy()
    idx = np.nonzero(d)[0]
    A0[:, idx] = A1[:, idx] + (nA[idx] / d[idx]) * (A2[:, idx] - A1[:, idx])
    B0[:, idx] = B1[:, idx] + (nB[idx] / d[idx]) * (B2[:, idx] - B1[:, idx])

    return A0, B0

