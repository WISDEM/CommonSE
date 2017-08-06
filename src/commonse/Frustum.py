#-------------------------------------------------------------------------
# Name:        SegIntersect.py
# Purpose: It Calculates Intersection of 2 segments in space. Also contains a simple 2-point distance calculator.
#
# Author:      RRD
#
# Created:     24/10/2012
# Copyright:   (c) rdamiani 2012
# Licence:     <your licence>
#-------------------------------------------------------------------------


import numpy as np

def frustum(Db, Dt, H):
    """This function returns a frustum's volume and center of mass, CM

    INPUT:
    Parameters
    ----------
    Db : float,        base diameter
    Dt : float,        top diameter
    H : float,         height

    OUTPUTs:
    -------
    vol : float,        volume
    cm : float,        geometric centroid relative to bottom (center of mass if uniform density)

    """

    vol = np.pi/12*H * (Db**2 + Dt**2 + Db * Dt)
    cm = H/4 * (Db**2 + 3*Dt**2 + 2*Db*Dt) / (Db**2 + Dt**2 + Db*Dt)

    return vol, cm


def frustumVol_radius(rb, rt, h):
    """This function returns a frustum's volume with radii inputs

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    vol : float (scalar/vector), volume
    """
    return ( np.pi * (h/3.0) * (rb*rb + rt*rt + rb*rt) )

def frustumVol_diameter(db, dt, h):
    """This function returns a frustum's volume with diameter inputs

    INPUTS:
    Parameters
    ----------
    db : float (scalar/vector),  base diameter
    dt : float (scalar/vector),  top diameter
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    vol : float (scalar/vector), volume
    """
    return frustumVol_radius(0.5*db, 0.5*dt, h)

def frustumCG_radius(rb, rt, h):
    """This function returns a frustum's center of mass/gravity (centroid) with radii inputs.
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    return (0.25*h * (rb**2 + 2.*rb*rt + 3.*rt**2) / (rb**2 + rb*rt + rt**2))

def frustumCG_diameter(db, dt, h):
    """This function returns a frustum's center of mass/gravity (centroid) with diameter inputs.
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    db : float (scalar/vector),  base diameter
    dt : float (scalar/vector),  top diameter
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    return frustumCG_radius(0.5*db, 0.5*dt, h)

def frustumShellCG_radius(rb, rt, h):
    """This function returns a frustum's center of mass/gravity (centroid) with radii inputs.
    NOTE: This is for a frustum SHELL, not a solid
    NOTE: Input radius here should be average shell radius (assuming wall thickness t<<r) R-0.5*t or (Ro+Ri)/2

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    return (h/3 * (rb + 2.*rt) / (rb + rt))

def frustumShellCG_diameter(db, dt, h):
    """This function returns a frustum's center of mass/gravity (centroid) with diameter inputs.
    NOTE: This is for a frustum SHELL, not a solid
    NOTE: Input diameter here should be average shell diameter (assuming wall thickness t<<d) D-t or (Do+Di)/2

    INPUTS:
    Parameters
    ----------
    db : float (scalar/vector),  base diameter
    dt : float (scalar/vector),  top diameter
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    return frustumShellCG_radius(0.5*db, 0.5*dt, h)


if __name__ == '__main__':
    Db=6.5
    Dt=4.
    H=120.

    print ('From commonse.Frustum: Sample Volume and CM of FRUSTUM='+4*'{:8.4f}, ').format(*frustum(Db,Dt,H)[0].flatten())

def main():
    pass
