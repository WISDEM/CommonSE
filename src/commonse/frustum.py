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
    vol = frustumVol(Db, Dt, H, diamFlag=True)
    cm  = frustumCG(Db, Dt, H, diamFlag=True)
    #vol = np.pi/12*H * (Db**2 + Dt**2 + Db * Dt)
    #cm = H/4 * (Db**2 + 3*Dt**2 + 2*Db*Dt) / (Db**2 + Dt**2 + Db*Dt)
    return vol, cm


def frustumVol(rb, rt, h, diamFlag=False):
    """This function returns a frustum's volume with radii or diameter inputs.

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    vol : float (scalar/vector), volume
    """
    if diamFlag:
        # Convert diameters to radii
        rb *= 0.5
        rt *= 0.5
    return ( np.pi * (h/3.0) * (rb*rb + rt*rt + rb*rt) )

def frustumCG(rb, rt, h, diamFlag=False):
    """This function returns a frustum's center of mass/gravity (centroid) with radii or diameter inputs.
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    if diamFlag:
        # Convert diameters to radii
        rb *= 0.5
        rt *= 0.5
    return (0.25*h * (rb**2 + 2.*rb*rt + 3.*rt**2) / (rb**2 + rb*rt + rt**2))

def frustumShellVolume(rb, rt, tb, tt, h, diamFlag=False):
    """This function returns a frustum shell's volume (for computing mass with density) with radii or diameter inputs.
    NOTE: This is for a frustum SHELL, not a solid
    NOTE: Input radius here should be average shell radius (assuming wall thickness t<<r) R-0.5*t or (Ro+Ri)/2

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    tb : float (scalar/vector),  base thickness
    tt : float (scalar/vector),  top thickness
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    if diamFlag:
        # Convert diameters to radii
        rb *= 0.5
        rt *= 0.5
    # Integrate 2*pi*r*t*dz from 0 to H
    return ( (np.pi*h/3.0) * ( rb*(2*tb + tt) + rt*(tb + 2*tt) ) )

def frustumShellCG(rb, rt, h, diamFlag=False):
    """This function returns a frustum's center of mass/gravity (centroid) with radii or diameter inputs.
    NOTE: This is for a frustum SHELL, not a solid
    NOTE: Input radius here should be average shell radius (assuming wall thickness t<<r) R-0.5*t or (Ro+Ri)/2

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    if diamFlag:
        # Convert diameters to radii
        rb *= 0.5
        rt *= 0.5
    return (h/3 * (rb + 2.*rt) / (rb + rt))


if __name__ == '__main__':
    Db=6.5
    Dt=4.
    H=120.

    print ('From commonse.Frustum: Sample Volume and CM of FRUSTUM='+4*'{:8.4f}, ').format(*frustum(Db,Dt,H)[0].flatten())

def main():
    pass
