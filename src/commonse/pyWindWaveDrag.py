#!/usr/bin/env python
# encoding: utf-8
"""
WindWaveDrag.py

Created by RRD on 2015-07-13.
Copyright (c) NREL. All rights reserved.
"""

#-------------------------------------------------------------------------------
# Name:        WindWaveDrag.py
# Purpose:     It contains OpenMDAO's Components to calculate wind or wave drag
#              on cylinders.
#
# Author:      ANing/RRD
#
# Created:     13/07/2015 - It is based on load function calculations developed for tower and jacket.
#                             Reestablished elements needed by jacketSE that were removed. Changed names to vartrees.
# Copyright:   (c) rdamiani 2015
# Licence:     <Apache 2015>
#-------------------------------------------------------------------------------
import math
import numpy as np


from commonse.utilities import sind, cosd  # , linspace_with_deriv, interp_with_deriv, hstack, vstack
from commonse.csystem import DirectionVector

from akima import Akima

#TODO CHECK

# -----------------
#  Helper Functions
# -----------------
# "Experiments on the Flow Past a Circular Cylinder at Very High Reynolds Numbers", Roshko
Re_pt = [0.00001, 0.0001, 0.0010, 0.0100, 0.0200, 0.1220, 0.2000, 0.3000, 0.4000,
         0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 5.0000, 10.0000]
cd_pt = [4.0000,  2.0000, 1.1100, 1.1100, 1.2000, 1.2000, 1.1700, 0.9000, 0.5400,
         0.3100, 0.3800, 0.4600, 0.5300, 0.5700, 0.6100, 0.6400, 0.6700, 0.7000, 0.7000]

drag_spline = Akima(np.log10(Re_pt), cd_pt, delta_x=0.0)  # exact akima because control points do not change

def cylinderDrag(Re):
    """Drag coefficient for a smooth circular cylinder.

    Parameters
    ----------
    Re : array_like
        Reynolds number

    Returns
    -------
    cd : array_like
        drag coefficient (normalized by cylinder diameter)

    """

    ReN = Re / 1.0e6

    cd = np.zeros_like(Re)
    dcd_dRe = np.zeros_like(Re)
    idx = ReN > 0
    cd[idx], dcd_dRe[idx] = drag_spline.interp(np.log10(ReN[idx]))
    dcd_dRe[idx] /= (Re[idx]*math.log(10))  # chain rule

    return cd, dcd_dRe

# -----------------
#  Components
# -----------------

class AeroHydroLoads(object):

    def __init__(self, nPoints):

        super(AeroHydroLoads, self).__init__()

        ##inputs
        # # self.add_param('windLoads_Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        # # self.add_param('windLoads_Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        # # self.add_param('windLoads_Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        # # self.add_param('windLoads_qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        # # self.add_param('windLoads_z', np.zeros(nPoints), units='m', desc='corresponding heights')
        # # self.add_param('windLoads_d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        # # self.add_param('windLoads_beta', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')

        # # self.add_param('waveLoads_Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        # # self.add_param('waveLoads_Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        # # self.add_param('waveLoads_Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        # # self.add_param('waveLoads_qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        # # self.add_param('waveLoads_z', np.zeros(nPoints), units='m', desc='corresponding heights')
        # # self.add_param('waveLoads_d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        # # self.add_param('waveLoads_beta', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')

        # # self.add_param('z', np.zeros(nPoints), units='m', desc='locations along cylinder')
        # # self.add_param('yaw', 0.0, units='deg', desc='yaw angle')

        #outputs
        self.Px = np.zeros(nPoints) # self.Px = np.zeros(nPoints) # self.add_output('Px', np.zeros(nPoints), units='N/m', desc='force per unit length in x-direction')
        self.Py = np.zeros(nPoints) # self.Py = np.zeros(nPoints) # self.add_output('Py', np.zeros(nPoints), units='N/m', desc='force per unit length in y-direction')
        self.Pz = np.zeros(nPoints) # self.Pz = np.zeros(nPoints) # self.add_output('Pz', np.zeros(nPoints), units='N/m', desc='force per unit length in z-direction')
        self.qdyn = np.zeros(nPoints) # self.qdyn = np.zeros(nPoints) # self.add_output('qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')

    def compute(self, windLoads_Px, windLoads_Py, windLoads_Pz, windLoads_qdyn, windLoads_z, windLoads_d, windLoads_beta, waveLoads_Px, waveLoads_Py,
                waveLoads_Pz, waveLoads_qdyn, waveLoads_z, waveLoads_d, waveLoads_beta, z, yaw):
        # aero/hydro loads
        z = z
        hubHt = z[-1]  # top of cylinder
        windLoads = DirectionVector(windLoads_Px, windLoads_Py, windLoads_Pz).inertialToWind(windLoads_beta).windToYaw(yaw)
        waveLoads = DirectionVector(waveLoads_Px, waveLoads_Py, waveLoads_Pz).inertialToWind(waveLoads_beta).windToYaw(yaw)

        self.Px = np.interp(z, windLoads_z, windLoads.x) + np.interp(z, waveLoads_z, waveLoads.x)
        self.Py = np.interp(z, windLoads_z, windLoads.y) + np.interp(z, waveLoads_z, waveLoads.y)
        self.Pz = np.interp(z, windLoads_z, windLoads.z) + np.interp(z, waveLoads_z, waveLoads.z)
        self.qdyn = np.interp(z, windLoads_z, windLoads_qdyn) + np.interp(z, waveLoads_z, waveLoads_qdyn)

# -----------------

class CylinderWindDrag(object):
    """drag forces on a cylindrical cylinder due to wind"""

    def __init__(self, nPoints):

        super(CylinderWindDrag, self).__init__()

        # # variables
        # # self.add_param('U', np.zeros(nPoints), units='m/s', desc='magnitude of wind speed')
        # # self.add_param('z', np.zeros(nPoints), units='m', desc='heights where wind speed was computed')
        # # self.add_param('d', np.zeros(nPoints), units='m', desc='corresponding diameter of cylinder section')

        # # parameters
        # # self.add_param('beta', 0.0, units='deg', desc='corresponding wind angles relative to inertial coordinate system')
        # # self.add_param('rho', 0.0, units='kg/m**3', desc='air density')
        # # self.add_param('mu', 0.0, units='kg/(m*s)', desc='dynamic viscosity of air')
        # #TODO not sure what to do here?
        # # self.add_param('cd_usr', np.inf, desc='User input drag coefficient to override Reynolds number based one')

        # out
        self.windLoads_Px = np.zeros(nPoints) # self.windLoads_Px = np.zeros(nPoints) # self.add_output('windLoads_Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.windLoads_Py = np.zeros(nPoints) # self.windLoads_Py = np.zeros(nPoints) # self.add_output('windLoads_Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.windLoads_Pz = np.zeros(nPoints) # self.windLoads_Pz = np.zeros(nPoints) # self.add_output('windLoads_Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.windLoads_qdyn = np.zeros(nPoints) # self.windLoads_qdyn = np.zeros(nPoints) # self.add_output('windLoads_qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.windLoads_z = np.zeros(nPoints) # self.windLoads_z = np.zeros(nPoints) # self.add_output('windLoads_z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.windLoads_d = np.zeros(nPoints) # self.windLoads_d = np.zeros(nPoints) # self.add_output('windLoads_d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.windLoads_beta = 0.0 # self.windLoads_beta = 0.0 # self.add_output('windLoads_beta', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')


    def compute(self, U, z, d, beta, rho, mu, cd_usr):

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        if cd_usr in [np.inf, -np.inf, None, np.nan]:
            Re = rho*U*d/mu
            cd, self.dcd_dRe = cylinderDrag(Re)
        else:
            cd = cd_usr
            Re = 1.0
            self.dcd_dRe = 0.0
        Fp = q*cd*d

        # components of distributed loads
        Px = Fp*cosd(beta)
        Py = Fp*sind(beta)
        Pz = 0*Fp

        # pack data
        self.windLoads_Px = Px
        self.windLoads_Py = Py
        self.windLoads_Pz = Pz
        self.windLoads_qdyn = q
        self.windLoads_z = z
        self.windLoads_beta = beta

        # derivatives
        self.dq_dU = rho*U
        const = (self.dq_dU*cd + q*self.dcd_dRe*rho*d/mu)*d
        self.dPx_dU = const*cosd(beta)
        self.dPy_dU = const*sind(beta)

        const = (cd + self.dcd_dRe*Re)*q
        self.dPx_dd = const*cosd(beta)
        self.dPy_dd = const*sind(beta)

        self.n = len(z)

        self.zeron = np.zeros((self.n, self.n))


    def provideJ(self, params, unknowns, resids):

        # derivatives
        self.J = np.array([[np.diag(self.dPx_dU), self.zeron, np.diag(self.dPx_dd)]
                           [self.np.diag(dPy_dU), self.zeron, self.np.diag(dPx_dd)]
                           [self.zeron, self.zeron, self.zeron]
                           [self.np.diag(dq_dU), self.zeron, self.zeron]
                           [self.zeron, self.np.eye(n), self.zeron]])

        return self.J

# -----------------

class CylinderWaveDrag(object):
    """drag forces on a cylindrical cylinder due to waves"""

    def __init__(self, nPoints):

        super(CylinderWaveDrag, self).__init__()

        # # variables
        # # self.add_param('U', np.zeros(nPoints), units='m/s', desc='magnitude of wave speed')
        # # self.add_param('A', np.zeros(nPoints), units='m/s**2', desc='magnitude of wave acceleration')
        # # self.add_param('p', np.zeros(nPoints), units='N/m**2', desc='pressure oscillation')
        # # self.add_param('z', np.zeros(nPoints), units='m', desc='heights where wave speed was computed')
        # # self.add_param('d', np.zeros(nPoints), units='m', desc='corresponding diameter of cylinder section')

        # # parameters
        # # self.add_param('beta', 0.0, units='deg', desc='corresponding wave angles relative to inertial coordinate system')
        # # self.add_param('rho', 0.0, units='kg/m**3', desc='water density')
        # # self.add_param('mu', 0.0, units='kg/(m*s)', desc='dynamic viscosity of water')
        # # self.add_param('cm', 0.0, desc='mass coefficient')
        # # self.add_param('cd_usr', np.inf, desc='User input drag coefficient to override Reynolds number based one')

        # out
        self.waveLoads_Px = np.zeros(nPoints) # self.add_output('waveLoads_Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.waveLoads_Py = np.zeros(nPoints) # self.add_output('waveLoads_Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.waveLoads_Pz = np.zeros(nPoints) # self.add_output('waveLoads_Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.waveLoads_qdyn = np.zeros(nPoints) # self.add_output('waveLoads_qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.waveLoads_pt = np.zeros(nPoints) # self.add_output('waveLoads_pt', np.zeros(nPoints), units='N/m**2', desc='total (static+dynamic) pressure')
        self.waveLoads_z = np.zeros(nPoints) # self.add_output('waveLoads_z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.waveLoads_d = np.zeros(nPoints) # self.add_output('waveLoads_d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.waveLoads_beta = 0.0 # self.add_output('waveLoads_beta', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')


    def compute(self, U, A, p, z, d, beta, rho, mu, cm, cd_usr):

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        if cd_usr in [np.inf, -np.inf, None, np.nan]:
            Re = rho*U*d/mu
            cd, self.dcd_dRe = cylinderDrag(Re)
        else:
            cd = cd_usr*np.ones_like(d)
            Re = 1.0
            self.dcd_dRe = 0.0

        # inertial and drag forces
        Fi = rho*cm*math.pi/4.0*d**2*A  # Morrison's equation
        Fd = q*cd*d
        Fp = Fi + Fd

        # components of distributed loads
        Px = Fp*cosd(beta)
        Py = Fp*sind(beta)
        Pz = 0.*Fp

        # pack data
        self.waveLoads_Px = Px
        self.waveLoads_Py = Py
        self.waveLoads_Pz = Pz
        self.waveLoads_qdyn = q
        self.waveLoads_pt = q + p
        self.waveLoads_z = z
        self.waveLoads_beta = beta
        self.waveLoads_d = d

        # derivatives
        self.dq_dU = rho*U
        const = (self.dq_dU*cd + q*self.dcd_dRe*rho*d/mu)*d
        dPx_dU = const*cosd(beta)
        dPy_dU = const*sind(beta)

        const = (cd + self.dcd_dRe*Re)*q + rho*cm*math.pi/4.0*2*d*A
        self.dPx_dd = const*cosd(beta)
        self.dPy_dd = const*sind(beta)

        const = rho*cm*math.pi/4.0*d**2
        self.dPx_dA = const*cosd(beta)
        self.dPy_dA = const*sind(beta)

        self.n = len(z)

        self.zeron = np.zeros((self.n, self.n))

    def provideJ(self, params, unknowns, resids):
        n = self.n
        zeron = self.zeron

        self.J = np.array([[np.diag(self.dPx_dU), np.diag(self.dPx_dA), zeron, np.diag(self.dPx_dd), zeron]
                        [self.np.diag(dPy_dU), np.diag(self.dPy_dA), zeron, np.diag(self.dPy_dd), zeron]
                        [zeron, zeron, zeron, zeron, zeron]
                        [np.diag(self.dq_dU), zeron, zeron, zeron, zeron]
                        [np.diag(self.dq_dU), zeron, zeron, zeron, 1.0]
                        [zeron, zeron, np.eye(n), zeron, zeron]])

        return self.J

#___________________________________________#

def main():
    # initialize problem
    U = np.array([20., 25., 30.])
    z = np.array([10., 30., 80.])
    d = np.array([5.5, 4., 3.])

    beta = np.array([45., 45., 45.])
    rho = 1.225
    mu = 1.7934e-5
    cd_usr = 0.7

    nPoints = len(z)


    #run
    p1 = CylinderWindDrag(nPoints)
    p1.compute(U, z, d, beta, rho, mu, cd_usr)


    # out
    Re = rho*U*d/mu
    cd, dcd_dRe = cylinderDrag(Re)
    print cd
    import matplotlib.pyplot as plt

    plt.plot(p1.windLoads_Px, p1.windLoads_z)
    plt.plot(p1.windLoads_Py, p1.windLoads_z)
    plt.plot(p1.windLoads_qdyn, p1.windLoads_z)
    plt.show()

if __name__ == '__main__':
    main()
