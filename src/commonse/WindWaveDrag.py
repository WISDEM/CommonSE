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

from openmdao.main.api import Component, VariableTree
from openmdao.main.datatypes.api import Float, Array,VarTree
from commonse.utilities import sind, cosd  # , linspace_with_deriv, interp_with_deriv, hstack, vstack
from commonse.csystem import DirectionVector

from akima import Akima

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
#  Variable Trees
# -----------------

class FluidLoads(VariableTree):
    """wind/wave loads"""

    Px = Array(units='N/m', desc='distributed loads, force per unit length in x-direction')
    Py = Array(units='N/m', desc='distributed loads, force per unit length in y-direction')
    Pz = Array(units='N/m', desc='distributed loads, force per unit length in z-direction')
    qdyn = Array(units='N/m**2', desc='dynamic pressure')
    z = Array(units='m', desc='corresponding heights')
    d = Array(units='m', desc='corresponding diameters')

    beta = Array(units='deg', desc='wind/wave angle relative to inertia c.s.')
    Px0= Float(units='N/m', desc='Distributed load at z=0 MSL')
    Py0= Float(units='N/m', desc='Distributed load at z=0 MSL')
    Pz0= Float(units='N/m', desc='Distributed load at z=0 MSL')
    qdyn0 = Float(units='N/m**2', desc='dynamic pressure at z=0 MSL')
    beta0 = Float(units='deg', desc='wind/wave angle relative to inertia c.s.')
# -----------------
#  Components
# -----------------

class AeroHydroLoads(Component):

    windLoads = VarTree(FluidLoads(), iotype='in', desc='wind loads in inertial coordinate system')
    waveLoads = VarTree(FluidLoads(), iotype='in', desc='wave loads in inertial coordinate system')
    z = Array(iotype='in', desc='locations along tower')
    yaw = Float(0.0, iotype='in', units='deg', desc='yaw angle')

    outloads= VarTree(FluidLoads(), iotype='in', desc='combined wind and wave loads')
    Px = Array(iotype='out', units='N/m', desc='force per unit length in x-direction')
    Py = Array(iotype='out', units='N/m', desc='force per unit length in y-direction')
    Pz = Array(iotype='out', units='N/m', desc='force per unit length in z-direction')
    qdyn = Array(iotype='out', units='N/m**2', desc='dynamic pressure')

    def execute(self):
        # aero/hydro loads
        wind = self.windLoads
        wave = self.waveLoads
        hubHt = self.z[-1]  # top of tower
        betaMain = np.interp(hubHt, self.z, wind.beta)  # wind coordinate system defined relative to hub height
        windLoads = DirectionVector(wind.Px, wind.Py, wind.Pz).inertialToWind(betaMain).windToYaw(self.yaw)
        waveLoads = DirectionVector(wave.Px, wave.Py, wave.Pz).inertialToWind(betaMain).windToYaw(self.yaw)

        self.outloads.Px = np.interp(self.z, wind.z, windLoads.x) + np.interp(self.z, wave.z, waveLoads.x)
        self.outloads.Py = np.interp(self.z, wind.z, windLoads.y) + np.interp(self.z, wave.z, waveLoads.y)
        self.outloads.Pz = np.interp(self.z, wind.z, windLoads.z) + np.interp(self.z, wave.z, waveLoads.z)
        self.outloads.qdyn = np.interp(self.z, wind.z, wind.q) + np.interp(self.z, wave.z, wave.q)
        self.outloads.z = self.z
        #The following are redundant, at one point we will consolidate them to something that works for both tower (not using vartrees) and jacket (still using vartrees)
        self.Px   =self.outloads.Px
        self.Py   =self.outloads.Py
        self.Pz   =self.outloads.Pz
        self.qdyn =self.outloads.qdyn

# -----------------

class TowerWindDrag(Component):
    """drag forces on a cylindrical tower due to wind"""

    # variables
    U = Array(iotype='in', units='m/s', desc='magnitude of wind speed')
    z = Array(iotype='in', units='m', desc='heights where wind speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')

    # parameters
    beta = Array(iotype='in', units='deg', desc='corresponding wind angles relative to inertial coordinate system')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='air density')
    mu = Float(1.7934e-5, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')
    cd_usr = Float(iotype='in', desc='User input drag coefficient to override Reynolds number based one')

    # out
    windLoads = VarTree(FluidLoads(), iotype='out', desc='wind loads in inertial coordinate system')

    missing_deriv_policy = 'assume_zero'


    def execute(self):

        rho = self.rho
        U = self.U
        d = self.d
        mu = self.mu
        beta = self.beta

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        if self.cd_usr:
            cd = self.cd_usr
            Re = 1.0
            dcd_dRe = 0.0
        else:
            Re = rho*U*d/mu
            cd, dcd_dRe = cylinderDrag(Re)
        Fp = q*cd*d

        # components of distributed loads
        Px = Fp*cosd(beta)
        Py = Fp*sind(beta)
        Pz = 0*Fp

        # pack data
        self.windLoads.Px = Px
        self.windLoads.Py = Py
        self.windLoads.Pz = Pz
        self.windLoads.qdyn = q
        self.windLoads.z = self.z
        self.windLoads.beta = beta

        # derivatives
        self.dq_dU = rho*U
        const = (self.dq_dU*cd + q*dcd_dRe*rho*d/mu)*d
        self.dPx_dU = const*cosd(beta)
        self.dPy_dU = const*sind(beta)

        const = (cd + dcd_dRe*Re)*q
        self.dPx_dd = const*cosd(beta)
        self.dPy_dd = const*sind(beta)


    def list_deriv_vars(self):

        inputs = ('U', 'z', 'd')
        outputs = ('windLoads.Px', 'windLoads.Py', 'windLoads.Pz', 'windLoads.q', 'windLoads.z')

        return inputs, outputs


    def provideJ(self):

        n = len(self.z)

        zeron = np.zeros((n, n))

        dPx = np.hstack([np.diag(self.dPx_dU), zeron, np.diag(self.dPx_dd)])
        dPy = np.hstack([np.diag(self.dPy_dU), zeron, np.diag(self.dPy_dd)])
        dPz = np.zeros((n, 3*n))
        dq = np.hstack([np.diag(self.dq_dU), np.zeros((n, 2*n))])
        dz = np.hstack([zeron, np.eye(n), zeron])

        J = np.vstack([dPx, dPy, dPz, dq, dz])

        return J

# -----------------

class TowerWaveDrag(Component):
    """drag forces on a cylindrical tower due to waves"""

    # variables
    U = Array(iotype='in', units='m/s', desc='magnitude of wave speed')
    U0= Float(iotype='in', units='m/s', desc='magnitude of wave speed at z=0 MSL')
    A = Array(iotype='in', units='m/s**2', desc='magnitude of wave acceleration')
    A0= Float(iotype='in', units='m/s**2', desc='magnitude of wave acceleration at z=0 MSL')
    z = Array(iotype='in', units='m', desc='heights where wave speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')

    # parameters
    wlevel = Float(iotype='in', units='m', desc='Water Level, to assess z w.r.t. MSL')
    beta = Array(iotype='in', units='deg', desc='corresponding wave angles relative to inertial coordinate system')
    beta0= Float(iotype='in', units='deg', desc='corresponding wave angles relative to inertial coordinate system at z=0 MSL')
    rho = Float(1027.0, iotype='in', units='kg/m**3', desc='water density')
    mu = Float(1.3351e-3, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    cm = Float(2.0, iotype='in', desc='mass coefficient')
    cd_usr = Float(iotype='in', desc='User input drag coefficient to override Reynolds number based one')

    # out
    waveLoads = VarTree(FluidLoads(), iotype='out', desc='wave loads in inertial coordinate system')


    missing_deriv_policy = 'assume_zero'


    def execute(self):

        rho = self.rho
        U = self.U
        U0 = self.U0
        d = self.d
        zrel= self.z-self.wlevel
        mu = self.mu
        beta = self.beta
        beta0= self.beta0

        # dynamic pressure
        q = 0.5*rho*U**2
        q0= 0.5*rho*U0**2

        # Reynolds number and drag
        if self.cd_usr:
            cd = self.cd_usr*np.ones_like(self.d)
            Re = 1.0
            dcd_dRe = 0.0
        else:
            Re = rho*U*d/mu
            cd, dcd_dRe = cylinderDrag(Re)

        d = self.d
        mu = self.mu
        beta = self.beta

        # inertial and drag forces
        Fi = rho*self.cm*math.pi/4.0*d**2*self.A  # Morrison's equation
        Fd = q*cd*d
        Fp = Fi + Fd

        #FORCES [N/m] AT z=0 m
        idx0 = np.abs(zrel).argmin()  # closest index to z=0, used to find d at z=0
        d0 = d[idx0]  # initialize
        cd0 = cd[idx0]  # initialize
        if (zrel[idx0]<0.) and (idx0< (zrel.size-1)):       # point below water
            d0 = np.mean(d[idx0:idx0+2])
            cd0 = np.mean(cd[idx0:idx0+2])
        elif (zrel[idx0]>0.) and (idx0>0):     # point above water
            d0 = np.mean(d[idx0-1:idx0+1])
            cd0 = np.mean(cd[idx0-1:idx0+1])
        Fi0 = rho*self.cm*math.pi/4.0*d0**2*self.A0  # Morrison's equation
        Fd0 = q0*cd0*d0
        Fp0 = Fi0 + Fd0

        # components of distributed loads
        Px = Fp*cosd(beta)
        Py = Fp*sind(beta)
        Pz = 0.*Fp

        Px0 = Fp0*cosd(beta0)
        Py0 = Fp0*sind(beta0)
        Pz0 = 0.*Fp0

        #Store qties at z=0 MSL
        self.waveLoads.Px0 = Px0
        self.waveLoads.Py0 = Py0
        self.waveLoads.Pz0 = Pz0
        self.waveLoads.q0 = q0
        self.waveLoads.beta0 = beta0

        # pack data
        self.waveLoads.Px = Px
        self.waveLoads.Py = Py
        self.waveLoads.Pz = Pz
        self.waveLoads.qdyn = q
        self.waveLoads.z = self.z
        self.waveLoads.beta = beta
        self.waveLoads.d = d


        # derivatives
        self.dq_dU = rho*U
        const = (self.dq_dU*cd + q*dcd_dRe*rho*d/mu)*d
        self.dPx_dU = const*cosd(beta)
        self.dPy_dU = const*sind(beta)

        const = (cd + dcd_dRe*Re)*q + rho*self.cm*math.pi/4.0*2*d*self.A
        self.dPx_dd = const*cosd(beta)
        self.dPy_dd = const*sind(beta)

        const = rho*self.cm*math.pi/4.0*d**2
        self.dPx_dA = const*cosd(beta)
        self.dPy_dA = const*sind(beta)


    def list_deriv_vars(self):

        inputs = ('U', 'A', 'z', 'd')
        outputs = ('waveLoads.Px', 'waveLoads.Py', 'waveLoads.Pz', 'waveLoads.q', 'waveLoads.z', 'waveLoads.beta')

        return inputs, outputs


    def provideJ(self):

        n = len(self.z)

        zeron = np.zeros((n, n))

        dPx = np.hstack([np.diag(self.dPx_dU), np.diag(self.dPx_dA), zeron, np.diag(self.dPx_dd)])
        dPy = np.hstack([np.diag(self.dPy_dU), np.diag(self.dPy_dA), zeron, np.diag(self.dPy_dd)])
        dPz = np.zeros((n, 4*n))
        dq = np.hstack([np.diag(self.dq_dU), np.zeros((n, 3*n))])
        dz = np.hstack([zeron, zeron, np.eye(n), zeron])

        J = np.vstack([dPx, dPy, dPz, dq, dz, np.zeros((n, 4*n))])  # TODO: remove these zeros after OpenMDAO bug fix (don't need waveLoads.beta)

        return J

#___________________________________________#

def main():
    load=TowerWindDrag()
    load.U = np.array([20., 25., 30.])
    load.z = np.array([10., 30., 80.])
    load.d = np.array([5.5, 4., 3.])

    load.beta = np.array([45., 45., 45.])
    load.rho = 1.225
    load.mu = 1.7934e-5
    load.cd_usr = 0.7

    #run
    load.run()

    # out
    import matplotlib.pyplot as plt
    plt.plot(load.windLoads.Px, load.windLoads.z)
    plt.plot(load.windLoads.Py, load.windLoads.z)
    plt.plot(load.windLoads.q, load.windLoads.z)
    plt.show()

if __name__ == '__main__':
    main()