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

from openmdao.api import Component
from commonse.utilities import sind, cosd  # , linspace_with_deriv, interp_with_deriv, hstack, vstack
from commonse.csystem import DirectionVector

from akima import Akima

#TODO still need to do derivatives

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

"""
# -----------------
#  Variable Trees
# -----------------

class FluidLoads(VariableTree):
"""
    #wind/wave loads
"""

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
"""
# -----------------
#  Components
# -----------------

class AeroHydroLoads(Component):

    def __init__(self, nPoints):

        super(AeroHydroLoads, self).__init__()

        #inputs

        self.add_param('windLoads:Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.add_param('windLoads:Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.add_param('windLoads:Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.add_param('windLoads:qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.add_param('windLoads:z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.add_param('windLoads:d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.add_param('windLoads:beta', np.zeros(nPoints), units='deg', desc='wind/wave angle relative to inertia c.s.')
        self.add_param('windLoads:Px0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_param('windLoads:Py0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_param('windLoads:Pz0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_param('windLoads:qdyn0', 0.0, units='N/m**2', desc='dynamic pressure at z=0 MSL')
        self.add_param('windLoads:beta0', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')


        self.add_param('waveLoads:Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.add_param('waveLoads:Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.add_param('waveLoads:Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.add_param('waveLoads:qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.add_param('waveLoads:z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.add_param('waveLoads:d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.add_param('waveLoads:beta', np.zeros(nPoints), units='deg', desc='wind/wave angle relative to inertia c.s.')
        self.add_param('waveLoads:Px0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_param('waveLoads:Py0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_param('waveLoads:Pz0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_param('waveLoads:qdyn0', 0.0, units='N/m**2', desc='dynamic pressure at z=0 MSL')
        self.add_param('waveLoads:beta0', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')


        self.add_param('z', np.zeros(nPoints), desc='locations along tower')
        self.add_param('yaw', 0.0, units='deg', desc='yaw angle')


        self.add_param('outloads:Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.add_param('outloads:Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.add_param('outloads:Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.add_param('outloads:qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.add_param('outloads:z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.add_param('outloads:d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.add_param('outloads:beta', np.zeros(nPoints), units='deg', desc='wind/wave angle relative to inertia c.s.')
        self.add_param('outloads:Px0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_param('outloads:Py0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_param('outloads:Pz0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_param('outloads:qdyn0', 0.0, units='N/m**2', desc='dynamic pressure at z=0 MSL')
        self.add_param('outloads:beta0', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')


        #outputs
        self.add_output('Px', np.zeros(nPoints), units='N/m', desc='force per unit length in x-direction')
        self.add_output('Py', np.zeros(nPoints), units='N/m', desc='force per unit length in y-direction')
        self.add_output('Pz', np.zeros(nPoints), units='N/m', desc='force per unit length in z-direction')
        self.add_output('qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')

    def solve_nonlinear(self, params, unknowns, resids):
        # aero/hydro loads
        wind = params['windLoads']
        wave = params['waveLoads']
        outloads = params['outloads']
        z = params['z']
        hubHt = z[-1]  # top of tower
        betaMain = np.interp(hubHt, z, wind.beta)  # wind coordinate system defined relative to hub height
        windLoads = DirectionVector(wind.Px, wind.Py, wind.Pz).inertialToWind(betaMain).windToYaw(params['yaw'])
        waveLoads = DirectionVector(wave.Px, wave.Py, wave.Pz).inertialToWind(betaMain).windToYaw(params['yaw'])

        outloads.Px = np.interp(z, wind.z, windLoads.x) + np.interp(z, wave.z, waveLoads.x)
        outloads.Py = np.interp(z, wind.z, windLoads.y) + np.interp(z, wave.z, waveLoads.y)
        outloads.Pz = np.interp(z, wind.z, windLoads.z) + np.interp(z, wave.z, waveLoads.z)
        outloads.qdyn = np.interp(z, wind.z, wind.qdyn) + np.interp(z, wave.z, wave.qdyn)
        outloads.z = z


        #The following are redundant, at one point we will consolidate them to something that works for both tower (not using vartrees) and jacket (still using vartrees)
        unknowns['Px'] = outloads.Px
        unknowns['Py'] = outloads.Py
        unknowns['Pz'] = outloads.Pz
        unknowns['qdyn'] = outloads.qdyn

# -----------------

class TowerWindDrag(Component):
    """drag forces on a cylindrical tower due to wind"""
    
    def __init__(self, nPoints):

        super(TowerWindDrag, self).__init__()

        # variables
        self.add_param('U', np.zeros(nPoints), units='m/s', desc='magnitude of wind speed')
        self.add_param('z', np.zeros(nPoints), units='m', desc='heights where wind speed was computed')
        self.add_param('d', np.zeros(nPoints), units='m', desc='corresponding diameter of cylinder section')

        # parameters
        self.add_param('beta', np.zeros(nPoints), units='deg', desc='corresponding wind angles relative to inertial coordinate system')
        self.add_param('rho', 1.225, units='kg/m**3', desc='air density')
        self.add_param('mu', 1.7934e-5, units='kg/(m*s)', desc='dynamic viscosity of air')
        #TODO not sure what to do here?
        self.add_param('cd_usr', 0.0, desc='User input drag coefficient to override Reynolds number based one')

        # out
        self.add_output('windLoads:Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.add_output('windLoads:Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.add_output('windLoads:Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.add_output('windLoads:qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.add_output('windLoads:z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.add_output('windLoads:d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.add_output('windLoads:beta', np.zeros(nPoints), units='deg', desc='wind/wave angle relative to inertia c.s.')
        self.add_output('windLoads:Px0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_output('windLoads:Py0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_output('windLoads:Pz0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_output('windLoads:qdyn0', 0.0, units='N/m**2', desc='dynamic pressure at z=0 MSL')
        self.add_output('windLoads:beta0', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')
            

    def solve_nonlinear(self, params, unknowns, resids):

        rho = params['rho']
        U = params['U']
        d = params['d']
        mu = params['mu']
        beta = params['beta']

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        if params['cd_usr']:
            cd = params['cd_usr']
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
        unknowns['windLoads'].Px = Px
        unknowns['windLoads'].Py = Py
        unknowns['windLoads'].Pz = Pz
        unknowns['windLoads'].qdyn = q
        unknowns['windLoads'].z = self.z
        unknowns['windLoads'].beta = beta

        #TODO need to do this still
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
        outputs = ('windLoads.Px', 'windLoads.Py', 'windLoads.Pz', 'windLoads.qdyn', 'windLoads.z')

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

    def __init__(self, nPoints):

        super(TowerWaveDrag, self).__init__()

        # variables
        self.add_param('U', np.zeros(nPoints), units='m/s', desc='magnitude of wave speed')
        self.add_param('U0', 0.0, units='m/s', desc='magnitude of wave speed at z=0 MSL')
        self.add_param('A', np.zeros(nPoints), units='m/s**2', desc='magnitude of wave acceleration')
        self.add_param('A0', 0.0, units='m/s**2', desc='magnitude of wave acceleration at z=0 MSL')
        self.add_param('z', np.zeros(nPoints), units='m', desc='heights where wave speed was computed')
        self.add_param('d', np.zeros(nPoints), units='m', desc='corresponding diameter of cylinder section')

        # parameters
        self.add_param('wlevel', 0.0, units='m', desc='Water Level, to assess z w.r.t. MSL')
        self.add_param('beta', np.zeros(nPoints), units='deg', desc='corresponding wave angles relative to inertial coordinate system')
        self.add_param('beta0', 0.0, units='deg', desc='corresponding wave angles relative to inertial coordinate system at z=0 MSL')
        self.add_param('rho', 1027.0, units='kg/m**3', desc='water density')
        self.add_param('mu', 1.3351e-3, units='kg/(m*s)', desc='dynamic viscosity of water')
        self.add_param('cm', 2.0, desc='mass coefficient')
        #TODO not sure what to do here?
        self.add_param('cd_usr', 0.0, desc='User input drag coefficient to override Reynolds number based one')

        # out
        self.add_output('waveLoads:Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.add_output('waveLoads:Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.add_output('waveLoads:Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.add_output('waveLoads:qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.add_output('waveLoads:z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.add_output('waveLoads:d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.add_output('waveLoads:beta', np.zeros(nPoints), units='deg', desc='wind/wave angle relative to inertia c.s.')
        self.add_output('waveLoads:Px0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_output('waveLoads:Py0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_output('waveLoads:Pz0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        self.add_output('waveLoads:qdyn0', 0.0, units='N/m**2', desc='dynamic pressure at z=0 MSL')
        self.add_output('waveLoads:beta0', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')


    def solve_nonlinear(self, params, unknowns, resids):

        rho = params['rho']
        U = params['U']
        U0 = params['U0']
        d = params['d']
        zrel= params['z']-params['wlevel']
        mu = params['mu']
        beta = params['beta']
        beta0 = params['beta0']

        # dynamic pressure
        q = 0.5*rho*U**2
        q0= 0.5*rho*U0**2

        # Reynolds number and drag
        if params['cd_usr']:
            cd = sparams['cd_usr']*np.ones_like(d)
            Re = 1.0
            dcd_dRe = 0.0
        else:
            Re = rho*U*d/mu
            cd, dcd_dRe = cylinderDrag(Re)

        # inertial and drag forces
        Fi = rho*self.cm*math.pi/4.0*d**2*params['A']  # Morrison's equation
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
        Fi0 = rho*params['cm']*math.pi/4.0*d0**2*params['A0']  # Morrison's equation
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
        unknowns['waveLoads'].Px0 = Px0
        unknowns['waveLoads'].Py0 = Py0
        unknowns['waveLoads'].Pz0 = Pz0
        unknowns['waveLoads'].qdyn0 = q0
        unknowns['waveLoads'].beta0 = beta0

        # pack data
        unknowns['waveLoads'].Px = Px
        unknowns['waveLoads'].Py = Py
        unknowns['waveLoads'].Pz = Pz
        unknowns['waveLoads'].qdyn = q
        unknowns['waveLoads'].z = self.z
        unknowns['waveLoads'].beta = beta
        unknowns['waveLoads'].d = d

        #TODO still need to do derivatives
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
        outputs = ('waveLoads.Px', 'waveLoads.Py', 'waveLoads.Pz', 'waveLoads.qdyn', 'waveLoads.z', 'waveLoads.beta')

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
    plt.plot(load.windLoads.qdyn, load.windLoads.z)
    plt.show()

if __name__ == '__main__':
    main()
