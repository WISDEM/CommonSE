#!/usr/bin/env python
# encoding: utf-8
"""
environment.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

import math
import numpy as np
from scipy.optimize import brentq
from openmdao.main.api import Component, Assembly
from openmdao.main.datatypes.api import Float, Array


## temporary convenience method
def check_gradient(c):

    class A(Assembly):

        def configure(self):

            self.add('comp', c)
            self.driver.workflow.add('comp')

    a = A()
    a.run()
    c.linearize()
    inputs, outputs, J = c.provideJ()
    inputs = list(inputs)
    outputs = list(outputs)
    for i, inp in enumerate(inputs):
        inputs[i] = 'comp.' + inp
    for i, outp in enumerate(outputs):
        outputs[i] = 'comp.' + outp
    a.check_gradient('comp', inputs=inputs, outputs=outputs, mode='forward')



# -----------------
#  Base Components
# -----------------


class WindBase(Component):
    """base component for wind speed/direction"""

    # in
    z = Array(iotype='in', units='m', desc='heights where wind speed should be computed')

    # out
    U = Array(iotype='out', units='m/s', desc='magnitude of wind speed at each z location')
    beta = Array(iotype='out', units='deg', desc='corresponding wind angles relative to inertial coordinate system')


class WaveBase(Component):
    """base component for wave speed/direction"""

    # in
    z = Array(iotype='in', units='m', desc='heights where wave speed should be computed')

    # out
    U = Array(iotype='out', units='m/s', desc='magnitude of wave speed at each z location')
    A = Array(iotype='out', units='m/s**2', desc='magnitude of wave acceleration at each z location')
    beta = Array(iotype='out', units='deg', desc='corresponding wave angles relative to inertial coordinate system')


    def execute(self):
        """default to no waves"""
        n = len(self.z)
        self.U = np.zeros(n)
        self.A = np.zeros(n)
        self.beta = np.zeros(n)


class SoilBase(Component):
    """base component for soil stiffness"""

    # out
    k = Array(iotype='out', units='N/m', desc='spring stiffness. rigid directions should use \
        ``float(''inf'')``. order: (x, theta_x, y, theta_y, z, theta_z)')




# -----------------------
#  Subclassed Components
# -----------------------


class PowerWind(WindBase):
    """power-law profile wind"""

    # variables
    Uref = Float(iotype='in', units='m/s', desc='reference velocity of power-law model')
    zref = Float(iotype='in', units='m', desc='corresponding reference height')

    # parameters
    z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')
    shearExp = Float(0.2, iotype='in', desc='shear exponent')
    betaWind = Float(0.0, iotype='in', units='deg', desc='wind angle relative to inertial coordinate system')


    def execute(self):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0

        # velocity
        self.U = np.zeros_like(z)
        idx = z > z0
        self.U[idx] = self.Uref*((z[idx] - z0)/(zref - z0))**self.shearExp
        self.beta = self.betaWind*np.ones_like(z)


    def linearize(self):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0

        dU_dUref = np.zeros_like(z)
        dU_dz = np.zeros_like(z)
        dU_dzref = np.zeros_like(z)

        idx = z > z0
        dU_dUref[idx] = ((z[idx] - z0)/(zref - z0))**self.shearExp
        dU_dz[idx] = self.U[idx]*self.shearExp * 1.0/(z[idx] - z0)
        dU_dzref[idx] = self.U[idx]*self.shearExp * -1.0/(zref - z0)

        # TODO: the gradient at z == z0 will always be wrong, must rethink.

        self.J = np.hstack((dU_dUref.reshape(-1, 1), np.diag(dU_dz), dU_dzref.reshape(-1, 1)))


    def provideJ(self):

        inputs = ('Uref', 'z', 'zref')
        outputs = ('U',)

        return inputs, outputs, self.J




class LogWind(WindBase):
    """logarithmic-profile wind"""

    # variables
    Uref = Float(iotype='in', units='m/s', desc='reference velocity of power-law model')
    zref = Float(iotype='in', units='m', desc='corresponding reference height')
    z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')

    # parameters
    z_roughness = Float(10.0, iotype='in', units='mm', desc='surface roughness length')
    betaWind = Float(0.0, iotype='in', units='deg', desc='wind angle relative to inertial coordinate system')


    def execute(self):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0
        z_roughness = self.z_roughness

        # find velocity
        self.U = np.zeros_like(z)
        idx = [z > z0]
        self.U[idx] = self.Uref*np.log((z[idx] - z0)/z_roughness) / math.log((zref - z0)/z_roughness)
        self.beta = self.betaWind*np.ones_like(z)


    def linearize(self):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0
        z_roughness = self.z_roughness

        dU_dUref = np.zeros_like(z)
        dU_dz = np.zeros_like(z)
        dU_dzref = np.zeros_like(z)
        dU_dz0 = np.zeros_like(z)

        idx = z > z0
        lt = np.log((z[idx] - z0)/z_roughness)
        lb = math.log((zref - z0)/z_roughness)
        dU_dUref[idx] = lt/lb
        dU_dz[idx] = self.Uref/lb / (z[idx] - z0)
        dU_dzref[idx] = -self.Uref*lt / math.log((zref - z0)/z_roughness)**2 / (zref - z0)
        dU_dz0[idx] = self.U[idx]*(-lb/(z[idx]-z0) + lt/(zref-z0))/lb**2

        dU = np.hstack((dU_dUref, dU_dz, dU_dzref, dU_dz0))

        self.J = np.array([dU])


    def provideJ(self):

        inputs = ('Uref', 'z', 'zref', 'z0')
        outputs = ('U',)

        return inputs, outputs, self.J


class LinearWaves(WaveBase):
    """linear (Airy) wave theory"""

    # variables
    Uc = Float(iotype='in', units='m/s', desc='mean current speed')

    # parameters
    z_surface = Float(iotype='in', units='m', desc='vertical location of water surface')
    hs = Float(iotype='in', units='m', desc='significant wave height (crest-to-trough)')
    T = Float(iotype='in', units='s', desc='period of waves')
    z_floor = Float(0.0, iotype='in', units='m', desc='vertical location of sea floor')
    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity')
    betaWave = Float(0.0, iotype='in', units='deg', desc='wave angle relative to inertial coordinate system')


    def execute(self):

        # water depth
        d = self.z_surface - self.z_floor

        # design wave height
        h = 1.1*self.hs

        # circular frequency
        omega = 2.0*math.pi/self.T

        # compute wave number from dispersion relationship
        k = brentq(lambda k: omega**2 - self.g*k*math.tanh(d*k), 0, 10*omega**2/self.g)

        # zero at surface
        z_rel = self.z - self.z_surface

        # maximum velocity
        self.U = h/2.0*omega*np.cosh(k*(z_rel + d))/math.sinh(k*d) + self.Uc

        # check heights
        self.U[np.logical_or(self.z < self.z_floor, self.z > self.z_surface)] = 0

        # acceleration
        self.A = self.U * omega

        # angles
        self.beta = self.betaWave*np.ones_like(self.z)

        # derivatives
        dU_dz = h/2.0*omega*np.sinh(k*(z_rel + d))/math.sinh(k*d)*k
        dU_dUc = 1.0
        dA_dz = omega*dU_dz
        dA_dUc = omega
        self.J = np.vstack([np.array([dU_dz, dU_dUc]), np.array([dA_dz, dA_dUc])])

    def provideJ(self):

        inputs = ('z', 'Uc')
        outputs = ('U', 'A')

        return inputs, outputs, self.J


class TowerSoil(SoilBase):
    """textbook soil stiffness method"""

    # variable
    r0 = Float(1.0, iotype='in', units='m', desc='radius of base of tower')
    depth = Float(1.0, iotype='in', units='m', desc='depth of foundation in the soil')

    # parameter
    G = Float(140e6, iotype='in', units='Pa', desc='shear modulus of soil')
    nu = Float(0.4, iotype='in', desc='Poisson''s ratio of soil')
    rigid = Array(iotype='in', dtype=np.bool, desc='directions that should be considered infinitely rigid\
        order is x, theta_x, y, theta_y, z, theta_z')


    def execute(self):

        G = self.G
        nu = self.nu
        h = self.depth
        r0 = self.r0

        # vertical
        eta = 1.0 + 0.6*(1.0-nu)*h/r0
        k_z = 4*G*r0*eta/(1.0-nu)

        # horizontal
        eta = 1.0 + 0.55*(2.0-nu)*h/r0
        k_x = 32.0*(1.0-nu)*G*r0*eta/(7.0-8.0*nu)

        # rocking
        eta = 1.0 + 1.2*(1.0-nu)*h/r0 + 0.2*(2.0-nu)*(h/r0)**3
        k_thetax = 8.0*G*r0**3*eta/(3.0*(1.0-nu))

        # torsional
        k_phi = 16.0*G*r0**3/3.0

        self.k = np.array([k_x, k_thetax, k_x, k_thetax, k_z, k_phi])
        self.k[self.rigid] = float('inf')


    def linearize(self):

        G = self.G
        nu = self.nu
        h = self.depth
        r0 = self.r0

        # vertical
        eta = 1.0 + 0.6*(1.0-nu)*h/r0
        deta_dr0 = -0.6*(1.0-nu)*h/r0**2
        dkz_dr0 = 4*G/(1.0-nu)*(eta + r0*deta_dr0)

        deta_dh = -0.6*(1.0-nu)/r0
        dkz_dh = 4*G*r0/(1.0-nu)*deta_dh

        # horizontal
        eta = 1.0 + 0.55*(2.0-nu)*h/r0
        deta_dr0 = -0.55*(2.0-nu)*h/r0**2
        dkx_dr0 = 32.0*(1.0-nu)*G/(7.0-8.0*nu)*(eta + r0*deta_dr0)

        deta_dh = 0.55*(2.0-nu)/r0
        dkx_dh = 32.0*(1.0-nu)*G*r0/(7.0-8.0*nu)*deta_dh

        # rocking
        eta = 1.0 + 1.2*(1.0-nu)*h/r0 + 0.2*(2.0-nu)*(h/r0)**3
        deta_dr0 = -1.2*(1.0-nu)*h/r0**2 - 3*0.2*(2.0-nu)*(h/r0)**3/r0
        dkthetax_dr0 = 8.0*G/(3.0*(1.0-nu))*(3*r0**2*eta + r0**3*deta_dr0)

        deta_dh = 1.2*(1.0-nu)/r0 + 3*0.2*(2.0-nu)*(1.0/r0)**3*h**2
        dkthetax_dh = 8.0*G*r0**3/(3.0*(1.0-nu))*deta_dh

        # torsional
        dkphi_dr0 = 16.0*G*3*r0**2/3.0
        dkphi_dh = 0.0

        dk_dr0 = np.array([dkx_dr0, dkthetax_dr0, dkx_dr0, dkthetax_dr0, dkz_dr0, dkphi_dr0])
        dk_dr0[self.rigid] = 0.0
        dk_dh = np.array([dkx_dh, dkthetax_dh, dkx_dh, dkthetax_dh, dkz_dh, dkphi_dh])
        dk_dh[self.rigid] = 0.0

        self.J = np.vstack((dk_dr0, dk_dh))


    def provideJ(self):

        inputs = ('r0', 'depth')
        outputs = ('k',)

        return inputs, outputs, self.J



if __name__ == '__main__':
    p = PowerWind()
    p.Uref = 10.0
    p.zref = 100.0
    p.z0 = 1.0
    p.z = np.linspace(1.0, 120.0, 20)
    p.shearExp = 0.2
    p.betaWind = 0.0
    # check_gradient(p)

    p.run()
    U0 = p.U[0]

    p.z[0] += 1e-6
    p.run()
    print p.U[0]
    print U0
    print (p.U[0] - U0)/1e-6

    # l = LogWind()
    # l.Uref = 12.0
    # l.zref = 100.0
    # l.z0 = 5.0

    # # parameters
    # z_roughness = Float(10.0, iotype='in', units='mm', desc='surface roughness length')
    # betaWind = Float(0.0
