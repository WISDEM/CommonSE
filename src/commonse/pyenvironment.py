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

import sys

from utilities import hstack, vstack
from constants import gravity

#TODO CHECK

# -----------------
#  Base Components
# -----------------


class WindBase(object):
    """base component for wind speed/direction"""

    def __init__(self, nPoints):

        super(WindBase, self).__init__()
        self.npts = nPoints
        
        # TODO: if I put required=True here for Uref there is another bug

        # variables
        # self.add_param('Uref', 0.0, units='m/s', desc='reference wind speed (usually at hub height)')
        # self.add_param('zref', 0.0, units='m', desc='corresponding reference height')
        # self.add_param('z', np.zeros(nPoints), units='m', desc='heights where wind speed should be computed')

        # parameters
        # self.add_param('z0', 0.0, units='m', desc='bottom of wind profile (height of ground/sea)')

        self.Uref = 0.0
        self.zref = 0.0
        self.z = np.zeros(nPoints)
        self.z0 = 0.0

        # out
        self.U = np.zeros(nPoints) # self.add_output('U', np.zeros(nPoints), units='m/s', desc='magnitude of wind speed at each z location')


class WaveBase(object):
    """base component for wave speed/direction"""

    def __init__(self, nPoints):

        super(WaveBase, self).__init__()
        self.npts = nPoints

        # variables
        # self.add_param('rho', 0.0, units='kg/m**3', desc='water density')
        # self.add_param('z', np.zeros(nPoints), units='m', desc='heights where wave speed should be computed')
        # self.add_param('z_surface', 0.0, units='m', desc='vertical location of water surface')
        # self.add_param('z_floor', 0.0, units='m', desc='vertical location of sea floor')

        # out
        self.U = np.zeros(nPoints) # self.add_output('U', np.zeros(nPoints), units='m/s', desc='horizontal wave velocity at each z location')
        self.W = np.zeros(nPoints) # self.add_output('W', np.zeros(nPoints), units='m/s', desc='vertical wave velocity at each z location')
        self.V = np.zeros(nPoints) # self.add_output('V', np.zeros(nPoints), units='m/s', desc='total wave velocity at each z location')
        self.A = np.zeros(nPoints) # self.add_output('A', np.zeros(nPoints), units='m/s**2', desc='horizontal wave acceleration at each z location')
        self.p = np.zeros(nPoints) # self.add_output('p', np.zeros(nPoints), units='N/m**2', desc='pressure oscillation at each z location')
        #self.U0 = 0.0 # self.add_output('U0', 0.0, units='m/s', desc='magnitude of wave speed at z=MSL')
        #self.A0 = 0.0 # self.add_output('A0', 0.0, units='m/s**2', desc='magnitude of wave acceleration at z=MSL')


    def compute(self, rho, z, z_surface, z_floor):
        """default to no waves"""
        n = len(z)
        self.U = np.zeros(n)
        self.W = np.zeros(n)
        self.V = np.zeros(n)
        self.A = np.zeros(n)
        self.p = np.zeros(n)
        #self.U0 = 0.
        #self.A0 = 0.



class SoilBase(object):
    """base component for soil stiffness"""

    def __init__(self):

        super(SoilBase, self).__init__()

        # out
        self.k = np.zeros(6) # self.add_output('k', np.zeros(6), units='N/m', required=True, desc='spring stiffness. rigid directions should use ``float(''inf'')``. order: (x, theta_x, y, theta_y, z, theta_z)')


# -----------------------
#  Subclassed Components
# -----------------------


class PowerWind(WindBase):
    """power-law profile wind.  any nodes must not cross z0, and if a node is at z0
    it must stay at that point.  otherwise gradients crossing the boundary will be wrong."""

    def __init__(self, nPoints):

        super(PowerWind, self).__init__(nPoints)

        # parameters
        # self.add_param('shearExp', 0.0, desc='shear exponent')


    def compute(self, shearExp):

        # rename
        z = self.z
        if isinstance(z, float) or isinstance(z,np.float_): z=np.array([z])
        zref = self.zref
        z0 = self.z0
        Uref = self.Uref

        # velocity
        idx = z > z0
        self.U = np.zeros(self.npts)
        self.U[idx] = Uref*((z[idx] - z0)/(zref - z0))**shearExp

        # # add small cubic spline to allow continuity in gradient
        # k = 0.01  # fraction of profile with cubic spline
        # zsmall = z0 + k*(zref - z0)

        # self.spline = CubicSpline(x1=z0, x2=zsmall, f1=0.0, f2=Uref*k**shearExp,
        #     g1=0.0, g2=Uref*k**shearExp*shearExp/(zsmall - z0))

        # idx = np.logical_and(z > z0, z < zsmall)
        # self.U[idx] = self.spline.eval(z[idx])

        # self.zsmall = zsmall
        # self.k = k

        # gradients
        self.dU_dUref = np.zeros(self.npts)
        self.dU_dz = np.zeros(self.npts)
        self.dU_dzref = np.zeros(self.npts)

        idx = z > z0
        self.dU_dUref[idx] = self.U[idx]/Uref
        self.dU_dz[idx] = self.U[idx]*shearExp/(z[idx] - z0)
        self.dU_dzref[idx] = -self.U[idx]*shearExp/(zref - z0)

    def provideJ(self, params, unknowns, resids):

        self.J = [self.dU_dUref, self.np.diag(dU_dz), self.dU_dzref]

        #TODO still missing several partials? This is what was in the original code though...

        # # cubic spline region
        # idx = np.logical_and(z > z0, z < zsmall)

        # # d w.r.t z
        # dU_dz[idx] = self.spline.eval_deriv(z[idx])

        # # d w.r.t. Uref
        # df2_dUref = k**shearExp
        # dg2_dUref = k**shearExp*shearExp/(zsmall - z0)
        # dU_dUref[idx] = self.spline.eval_deriv_params(z[idx], 0.0, 0.0, 0.0, df2_dUref, 0.0, dg2_dUref)

        # # d w.r.t. zref
        # dx2_dzref = k
        # dg2_dzref = -Uref*k**shearExp*shearExp/k/(zref - z0)**2
        # dU_dzref[idx] = self.spline.eval_deriv_params(z[idx], 0.0, dx2_dzref, 0.0, 0.0, 0.0, dg2_dzref)

        return self.J


class LogWind(WindBase):
    """logarithmic-profile wind"""

    def __init__(self, nPoints):

        super(LogWind, self).__init__(nPoints)

        # parameters
        # self.add_param('z_roughness', 0.0, units='mm', desc='surface roughness length')


    def compute(self, z_roughness):

        # rename
        z = self.z
        if isinstance(z, float) or isinstance(z,np.float_): z=np.array([z])
        zref = self.zref
        z0 = self.z0
        z_roughness = z_roughness/1e3  # convert to m
        Uref = self.Uref

        # find velocity
        idx = [z - z0 > z_roughness]
        self.U = np.zeros_like(z)
        self.U[idx] = Uref*np.log((z[idx] - z0)/z_roughness) / math.log((zref - z0)/z_roughness)

        # gradients
        self.dU_dUref = np.zeros(self.npts)
        self.dU_dz_diag = np.zeros(self.npts)
        self.dU_dzref = np.zeros(self.npts)

        idx = [z - z0 > z_roughness]
        lt = np.log((z[idx] - z0)/z_roughness)
        lb = math.log((zref - z0)/z_roughness)
        self.dU_dUref[idx] = lt/lb
        self.dU_dz_diag[idx] = Uref/lb / (z[idx] - z0)
        self.dU_dzref[idx] = -Uref*lt / math.log((zref - z0)/z_roughness)**2 / (zref - z0)

    def provideJ(self, params, unknowns, resids):

        J = [self.dU_dUref, np.diag(self.dU_dz_diag), self.dU_dzref]
        
        return J



class LinearWaves(WaveBase):
    """linear (Airy) wave theory"""

    def __init__(self, nPoints):

        super(LinearWaves, self).__init__(nPoints)

        # variables
        # self.add_param('Uc', 0.0, units='m/s', desc='mean current speed')

        # parameters
        # self.add_param('hmax', 0.0, units='m', desc='maximum wave height (crest-to-trough)')
        # self.add_param('T', 0.0, units='s', desc='period of maximum wave height')

        # For Ansys AQWA connection
        self.phase_speed = 0.0 # self.add_output('phase_speed', val=0.0, units='m/s', desc='phase speed of wave')

    def compute(self, Uc, hmax, T):

        # renaming
        z_floor = self.z_floor
        z_surface = self.z_surface
        z = self.z
        rho = self.rho

        # water depth
        if z_floor > 0.0: z_floor *= -1.0
        d = z_surface-z_floor

        # design wave height
        h = hmax

        # circular frequency
        omega = 2.0*math.pi/T

        # compute wave number from dispersion relationship
        k = brentq(lambda k: omega**2 - gravity*k*math.tanh(d*k), 0, 1e3*omega**2/gravity)
        self.k = k
        self.phase_speed = omega / k
        
        # zero at surface
        z_rel = z - z_surface

        # Amplitude
        a = 0.5 * h
        
        # maximum velocity
        self.U = a*omega*np.cosh(k*(z_rel + d))/np.sinh(k*d) + Uc
        self.W = -a*omega*np.sinh(k*(z_rel + d))/np.sinh(k*d)
        self.V = np.sqrt(self.U**2.0 + self.W**2.0)
        #self.U0 = a*omega*np.cosh(k*(0. + d))/np.sinh(k*d) + Uc

        # acceleration
        self.A  = (self.U - Uc) * omega
        #self.A0 = (self.U0 - Uc) * omega

        # Pressure oscillation is just sum of static and dynamic contributions
        # Hydrostatic is simple rho * g * z
        # Dynamic is from standard solution to Airy (Potential Flow) Wave theory
        # Full pressure would also include standard dynamic head (0.5*rho*V^2)
        self.p = rho * gravity * (a * np.cosh(k*(z_rel + d)) / np.cosh(k*d) - z_rel)

        # check heights
        idx = np.logical_or(z < z_floor, z > z_surface)
        self.U[idx] = 0.0
        self.W[idx] = 0.0
        self.V[idx] = 0.0
        self.A[idx] = 0.0
        self.p[idx] = 0.0

        # derivatives
        self.dU_dz = h/2.0*omega*np.sinh(k*(z_rel + d))/np.sinh(k*d)*k
        self.dU_dUc = np.ones_like(z)
        self.dW_dz = -h/2.0*omega*np.cosh(k*(z_rel + d))/np.sinh(k*d)*k
        self.dV_dz = 0.5/self.V*(2*self.U*self.dU_dz +2*self.W*self.dW_dz)
        self.dV_dUc = 0.5/self.V*(2*self.U*self.dU_dUc)
        self.dA_dz = omega*self.dU_dz
        self.dA_dUc = 0.0 #omega*dU_dUc
        self.dp_dz = rho * gravity * (a*np.sinh(k*(z_rel + d))*k / np.cosh(k*d) - 1.0)

        idx = np.logical_or(z < z_floor, z > z_surface)
        self.dU_dz[idx] = 0.0
        self.dW_dz[idx] = 0.0
        self.dV_dz[idx] = 0.0
        self.dA_dz[idx] = 0.0
        self.dp_dz[idx] = 0.0
        self.dU_dUc[idx] = 0.0
        self.dV_dUc[idx] = 0.0
        #dU0 = np.zeros((1,self.npts))
        #dA0 = omega * dU0

    def linearize(self, params, unknowns, resids):

        self.J = [[np.diag(self.dU_dz), self.dU_dUc]
                  [np.diag(self.dW_dz), 0.0]
                  [np.diag(self.dV_dz), 0.0]
                  [np.diag(self.dA_dz), 0.0]
                  [np.diag(self.dp_dz), 0.0]]

        return self.J

class TowerSoilK(SoilBase):
    """Passthrough of Soil-Structure-INteraction equivalent spring constants used to bypass TowerSoil."""

    def __init__(self):

        super(TowerSoilK, self).__init__()

        # variable
        # self.add_param('kin', np.ones(6)*float('inf'),  desc='spring stiffness. rigid directions should use ``float(''inf'')``. order: (x, theta_x, y, theta_y, z, theta_z)')
        # self.add_param('rigid', np.ones(6), dtype=np.bool, desc='directions that should be considered infinitely rigid order is x, theta_x, y, theta_y, z, theta_z')


    def compute(self, kin, rigid):
        self.k = kin
        k[rigid] = float('inf')

class TowerSoil(SoilBase):
    """textbook soil stiffness method"""
    def __init__(self):

        super(TowerSoil, self).__init__()
        # variable
        # self.add_param('r0', 1.0, units='m', desc='radius of base of tower')
        # self.add_param('depth', 1.0, units='m', desc='depth of foundation in the soil')

        # parameter
        # self.add_param('G', 140e6, units='Pa', desc='shear modulus of soil')
        # self.add_param('nu', 0.4, desc='Poisson''s ratio of soil')
        # self.add_param('rigid', np.ones(6), dtype=np.bool, desc='directions that should be considered infinitely rigid order is x, theta_x, y, theta_y, z, theta_z')


    def compute(self, r0, depth, G, nu, rigid):

        h = depth

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
        self.k[rigid] = float('inf')

        # derivatives
        # vertical
        eta = 1.0 + 0.6*(1.0-nu)*h/r0
        deta_dr0 = -0.6*(1.0-nu)*h/r0**2
        dkz_dr0 = 4*G/(1.0-nu)*(eta + r0*deta_dr0)

        deta_dh = 0.6*(1.0-nu)/r0
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

        self.dk_dr0 = np.array([dkx_dr0, dkthetax_dr0, dkx_dr0, dkthetax_dr0, dkz_dr0, dkphi_dr0])
        self.dk_dr0[rigid] = 0.0
        self.dk_dh = np.array([dkx_dh, dkthetax_dh, dkx_dh, dkthetax_dh, dkz_dh, dkphi_dh])
        self.dk_dh[rigid] = 0.0

    def provideJ(self, params, unknowns, resids):

        self.J = [self.dk_dr0, self.dk_dh]

        return J




if __name__ == '__main__':

    z = np.linspace(1.0, 5, 100)
    nPoints = len(z)

    # J = prob.check_total_derivatives(out_stream=None)
    # print J

    #print prob['p1.z']

    p1 = PowerWind(nPoints)
    p1.z = z
    p1.Uref = 10.0
    p1.zref = 100.0
    p1.z0 = 1.0
    shearExp = 0.2
    p1.compute(shearExp)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(p1.z, p1.U, label='Power')




    z = np.linspace(1.0, 5, 100)
    nPoints = len(z)


    p1 = LogWind(nPoints)
    p1.z = z
    p1.Uref = 10.0
    p1.zref = 100.0
    p1.z0 = 1.0
    z_roughness = 0.5
    p1.compute(z_roughness)


    import matplotlib.pyplot as plt
    plt.plot(p1.z, p1.U, label='Log')
    plt.legend()
    plt.show()
