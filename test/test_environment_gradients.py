#!/usr/bin/env python
# encoding: utf-8
"""
test_environment_gradients.py

Created by Andrew Ning on 2013-12-18.
Copyright (c) NREL. All rights reserved.
"""


import unittest
import numpy as np
from commonse.utilities import check_gradient
from commonse.environment import PowerWind, LogWind, LinearWaves, TowerSoil
from openmdao.api import pyOptSparseDriver, Problem, Group
import sys
from openmdao.api import ScipyOptimizer, IndepVarComp


class TestPowerWind(unittest.TestCase):


    def setUp(self):

        z = np.linspace(0.0, 100.0, 20)
        nPoints = len(z)
        Uref = 10.0
        zref = 100.0
        z0 = 0.001 #Fails when z0 = 0, What to do here?
        shearExp = 0.2
        betaWind = 0.0

        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('z', z))
        root.add('p2', IndepVarComp('zref', zref))
        root.add('p3', IndepVarComp('Uref', Uref))
        root.add('p', PowerWind(nPoints))


        root.connect('p1.z', 'p.z')
        root.connect('p2.zref', 'p.zref')
        root.connect('p3.Uref', 'p.Uref')

        prob.driver.add_objective('p.U', scaler=1E-6)

        prob.driver.add_desvar('p1.z', lower=np.ones(nPoints), upper=np.ones(nPoints)*1000, scaler=1E-6)
        prob.driver.add_desvar('p2.zref', lower=0, upper=1000, scaler=1E-6)
        prob.driver.add_desvar('p3.Uref', lower=0, upper=1000, scaler=1E-6)

        prob.setup()

        prob['p.z0'] = z0
        prob['p.shearExp'] = shearExp
        prob['p.betaWind'] = betaWind

        prob.run()

        print prob['p.U']

        self.J = prob.check_total_derivatives(out_stream=None)

    def test_U(self):
        np.testing.assert_allclose(self.J[('p.U', 'p3.Uref')]['J_fwd'], self.J[('p.U', 'p3.Uref')]['J_fd'], 1e-6, 1e-6)
        np.testing.assert_allclose(self.J[('p.U', 'p2.zref')]['J_fwd'], self.J[('p.U', 'p2.zref')]['J_fd'], 1e-6, 1e-6)
        np.testing.assert_allclose(self.J[('p.U', 'p1.z')]['J_fwd'], self.J[('p.U', 'p1.z')]['J_fd'], 1e-6, 1e-6)



class TestLogWind(unittest.TestCase):


    def setUp(self):

        z = np.linspace(0.0, 100.0, 20)
        nPoints = len(z)
        Uref = 10.0
        zref = 100.0
        z0 = 0. #Fails when z0 = 0
        betaWind = 0.0

        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('z', z))
        root.add('p2', IndepVarComp('zref', Uref))
        root.add('p3', IndepVarComp('Uref', zref))
        root.add('p', LogWind(nPoints))


        root.connect('p1.z', 'p.z')
        root.connect('p2.zref', 'p.zref')
        root.connect('p3.Uref', 'p.Uref')

        prob.driver.add_objective('p.U', scaler=1E-6)

        prob.driver.add_desvar('p1.z', lower=np.ones(nPoints), upper=np.ones(nPoints)*1000, scaler=1E-6)
        prob.driver.add_desvar('p2.zref', lower=0, upper=1000, scaler=1E-6)
        prob.driver.add_desvar('p3.Uref', lower=0, upper=1000, scaler=1E-6)

        prob.setup()

        prob['p.z0'] = z0
        prob['p.betaWind'] = betaWind

        prob.run()

        print prob['p.U']

        self.J = prob.check_total_derivatives(out_stream=None)

    def test_U(self):
        np.testing.assert_allclose(self.J[('p.U', 'p3.Uref')]['J_fwd'], self.J[('p.U', 'p3.Uref')]['J_fd'], 1e-6, 1e-6)
        np.testing.assert_allclose(self.J[('p.U', 'p2.zref')]['J_fwd'], self.J[('p.U', 'p2.zref')]['J_fd'], 1e-6, 1e-6)
        np.testing.assert_allclose(self.J[('p.U', 'p1.z')]['J_fwd'], self.J[('p.U', 'p1.z')]['J_fd'], 1e-6, 1e-6)


class TestLinearWave(unittest.TestCase):


    def setUp(self):

        Uc = 7.0
        z_surface = 20.0
        hs = 10.0
        T = 2.0
        z_floor = 0.1
        betaWave = 3.0
        z = np.linspace(z_floor, z_surface, 20)
        nPoints = len(z)

        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('z', z))
        root.add('p2', IndepVarComp('Uc', Uc))
        root.add('p', LinearWaves(nPoints))

        root.connect('p1.z', 'p.z')
        root.connect('p2.Uc', 'p.Uc')

        prob.driver.add_objective('p.U', scaler=1E-6)

        prob.driver.add_desvar('p1.z', lower=np.ones(nPoints), upper=np.ones(nPoints)*1000, scaler=1E-6)
        prob.driver.add_desvar('p2.Uc', lower=0, upper=1000, scaler=1E-6)

        prob.setup()

        prob['p.z_surface'] = z_surface
        prob['p.z_floor'] = z_floor
        prob['p.hmax'] = hs
        prob['p.T'] = T
        prob['p.betaWave'] = betaWave

        prob.run()

        print prob['p.U']

        self.J = prob.check_total_derivatives(out_stream=None)

        print self.J

    def test_U(self):

        np.testing.assert_allclose(self.J[('p.U', 'p2.Uc')]['J_fwd'], self.J[('p.U', 'p2.Uc')]['J_fd'], 1e-6, 1e-6)
        #np.testing.assert_allclose(self.J[('p.U', 'p1.z')]['J_fwd'], self.J[('p.U', 'p1.z')]['J_fd'], 1e-6, 1e-6)


class TestSoil(unittest.TestCase):

    def setUp(self):

        r0 = 10.0
        depth = 30.0
        G = 140e6
        nu = 0.4
        rigid = [False, False, False, False, False, False]

        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('r0', r0))
        root.add('p2', IndepVarComp('depth', depth))
        root.add('p', TowerSoil())

        root.connect('p1.r0', 'p.r0')
        root.connect('p2.depth', 'p.depth')

        prob.driver.add_objective('p.k', scaler=1E-6)

        prob.driver.add_desvar('p1.r0', lower=0, upper=1000, scaler=1E-6)
        prob.driver.add_desvar('p2.depth', lower=0, upper=1000, scaler=1E-6)

        prob.setup()

        prob['p.G'] = G
        prob['p.nu'] = nu
        prob['p.rigid'] = rigid

        prob.run()

        self.J = prob.check_total_derivatives(out_stream=None)

        print self.J

    def test_k(self):

        np.testing.assert_allclose(self.J[('p.k', 'p1.r0')]['J_fwd'], self.J[('p.k', 'p1.r0')]['J_fd'], 1e-6, 1e-6)
        np.testing.assert_allclose(self.J[('p.k', 'p1.depth')]['J_fwd'], self.J[('p.k', 'p1.depth')]['J_fd'], 1e-6, 1e-6)


"""
class TestLinearWave(unittest.TestCase):


    def test1(self):

        lw = LinearWaves()
        lw.Uc = 7.0
        lw.z_surface = 20.0
        lw.hs = 10.0
        lw.T = 2.0
        lw.z_floor = 0.0
        lw.betaWave = 3.0
        lw.z = np.linspace(0.0, 20.0, 20)

        names, errors = check_gradient(lw)

        tol = 1e-4
        for name, err in zip(names, errors):

            if name in ('d_U[0] / d_z[0]', 'd_U[19] / d_z[19]', 'd_A[0] / d_z[0]', 'd_A[19] / d_z[19]'):
                continue  # the boundaries are not differentiable across bounds. these nodes must not move

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e


    def test2(self):

        lw = LinearWaves()
        lw.Uc = 5.0
        lw.z_surface = 20.0
        lw.hs = 2.0
        lw.T = 10.0
        lw.z_floor = 0.0
        lw.betaWave = 3.0
        lw.z = np.linspace(-5.0, 50.0, 20)

        names, errors = check_gradient(lw)

        tol = 1e-6
        for name, err in zip(names, errors):

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e

"""

if __name__ == '__main__':
    unittest.main()
