import __init__
import numpy as np
import numpy.testing as npt
import unittest
import Frustum

myones = np.ones((100,))
rb = 4.0
rt = 2.0
t  = 0.1
h  = 3.0

class TestFrustum(unittest.TestCase):

    def testFrustumVol(self):
        
        V = np.pi/3*h * (rb**2 + rt**2 + rb * rt)

        # Test volume- scalar and vector inputs
        self.assertEqual(Frustum.frustumVol_radius(rb, rt, h), V)
        self.assertEqual(Frustum.frustumVol_diameter(2*rb, 2*rt, h), V)
        npt.assert_equal(Frustum.frustumVol_radius(rb*myones, rt*myones, h*myones), V*myones)
        npt.assert_equal(Frustum.frustumVol_diameter(2*rb*myones, 2*rt*myones, h*myones), V*myones)

    def testFrustumCG_solid(self):
        
        cg_solid = h/4 * (rb**2 + 3*rt**2 + 2*rb*rt) / (rb**2 + rt**2 + rb*rt)

        # Test cg of solid- scalar and vector inputs
        self.assertEqual(Frustum.frustumCG_radius(rb, rt, h), cg_solid)
        self.assertEqual(Frustum.frustumCG_diameter(2*rb, 2*rt, h), cg_solid)
        npt.assert_equal(Frustum.frustumCG_radius(rb*myones, rt*myones, h*myones), cg_solid*myones)
        npt.assert_equal(Frustum.frustumCG_diameter(2*rb*myones, 2*rt*myones, h*myones), cg_solid*myones)

    def testFrustumCG_shell(self):
        
        cg_shell = h * (rb + 2*rt) /3 / (rb+rt)

        # Test cg of shell- scalar and vector inputs
        self.assertEqual(Frustum.frustumShellCG_radius(rb, rt, h), cg_shell)
        self.assertEqual(Frustum.frustumShellCG_diameter(2*rb, 2*rt, h), cg_shell)
        npt.assert_equal(Frustum.frustumShellCG_radius(rb*myones, rt*myones, h*myones), cg_shell*myones)
        npt.assert_equal(Frustum.frustumShellCG_diameter(2*rb*myones, 2*rt*myones, h*myones), cg_shell*myones)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFrustum))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
