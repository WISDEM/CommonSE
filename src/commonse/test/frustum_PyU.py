import __init__
import numpy as np
import numpy.testing as npt
import unittest
import frustum as f
from commonse import eps

myones = np.ones((100,))
rb = 4.0
rt = 2.0
t  = 0.1
h  = 3.0

class TestFrustum(unittest.TestCase):

    def testFrustumVol(self):
        
        V = np.pi/3*h * (rb**2 + rt**2 + rb * rt)

        # Test volume- scalar and vector inputs
        self.assertEqual(f.frustumVol(rb, rt, h, False), V)
        self.assertEqual(f.frustumVol(2*rb, 2*rt, h, True), V)
        npt.assert_equal(f.frustumVol(rb*myones, rt*myones, h*myones, False), V*myones)
        npt.assert_equal(f.frustumVol(2*rb*myones, 2*rt*myones, h*myones, True), V*myones)

    def testFrustumCG_solid(self):
        
        cg_solid = h/4 * (rb**2 + 3*rt**2 + 2*rb*rt) / (rb**2 + rt**2 + rb*rt)

        # Test cg of solid- scalar and vector inputs
        self.assertEqual(f.frustumCG(rb, rt, h, False), cg_solid)
        self.assertEqual(f.frustumCG(2*rb, 2*rt, h, True), cg_solid)
        npt.assert_equal(f.frustumCG(rb*myones, rt*myones, h*myones, False), cg_solid*myones)
        npt.assert_equal(f.frustumCG(2*rb*myones, 2*rt*myones, h*myones, True), cg_solid*myones)

    def testFrustum_shell(self):

        # In limit of thickness approaching radius, should recover regular formulas
        self.assertEqual(f.frustumShellVol(rb, rt, rb, rt, h, False), f.frustumVol(rb, rt, h, False))
        self.assertEqual(f.frustumShellVol(2*rb, 2*rt, rb, rt, h, True), f.frustumVol(rb, rt, h, False))

        self.assertEqual(f.frustumShellCG(rb, rt, rb, rt, h, False), f.frustumCG(rb, rt, h, False))
        self.assertEqual(f.frustumShellCG(2*rb, 2*rt, rb, rt, h, True), f.frustumCG(rb, rt, h, False))

        self.assertEqual(f.frustumShellIzz(rb, rt, rb, rt, h, False), f.frustumIzz(rb, rt, h, False))
        self.assertEqual(f.frustumShellIzz(2*rb, 2*rt, rb, rt, h, True), f.frustumIzz(rb, rt, h, False))

        self.assertAlmostEqual(f.frustumShellIxx(rb, rt, rb-eps, rt-eps, h, False), f.frustumIxx(rb, rt, h, False))
        self.assertAlmostEqual(f.frustumShellIxx(2*rb, 2*rt, rb-eps, rt-eps, h, True), f.frustumIxx(rb, rt, h, False))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFrustum))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
