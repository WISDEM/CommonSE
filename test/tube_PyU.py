import numpy as np
import numpy.testing as npt
import unittest
from commonse.tube import Tube, CylindricalShellProperties

npts = 100
myones = np.ones((npts,))

class TestTube(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['d'] = 2*5.0
        self.params['t'] = 1.0
        
        self.mytube = CylindricalShellProperties(npts)


    def testTubeProperties(self):
        T = Tube(2*5.0, 1.0)
        
        self.assertAlmostEqual(T.Area, np.pi*9.0)
        self.assertAlmostEqual(T.Jxx,  np.pi*369.0/4.0)
        self.assertAlmostEqual(T.Jyy,  np.pi*369.0/4.0)
        self.assertAlmostEqual(T.J0,  np.pi*369.0/2.0)
        self.assertAlmostEqual(T.S,  np.pi*369.0/4.0/5.0)
        self.assertAlmostEqual(T.C,  np.pi*369.0/2.0/5.0)
    
    def testOutputsIncremental(self):
        self.mytube.solve_nonlinear(self.params, self.unknowns, self.resid)

        self.assertAlmostEqual(self.unknowns['Az'], np.pi*9.0)
        self.assertAlmostEqual(self.unknowns['Ixx'],  np.pi*369.0/4.0)
        self.assertAlmostEqual(self.unknowns['Iyy'],  np.pi*369.0/4.0)
        self.assertAlmostEqual(self.unknowns['Jz'],  np.pi*369.0/2.0)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTube))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
