import numpy as np
import numpy.testing as npt
import unittest
import commonse.utilities as util

npts = 100
myones = np.ones((npts,))

class TestAny(unittest.TestCase):

            
    def testNodal2Sectional(self):
        x,dx = util.nodal2sectional(np.array([8.0, 10.0, 12.0]))
        npt.assert_equal(x, np.array([9.0, 11.0]))
        npt.assert_equal(dx, np.array([[0.5, 0.5, 0.0],[0.0, 0.5, 0.5]]))

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAny))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
