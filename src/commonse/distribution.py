import numpy as np
from math import gamma
from openmdao.api import Component

# ---------------------
# Map Design Variables to Discretization
# ---------------------

class PDFBase(Component):
    """probability distribution function"""
    def __init__(self, nspline):
        super(PDFBase, self).__init__()
        self.add_param('x', shape=nspline)

        self.add_output('f', shape=nspline)


class CDFBase(Component):
    """cumulative distribution function"""
    def __init__(self, nspline):
        super(CDFBase, self).__init__()

        self.add_param('x', shape=nspline,  units='m/s', desc='corresponding reference height')
        self.add_param('k', shape=1, desc='shape or form factor')

        self.add_output('F', shape=nspline, units='m/s', desc='magnitude of wind speed at each z location')


class WeibullCDF(CDFBase):
    def __init__(self, n):
        super(WeibullCDF, self).__init__(n)
        """Weibull cumulative distribution function"""

        self.add_param('A', shape=1, desc='scale factor')
        
	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['F'] = 1.0 - np.exp(-(params['x']/params['A'])**params['k'])

    def list_deriv_vars(self):
        inputs = ('x',)
        outputs = ('F',)

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        x = params['x']
        A = params['A']
        k = params['k']
        J = {}
        J['F', 'x'] = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)
        return J


class WeibullWithMeanCDF(CDFBase):
    def __init__(self, n):
        super(WeibullWithMeanCDF, self).__init__(n)
        """Weibull cumulative distribution function"""

        self.add_param('xbar', shape=1, desc='mean value of distribution')

	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):

        A = params['xbar'] / gamma(1.0 + 1.0/params['k'])

        unknowns['F'] = 1.0 - np.exp(-(params['x']/A)**params['k'])


    def list_deriv_vars(self):

        inputs = ('x', 'xbar')
        outputs = ('F',)

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        x = params['x']
        k = params['k']
        A = params['xbar'] / gamma(1.0 + 1.0/k)
        dx = np.diag(np.exp(-(x/A)**k)*(x/A)**(k-1)*k/A)
        dxbar = -np.exp(-(x/A)**k)*(x/A)**(k-1)*k*x/A**2/gamma(1.0 + 1.0/k)
        J = {}
        J['F', 'x'] = dx
        J['F', 'xbar'] = dxbar

        return J

class RayleighCDF(CDFBase):
    def __init__(self, n):
        super(RayleighCDF,  self).__init__(n)

        # variables
        self.add_param('xbar', shape=1, units='m/s', desc='reference wind speed (usually at hub height)')

	self.deriv_options['form'] = 'central'
	self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['F'] = 1.0 - np.exp(-np.pi/4.0*(params['x']/params['xbar'])**2)

    def linearize(self, params, unknowns, resids):

        x = params['x']
        xbar = params['xbar']
        dx = np.diag(np.exp(-np.pi/4.0*(x/xbar)**2)*np.pi*x/(2.0*xbar**2))
        dxbar = -np.exp(-np.pi/4.0*(x/xbar)**2)*np.pi*x**2/(2.0*xbar**3)
        
        J = {}
        J['F', 'x'] = dx
        J['F', 'xbar'] = dxbar
        
        return J
