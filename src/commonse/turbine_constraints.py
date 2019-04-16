from openmdao.api import Group, Component
from .csystem import DirectionVector
from .utilities import interp_with_deriv
from commonse import NFREQ
import numpy as np


class TowerModes(Component):
    def __init__(self):
        super(TowerModes, self).__init__()

        self.add_param('tower_freq', val=np.zeros(NFREQ), units='Hz', desc='First natural frequencies of tower (and substructure)')
        self.add_param('rotor_omega', val=0.0, units='rpm', desc='rated rotor rotation speed')
        self.add_param('gamma_freq', val=0.0, desc='partial safety factor for fatigue')
        self.add_param('blade_number', 3, desc='number of rotor blades', pass_by_obj=True)

        self.add_output('frequency3P_margin_low', val=np.zeros(NFREQ), desc='Upper bound constraint of tower/structure frequency to blade passing frequency with margin')
        self.add_output('frequency3P_margin_high', val=np.zeros(NFREQ), desc='Lower bound constraint of tower/structure frequency to blade passing frequency with margin')
        self.add_output('frequency1P_margin_low', val=np.zeros(NFREQ), desc='Upper bound constraint of tower/structure frequency to rotor frequency with margin')
        self.add_output('frequency1P_margin_high', val=np.zeros(NFREQ), desc='Lower bound constraint of tower/structure frequency to rotor frequency with margin')

    def solve_nonlinear(self, params, unknowns, resids):
        freq_struct = params['tower_freq']
        gamma       = params['gamma_freq']
        oneP        = (params['rotor_omega']/60.0)
        oneP_high   = oneP * gamma
        oneP_low    = oneP / gamma
        threeP      = oneP * params['blade_number']
        threeP_high = threeP * gamma
        threeP_low  = threeP / gamma
        
        # Compute margins between (N/3)P and structural frequencies
        indicator_high = threeP_high * np.ones(freq_struct.shape)
        indicator_high[freq_struct < threeP_low] = 1e-16
        unknowns['frequency3P_margin_high'] = freq_struct / indicator_high

        indicator_low = threeP_low * np.ones(freq_struct.shape)
        indicator_low[freq_struct > threeP_high] = 1e30
        unknowns['frequency3P_margin_low']  = freq_struct / indicator_low

        # Compute margins between 1P and structural frequencies
        indicator_high = oneP_high * np.ones(freq_struct.shape)
        indicator_high[freq_struct < oneP_low] = 1e-16
        unknowns['frequency1P_margin_high'] = freq_struct / indicator_high

        indicator_low = oneP_low * np.ones(freq_struct.shape)
        indicator_low[freq_struct > oneP_high] = 1e30
        unknowns['frequency1P_margin_low']  = freq_struct / indicator_low

    
class MaxTipDeflection(Component):

    def __init__(self, nFullTow):
        super(MaxTipDeflection, self).__init__()


        self.add_param('downwind',       val=False, pass_by_obj=True)
        self.add_param('tip_deflection', val=0.0,               units='m',  desc='Blade tip deflection in yaw x-direction')
        self.add_param('Rtip',           val=0.0,               units='m',  desc='Blade tip location in z_b')
        self.add_param('precurveTip',    val=0.0,               units='m',  desc='Blade tip location in x_b')
        self.add_param('presweepTip',    val=0.0,               units='m',  desc='Blade tip location in y_b')
        self.add_param('precone',        val=0.0,               units='deg',desc='Rotor precone angle')
        self.add_param('tilt',           val=0.0,               units='deg',desc='Nacelle uptilt angle')
        self.add_param('hub_cm',         val=np.zeros(3),       units='m',  desc='Location of hub relative to tower-top in yaw-aligned c.s.')
        self.add_param('z_full',         val=np.zeros(nFullTow),units='m',  desc='z-coordinates of tower at fine-section nodes')
        self.add_param('d_full',         val=np.zeros(nFullTow),units='m',  desc='Diameter of tower at fine-section nodes')
        self.add_param('gamma_m',        val=0.0, desc='safety factor on materials')

        self.add_output('tip_deflection_ratio',      val=0.0,           desc='Ratio of blade tip deflectiion towardsa the tower and clearance between undeflected blade tip and tower')
        self.add_output('blade_tip_tower_clearance', val=0.0, units='m',desc='Clearance between undeflected blade tip and tower in x-direction of yaw c.s.')
        self.add_output('ground_clearance',          val=0.0, units='m',desc='Distance between blade tip and ground')

        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        z_tower = params['z_full']
        d_tower = params['d_full']
        hub_cm  = params['hub_cm']
        precone = params['precone']
        tilt    = params['tilt']
        delta   = params['tip_deflection']
        upwind  = not params['downwind']
        

        # Coordinates of blade tip in yaw c.s.
        blade_yaw = DirectionVector(params['precurveTip'], params['presweepTip'], params['Rtip']).\
                    bladeToAzimuth(precone).azimuthToHub(180.0).hubToYaw(tilt)

        # Find the radius of tower where blade passes
        z_interp = z_tower[-1] + hub_cm[2] + blade_yaw.z
        d_interp, ddinterp_dzinterp, ddinterp_dtowerz, ddinterp_dtowerd = interp_with_deriv(z_interp, z_tower, d_tower)
        r_interp = 0.5 * d_interp
        drinterp_dzinterp = 0.5 * ddinterp_dzinterp
        drinterp_dtowerz  = 0.5 * ddinterp_dtowerz
        drinterp_dtowerd  = 0.5 * ddinterp_dtowerd

        # Max deflection before strike
        if upwind:
            parked_margin = -hub_cm[0] - blade_yaw.x - r_interp
        else:
            parked_margin = hub_cm[0] + blade_yaw.x - r_interp
        unknowns['blade_tip_tower_clearance']   = parked_margin
        unknowns['tip_deflection_ratio']        = delta * params['gamma_m'] / parked_margin
            
        # ground clearance
        unknowns['ground_clearance'] = z_interp


    
class TurbineConstraints(Group):
    def __init__(self, nFull):
        super(TurbineConstraints, self).__init__()

        self.add('modes', TowerModes(), promotes=['*'])
        self.add('tipd', MaxTipDeflection(nFull), promotes=['*'])
