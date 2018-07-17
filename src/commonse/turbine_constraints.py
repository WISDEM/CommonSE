from openmdao.api import Group, Component
from csystem import DirectionVector
from utilities import interp_with_deriv
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

    def __init__(self, nFull):
        super(MaxTipDeflection, self).__init__()


        self.add_param('downwind', val=False, pass_by_obj=True)
        self.add_param('tip_deflection', val=0.0, units='m', desc='tip deflection in yaw x-direction')
        self.add_param('Rtip', val=0.0, units='m', desc='tip location in z_b')
        self.add_param('precurveTip', val=0.0, units='m', desc='tip location in x_b')
        self.add_param('presweepTip', val=0.0, units='m', desc='tip location in y_b')
        self.add_param('precone', val=0.0, units='deg', desc='precone angle')
        self.add_param('tilt', val=0.0, units='deg', desc='tilt angle')
        self.add_param('hub_tt', val=np.zeros(3), units='m', desc='location of hub relative to tower-top in yaw-aligned c.s.')
        self.add_param('tower_z', val=np.zeros(nFull), units='m', desc='z-coordinates of tower at fine-section nodes')
        self.add_param('tower_d', val=np.zeros(nFull), units='m', desc='diameter of tower at fine-section nodes')
        self.add_param('gamma_m', val=0.0, desc='safety factor on materials')

        self.add_output('tip_deflection_ratio', val=0.0, units='m', desc='clearance between undeflected blade and tower')
        self.add_output('ground_clearance', val=0.0, units='m', desc='distance between blade tip and ground')

        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        z_tower = params['tower_z']
        d_tower = params['tower_d']
        hub_tt  = params['hub_tt']
        precone = params['precone']
        tilt    = params['tilt']
        delta   = params['tip_deflection']
        upwind  = not params['downwind']
        
        # coordinates of blade tip in yaw c.s.
        blade_yaw = DirectionVector(params['precurveTip'], params['presweepTip'], params['Rtip']).\
                    bladeToAzimuth(precone).azimuthToHub(180.0).hubToYaw(tilt)

        # find corresponding radius of tower
        z_interp = z_tower[-1] + hub_tt[2] + blade_yaw.z
        d_interp, ddinterp_dzinterp, ddinterp_dtowerz, ddinterp_dtowerd = interp_with_deriv(z_interp, z_tower, d_tower)
        r_interp = 0.5 * d_interp
        drinterp_dzinterp = 0.5 * ddinterp_dzinterp
        drinterp_dtowerz  = 0.5 * ddinterp_dtowerz
        drinterp_dtowerd  = 0.5 * ddinterp_dtowerd

        # max deflection before strike
        if upwind:
            parked_margin = -hub_tt[0] - blade_yaw.x - r_interp
        else:
            parked_margin = hub_tt[0] + blade_yaw.x - r_interp
        unknowns['tip_deflection_ratio'] = delta * params['gamma_m'] / parked_margin
            
        # ground clearance
        unknowns['ground_clearance'] = z_interp

        # Derivatives
        self.J = {}
        dbyx = blade_yaw.dx
        # dbyy = blade_yaw.dy
        dbyz = blade_yaw.dz
        dtdr_dpark = -unknowns['tip_deflection_ratio'] / parked_margin
        
        # Tip_deflection and gamma_m
        self.J['tip_deflection_ratio','tip_deflection'] = params['gamma_m'] / parked_margin
        self.J['tip_deflection_ratio','gamma_m']        = delta / parked_margin
        self.J['ground_clearance','tip_deflection']     = 0.0
        self.J['ground_clearance','gamma_m']            = 0.0
        
        # Rtip
        drinterp_dRtip = drinterp_dzinterp * dbyz['dz']
        if upwind:
            self.J['tip_deflection_ratio','Rtip'] = dtdr_dpark * (-dbyx['dz'] - drinterp_dRtip)
        else:
            self.J['tip_deflection_ratio','Rtip'] = dtdr_dpark * (dbyx['dz'] - drinterp_dRtip)
        self.J['ground_clearance','Rtip'] = dbyz['dz']

        # precurveTip
        drinterp_dprecurveTip = drinterp_dzinterp * dbyz['dx']
        if upwind:
            self.J['tip_deflection_ratio','precurveTip'] = dtdr_dpark * (-dbyx['dx'] - drinterp_dprecurveTip)
        else:
            self.J['tip_deflection_ratio','precurveTip'] = dtdr_dpark * (dbyx['dx'] - drinterp_dprecurveTip)
        self.J['ground_clearance','precurveTip'] = dbyz['dx']

        # presweep
        drinterp_dpresweepTip = drinterp_dzinterp * dbyz['dy']
        if upwind:
            self.J['tip_deflection_ratio','presweepTip'] = dtdr_dpark * (-dbyx['dy'] - drinterp_dpresweepTip)
        else:
            self.J['tip_deflection_ratio','presweepTip'] = dtdr_dpark * (dbyx['dy'] - drinterp_dpresweepTip)
        self.J['ground_clearance','presweepTip'] = dbyz['dy']


        # precone
        drinterp_dprecone = drinterp_dzinterp * dbyz['dprecone']
        if upwind:
            self.J['tip_deflection_ratio','precone'] = dtdr_dpark * (-dbyx['dprecone'] - drinterp_dprecone)
        else:
            self.J['tip_deflection_ratio','precone'] = dtdr_dpark * (dbyx['dprecone'] - drinterp_dprecone)
        self.J['ground_clearance','precone'] = dbyz['dprecone']

        # tilt
        drinterp_dtilt = drinterp_dzinterp * dbyz['dtilt']
        if upwind:
            self.J['tip_deflection_ratio','tilt'] = dtdr_dpark * (-dbyx['dtilt'] - drinterp_dtilt)
        else:
            self.J['tip_deflection_ratio','tilt'] = dtdr_dpark * (dbyx['dtilt'] - drinterp_dtilt)
        self.J['ground_clearance','tilt'] = dbyz['dtilt']

        # hubtt
        drinterp_dhubtt = drinterp_dzinterp * np.array([0.0, 0.0, 1.0])
        if upwind:
            self.J['tip_deflection_ratio','hub_tt'] = dtdr_dpark * (np.array([-1.0, 0.0, 0.0]) - drinterp_dhubtt)
        else:
            self.J['tip_deflection_ratio','hub_tt'] = dtdr_dpark * (np.array([1.0, 0.0, 0.0]) - drinterp_dhubtt)
        self.J['ground_clearance','hub_tt']   = np.array([0.0, 0.0, 1.0])

        # tower_z
        self.J['tip_deflection_ratio','tower_z'] = -dtdr_dpark * drinterp_dtowerz
        self.J['ground_clearance','tower_z']   = np.zeros(z_tower.shape)
        self.J['ground_clearance','tower_z'][-1] = 1.0

        # tower_d
        self.J['tip_deflection_ratio','tower_d'] = -dtdr_dpark * drinterp_dtowerd
        self.J['ground_clearance','tower_d']   = np.zeros(d_tower.shape)

    def list_deriv_vars(self):

        inputs = ('Rtip', 'precurveTip', 'presweepTip', 'precone', 'tilt', 'hub_tt', 'tower_z', 'tower_d','tip_deflection','gamma_m')
        outputs = ('tip_deflection_ratio', 'ground_clearance')

        return inputs, outputs

    def linearize(self, params, unknowns, resids):
        return self.J


    
class TurbineConstraints(Group):
    def __init__(self, nFull):
        super(TurbineConstraints, self).__init__()

        self.add('modes', TowerModes(), promotes=['*'])
        self.add('tipd', MaxTipDeflection(nFull), promotes=['*'])
