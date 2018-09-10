from __future__ import print_function
import numpy as np
from openmdao.api import Component, Group, IndepVarComp

from commonse.utilities import hstack, vstack
from commonse.csystem import DirectionVector
from commonse import gravity

#TODO CHECK


class RNAMass(Component):
    def __init__(self):

        super(RNAMass, self).__init__()
        # variables
        self.add_param('blades_mass', 0.0, units='kg', desc='mass of all blade')
        self.add_param('hub_mass', 0.0, units='kg', desc='mass of hub')
        self.add_param('nac_mass', 0.0, units='kg', desc='mass of nacelle')

        self.add_param('hub_cm', np.zeros(3), units='m', desc='location of hub center of mass relative to tower top in yaw-aligned c.s.')
        self.add_param('nac_cm', np.zeros(3), units='m', desc='location of nacelle center of mass relative to tower top in yaw-aligned c.s.')

        # order for all moments of inertia is (xx, yy, zz, xy, xz, yz) in the yaw-aligned coorinate system
        self.add_param('blades_I', np.zeros(6), units='kg*m**2', desc='mass moments of inertia of all blades about hub center')
        self.add_param('hub_I', np.zeros(6), units='kg*m**2', desc='mass moments of inertia of hub about its center of mass')
        self.add_param('nac_I', np.zeros(6), units='kg*m**2', desc='mass moments of inertia of nacelle about its center of mass')

        # outputs
        self.add_output('rotor_mass', 0.0, units='kg', desc='mass of blades and hub')
        self.add_output('rna_mass', 0.0, units='kg', desc='total mass of RNA')
        self.add_output('rna_cm', np.zeros(3), units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')
        self.add_output('rna_I_TT', np.zeros(6), units='kg*m**2', desc='mass moments of inertia of RNA about tower top in yaw-aligned coordinate system')



    def _assembleI(self, I):
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I[0], I[1], I[2], I[3], I[4], I[5] 
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


    def _unassembleI(self, I):
        return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])


    def solve_nonlinear(self, params, unknowns, resids):

        rotor_mass = params['blades_mass'] + params['hub_mass']
        nac_mass = params['nac_mass']

        # rna mass
        unknowns['rotor_mass'] = rotor_mass
        unknowns['rna_mass'] = rotor_mass + nac_mass

        # rna cm
        unknowns['rna_cm'] = (rotor_mass*params['hub_cm'] + nac_mass*params['nac_cm'])/unknowns['rna_mass']

        #TODO check if the use of assembleI and unassembleI functions are correct
        # rna I
        blades_I = self._assembleI(params['blades_I'])
        hub_I = self._assembleI(params['hub_I'])
        nac_I = self._assembleI(params['nac_I'])
        rotor_I = blades_I + hub_I

        R = params['hub_cm']
        rotor_I_TT = rotor_I + rotor_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        R = params['nac_cm']
        nac_I_TT = nac_I + params['nac_mass']*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        unknowns['rna_I_TT'] = self._unassembleI(rotor_I_TT + nac_I_TT)


    def linearize(self, params, unknowns, resids):

        blades_mass = params['blades_mass']
        hub_mass = params['hub_mass']
        nac_mass = params['nac_mass']
        hub_cm = params['hub_cm']
        nac_cm = params['nac_cm']
        hub_I = params['hub_I']
        nac_I = params['nac_I']
        rna_mass = unknowns['rna_mass']
        rotor_mass = blades_mass+hub_mass

        J = {}

        # mass
        J['rna_mass', 'blades_mass'] = 1.0
        J['rna_mass', 'hub_mass'] = 1.0
        J['rna_mass', 'nac_mass'] = 1.0
        J['rna_mass', 'hub_cm'] = np.zeros(3)
        J['rna_mass', 'nac_cm'] = np.zeros(3)
        J['rna_mass', 'blades_I'] = np.zeros(6)
        J['rna_mass', 'hub_I'] = np.zeros(6)
        J['rna_mass', 'nac_I'] = np.zeros(6)


        # cm
        numerator = (blades_mass+hub_mass)*hub_cm+nac_mass*nac_cm

        J['rna_cm', 'blades_mass'] = (rna_mass*hub_cm-numerator)/rna_mass**2
        J['rna_cm', 'hub_mass'] = (rna_mass*hub_cm-numerator)/rna_mass**2
        J['rna_cm', 'nac_mass'] = (rna_mass*nac_cm-numerator)/rna_mass**2
        J['rna_cm', 'hub_cm'] = rotor_mass/rna_mass*np.eye(3)
        J['rna_cm', 'nac_cm'] = nac_mass/rna_mass*np.eye(3)
        J['rna_cm', 'blades_I'] = np.zeros(3, 6)
        J['rna_cm', 'hub_I'] = np.zeros(3, 6)
        J['rna_cm', 'nac_I'] = np.zeros(3, 6)


        # I
        R = hub_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        J['rna_I', 'blades_mass'] = const
        J['rna_I', 'hub_mass'] = const
        dI_drx = rotor_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = rotor_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = rotor_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        J['rna_I', 'hub_cm'] = np.vstack([dI_drx, dI_dry, dI_drz]).T

        R = nac_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        J['rna_I', 'nac_mass'] = const
        dI_drx = nac_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = nac_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = nac_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        J['rna_I', 'nac_cm'] = np.vstack([dI_drx, dI_dry, dI_drz]).T

        J['rna_I', 'blades_I'] = np.eye(6)
        J['rna_I', 'hub_I'] = np.eye(6)
        J['rna_I', 'nac_I'] = np.eye(6)

        return J


class RotorLoads(Component):
    def __init__(self):
        super(RotorLoads, self).__init__()

        # variables
        self.add_param('F', np.array([0.0, 0.0, 0.0]), desc='forces in hub-aligned coordinate system')
        self.add_param('M', np.array([0.0, 0.0, 0.0]), desc='moments in hub-aligned coordinate system')
        self.add_param('r_hub', np.zeros(3), desc='position of rotor hub relative to tower top in yaw-aligned c.s.')
        self.add_param('rna_mass', 0.0, units='kg', desc='mass of rotor nacelle assembly')
        self.add_param('rna_cm', np.zeros(3), units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')

        self.add_param('rna_weightM', True, units=None, desc='Flag to indicate whether or not the RNA weight should be considered.\
                          An upwind overhang may lead to unconservative estimates due to the P-Delta effect(suggest not using). For downwind turbines set to True. ', pass_by_obj=True)

        # # These are used for backwards compatibility - do not use
        # T = Float(iotype='in', desc='thrust in hub-aligned coordinate system')  # THIS MEANS STILL YAWED THOUGH (Shaft tilt)
        # Q = Float(iotype='in', desc='torque in hub-aligned coordinate system')

        # parameters
        self.add_param('downwind', False, pass_by_obj=True)
        self.add_param('tilt', 0.0, units='deg')

        # out
        self.add_output('top_F', np.zeros(3))  # in yaw-aligned
        self.add_output('top_M', np.zeros(3))


    def solve_nonlinear(self, params, unknowns, resids):

        F = params['F']
        M = params['M']

        F = DirectionVector.fromArray(F).hubToYaw(params['tilt'])
        M = DirectionVector.fromArray(M).hubToYaw(params['tilt'])

        # change x-direction if downwind
        r_hub = np.copy(params['r_hub'])
        rna_cm = np.copy(params['rna_cm'])
        if params['downwind']:
            r_hub[0] *= -1
            rna_cm[0] *= -1
        r_hub = DirectionVector.fromArray(r_hub)
        rna_cm = DirectionVector.fromArray(rna_cm)
        self.save_rhub = r_hub
        self.save_rcm = rna_cm

        # aerodynamic moments
        M = M + r_hub.cross(F)
        self.saveF = F


        # add weight loads
        F_w = DirectionVector(0.0, 0.0, -params['rna_mass']*gravity)
        M_w = rna_cm.cross(F_w)
        self.saveF_w = F_w

        Fout = F + F_w

        if params['rna_weightM']:
            Mout = M + M_w
        else:
            Mout = M
            #REMOVE WEIGHT EFFECT TO ACCOUNT FOR P-Delta Effect
            print("!!!! No weight effect on rotor moments -TowerSE  !!!!")

        # put back in array
        unknowns['top_F'] = np.array([Fout.x, Fout.y, Fout.z])
        unknowns['top_M'] = np.array([Mout.x, Mout.y, Mout.z])


    def list_deriv_vars(self):

        inputs = ('F', 'M', 'r_hub', 'rna_mass', 'rna_cm')
        outputs = ('top_F', 'top_M')

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        dF = DirectionVector.fromArray(params['F']).hubToYaw(params['tilt'])
        dFx, dFy, dFz = dF.dx, dF.dy, dF.dz

        dtopF_dFx = np.array([dFx['dx'], dFy['dx'], dFz['dx']])
        dtopF_dFy = np.array([dFx['dy'], dFy['dy'], dFz['dy']])
        dtopF_dFz = np.array([dFx['dz'], dFy['dz'], dFz['dz']])
        dtopF_dF = hstack([dtopF_dFx, dtopF_dFy, dtopF_dFz])
        dtopF_w_dm = np.array([0.0, 0.0, -self.g])

        #dtopF = hstack([dtopF_dF, np.zeros((3, 6)), dtopF_w_dm, np.zeros((3, 3))])


        dM = DirectionVector.fromArray(params['M']).hubToYaw(params['tilt'])
        dMx, dMy, dMz = dM.dx, dM.dy, dM.dz
        dMxcross, dMycross, dMzcross = self.save_rhub.cross_deriv(self.saveF, 'dr', 'dF')

        dtopM_dMx = np.array([dMx['dx'], dMy['dx'], dMz['dx']])
        dtopM_dMy = np.array([dMx['dy'], dMy['dy'], dMz['dy']])
        dtopM_dMz = np.array([dMx['dz'], dMy['dz'], dMz['dz']])
        dtopM_dM = hstack([dtopM_dMx, dtopM_dMy, dtopM_dMz])
        dM_dF = np.array([dMxcross['dF'], dMycross['dF'], dMzcross['dF']])

        dtopM_dFx = np.dot(dM_dF, dtopF_dFx)
        dtopM_dFy = np.dot(dM_dF, dtopF_dFy)
        dtopM_dFz = np.dot(dM_dF, dtopF_dFz)
        dtopM_dF = hstack([dtopM_dFx, dtopM_dFy, dtopM_dFz])
        dtopM_dr = np.array([dMxcross['dr'], dMycross['dr'], dMzcross['dr']])

        dMx_w_cross, dMy_w_cross, dMz_w_cross = self.save_rcm.cross_deriv(self.saveF_w, 'dr', 'dF')

        if params['rna_weightM']:
            dtopM_drnacm = np.array([dMx_w_cross['dr'], dMy_w_cross['dr'], dMz_w_cross['dr']])
            dtopM_dF_w = np.array([dMx_w_cross['dF'], dMy_w_cross['dF'], dMz_w_cross['dF']])
        else:
            dtopM_drnacm = np.zeros((3, 3))
            dtopM_dF_w = np.zeros((3, 3))
        dtopM_dm = np.dot(dtopM_dF_w, dtopF_w_dm)

        if params['downwind']:
            dtopM_dr[:, 0] *= -1
            dtopM_drnacm[:, 0] *= -1

        #dtopM = hstack([dtopM_dF, dtopM_dM, dtopM_dr, dtopM_dm, dtopM_drnacm])

        J = {}
        J['top_F', 'F'] = dtopF_dF
        J['top_F', 'M'] = np.zeros((3, 3))
        J['top_F', 'r_hub'] = np.zeros((3, 3))
        J['top_F', 'rna_mass'] = dtopF_w_dm
        J['top_F', 'rna_cm'] = np.zeros((3, 3))

        J['top_M', 'F'] = dtopM_dF
        J['top_M', 'M'] = dtopM_dM
        J['top_M', 'r_hub'] = dtopM_dr
        J['top_M', 'rna_mass'] = dtopM_dm
        J['top_M', 'rna_cm'] = dtopM_drnacm

        return J


class RNA(Group):
    def __init__(self, nLC):
        super(RNA, self).__init__()

        self.add('mass', RNAMass(), promotes=['*'])
        for k in range(nLC):
            lc = '' if nLC==1 else str(k+1)
            self.add('loads'+lc, RotorLoads(), promotes=['rna_mass','rna_cm','r_hub','rna_weightM','downwind','tilt'])

        self.connect('hub_cm', 'r_hub')
        
