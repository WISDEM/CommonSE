
import numpy as np
from openmdao.api import Component
from commonse.tube import CylindricalShellProperties

from commonse import gravity, eps
import commonse.frustum as frustum
from commonse.UtilizationSupplement import hoopStressEurocode, hoopStress
from commonse.utilities import assembleI, unassembleI
import pyframe3dd.frame3dd as frame3dd


# -----------------
#  Components
# -----------------

#TODO need to check the length of each array
class CylinderDiscretization(Component):
    """discretize geometry into finite element nodes"""

    #inputs

    def __init__(self, nPoints, nRefine):
        
        super(CylinderDiscretization, self).__init__()

        self.nRefine = nRefine
        nFull = nRefine * (nPoints-1) + 1
        
         # variables
        self.add_param('section_height', np.zeros(nPoints-1), units='m', desc='parameterized section heights along cylinder')
        self.add_param('diameter', np.zeros(nPoints), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('wall_thickness', np.zeros(nPoints), units='m', desc='shell thickness at corresponding locations')

        #out
        self.add_output('z_param', np.zeros(nPoints), units='m', desc='parameterized locations along cylinder, linear lofting between')
        self.add_output('z_full', np.zeros(nFull), units='m', desc='locations along cylinder')
        self.add_output('d_full', np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_output('t_full', np.zeros(nFull), units='m', desc='shell thickness at corresponding locations')
        # Convenience outputs for export to other modules
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['z_param'] = np.r_[0.0, np.cumsum(params['section_height'])]
        z_full = np.array([])
        for k in range(unknowns['z_param'].size-1):
            zref = np.linspace(unknowns['z_param'][k], unknowns['z_param'][k+1], self.nRefine+1)
            z_full = np.append(z_full, zref)
        unknowns['z_full']  = np.unique(z_full) #np.linspace(unknowns['z_param'][0], unknowns['z_param'][-1], self.nFull) 
        unknowns['d_full']  = np.interp(unknowns['z_full'], unknowns['z_param'], params['diameter'])
        unknowns['t_full']  = np.interp(unknowns['z_full'], unknowns['z_param'], params['wall_thickness'])
        

class CylinderMass(Component):

    def __init__(self, nPoints):
        super(CylinderMass, self).__init__()
        
        self.add_param('d_full', val=np.zeros(nPoints), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('t_full', val=np.zeros(nPoints), units='m', desc='shell thickness at corresponding locations')
        self.add_param('z_full', val=np.zeros(nPoints), units='m', desc='parameterized locations along cylinder, linear lofting between')
        self.add_param('material_density', 0.0, units='kg/m**3', desc='material density')
        self.add_param('outfitting_factor', val=0.0, desc='Multiplier that accounts for secondary structure mass inside of cylinder')
        
        self.add_output('mass', val=np.zeros(nPoints-1), units='kg', desc='Total cylinder mass')
        self.add_output('center_of_mass', val=0.0, units='m', desc='z-position of center of mass of cylinder')
        self.add_output('section_center_of_mass', val=np.zeros(nPoints-1), units='m', desc='z position of center of mass of each can in the cylinder')
        self.add_output('I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of cylinder about base [xx yy zz xy xz yz]')
        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables for thickness and average radius at each can interface
        Tb  = params['t_full'][:-1]
        Tt  = params['t_full'][1:]
        Rb  = 0.5*params['d_full'][:-1]
        Rt  = 0.5*params['d_full'][1:]
        zz  = params['z_full']
        H   = np.diff(zz)
        rho = params['material_density'] * params['outfitting_factor']

        # Total mass of cylinder
        V_shell = frustum.frustumShellVol(Rb, Rt, Tb, Tt, H)
        unknowns['mass'] = rho * V_shell
        
        # Center of mass of each can/section
        cm_section = zz[:-1] + frustum.frustumShellCG(Rb, Rt, Tb, Tt, H)
        unknowns['section_center_of_mass'] = cm_section

        # Center of mass of cylinder
        V_shell += eps
        unknowns['center_of_mass'] = np.dot(V_shell, cm_section) / V_shell.sum()

        # Moments of inertia
        Izz_section = frustum.frustumShellIzz(Rb, Rt, Tb, Tt, H)
        Ixx_section = Iyy_section = frustum.frustumShellIxx(Rb, Rt, Tb, Tt, H)

        # Sum up each cylinder section using parallel axis theorem
        I_base = np.zeros((3,3))
        for k in xrange(Izz_section.size):
            R = np.array([0.0, 0.0, cm_section[k]])
            Icg = assembleI( [Ixx_section[k], Iyy_section[k], Izz_section[k], 0.0, 0.0, 0.0] )

            I_base += Icg + V_shell[k]*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
            
        # All of the mass and volume terms need to be multiplied by density
        I_base *= rho

        unknowns['I_base'] = unassembleI(I_base)
        



#@implement_base(CylinderFromCSProps)
class CylinderFrame3DD(Component):

    def __init__(self, npts, nK, nMass, nPL):

        super(CylinderFrame3DD, self).__init__()

        # cross-sectional data along cylinder.
        self.add_param('z', np.zeros(npts), units='m', desc='location along cylinder. start at bottom and go to top')
        self.add_param('Az', np.zeros(npts), units='m**2', desc='cross-sectional area')
        self.add_param('Asx', np.zeros(npts), units='m**2', desc='x shear area')
        self.add_param('Asy', np.zeros(npts), units='m**2', desc='y shear area')
        self.add_param('Jz', np.zeros(npts), units='m**4', desc='polar moment of inertia')
        self.add_param('Ixx', np.zeros(npts), units='m**4', desc='area moment of inertia about x-axis')
        self.add_param('Iyy', np.zeros(npts), units='m**4', desc='area moment of inertia about y-axis')

        self.add_param('E', val=0.0, units='N/m**2', desc='modulus of elasticity')
        self.add_param('G', val=0.0, units='N/m**2', desc='shear modulus')
        self.add_param('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_param('sigma_y', val=0.0, units='N/m**2', desc='yield stress')
        self.add_param('L_reinforced', val=0.0, units='m')

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_param('d', np.zeros(npts), units='m', desc='effective cylinder diameter for section')
        self.add_param('t', np.zeros(npts), units='m', desc='effective shell thickness for section')

        # spring reaction data.  Use float('inf') for rigid constraints.
        self.add_param('kidx', np.zeros(nK, dtype=np.int_), desc='indices of z where external stiffness reactions should be applied.', pass_by_obj=True)
        self.add_param('kx', np.zeros(nK), units='m', desc='spring stiffness in x-direction', pass_by_obj=True)
        self.add_param('ky', np.zeros(nK), units='m', desc='spring stiffness in y-direction', pass_by_obj=True)
        self.add_param('kz', np.zeros(nK), units='m', desc='spring stiffness in z-direction', pass_by_obj=True)
        self.add_param('ktx', np.zeros(nK), units='m', desc='spring stiffness in theta_x-rotation', pass_by_obj=True)
        self.add_param('kty', np.zeros(nK), units='m', desc='spring stiffness in theta_y-rotation', pass_by_obj=True)
        self.add_param('ktz', np.zeros(nK), units='m', desc='spring stiffness in theta_z-rotation', pass_by_obj=True)

        # extra mass
        self.add_param('midx', np.zeros(nMass, dtype=np.int_), desc='indices where added mass should be applied.', pass_by_obj=True)
        self.add_param('m', np.zeros(nMass), units='kg', desc='added mass')
        self.add_param('mIxx', np.zeros(nMass), units='kg*m**2', desc='x mass moment of inertia about some point p')
        self.add_param('mIyy', np.zeros(nMass), units='kg*m**2', desc='y mass moment of inertia about some point p')
        self.add_param('mIzz', np.zeros(nMass), units='kg*m**2', desc='z mass moment of inertia about some point p')
        self.add_param('mIxy', np.zeros(nMass), units='kg*m**2', desc='xy mass moment of inertia about some point p')
        self.add_param('mIxz', np.zeros(nMass), units='kg*m**2', desc='xz mass moment of inertia about some point p')
        self.add_param('mIyz', np.zeros(nMass), units='kg*m**2', desc='yz mass moment of inertia about some point p')
        self.add_param('mrhox', np.zeros(nMass), units='m', desc='x-location of p relative to node')
        self.add_param('mrhoy', np.zeros(nMass), units='m', desc='y-location of p relative to node')
        self.add_param('mrhoz', np.zeros(nMass), units='m', desc='z-location of p relative to node')
        self.add_param('addGravityLoadForExtraMass', True, desc='add gravitational load', pass_by_obj=True)

        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        self.add_param('plidx', np.zeros(nPL, dtype=np.int_), desc='indices where point loads should be applied.', pass_by_obj=True)
        self.add_param('Fx', np.zeros(nPL), units='N', desc='point force in x-direction')
        self.add_param('Fy', np.zeros(nPL), units='N', desc='point force in y-direction')
        self.add_param('Fz', np.zeros(nPL), units='N', desc='point force in z-direction')
        self.add_param('Mxx', np.zeros(nPL), units='N*m', desc='point moment about x-axis')
        self.add_param('Myy', np.zeros(nPL), units='N*m', desc='point moment about y-axis')
        self.add_param('Mzz', np.zeros(nPL), units='N*m', desc='point moment about z-axis')

        # combined wind-water distributed loads
        self.add_param('Px', np.zeros(npts), units='N/m', desc='force per unit length in x-direction')
        self.add_param('Py', np.zeros(npts), units='N/m', desc='force per unit length in y-direction')
        self.add_param('Pz', np.zeros(npts), units='N/m', desc='force per unit length in z-direction')
        self.add_param('qdyn', np.zeros(npts), units='N/m**2', desc='dynamic pressure')

        # options
        self.add_param('shear', True, desc='include shear deformation', pass_by_obj=True)
        self.add_param('geom', False, desc='include geometric stiffness', pass_by_obj=True)
        self.add_param('dx', 5.0, desc='z-axis increment for internal forces')
        self.add_param('nM', 2, desc='number of desired dynamic modes of vibration (below only necessary if nM > 0)', pass_by_obj=True)
        self.add_param('Mmethod', 1, desc='1: subspace Jacobi, 2: Stodola', pass_by_obj=True)
        self.add_param('lump', 0, desc='0: consistent mass, 1: lumped mass matrix', pass_by_obj=True)
        self.add_param('tol', 1e-9, desc='mode shape tolerance')
        self.add_param('shift', 0.0, desc='shift value ... for unrestrained structures')

        # outputs
        self.add_output('mass', 0.0)
        self.add_output('f1', 0.0, units='Hz', desc='First natural frequency')
        self.add_output('f2', 0.0, units='Hz', desc='Second natural frequency')
        self.add_output('top_deflection', 0.0, units='m', desc='Deflection of cylinder top in yaw-aligned +x direction')
        self.add_output('Fz_out', np.zeros(npts), units='N', desc='Axial foce in vertical z-direction in cylinder structure.')
        self.add_output('Vx_out', np.zeros(npts), units='N', desc='Shear force in x-direction in cylinder structure.')
        self.add_output('Vy_out', np.zeros(npts), units='N', desc='Shear force in y-direction in cylinder structure.')
        self.add_output('Mxx_out', np.zeros(npts), units='N*m', desc='Moment about x-axis in cylinder structure.')
        self.add_output('Myy_out', np.zeros(npts), units='N*m', desc='Moment about y-axis in cylinder structure.')
        self.add_output('Mzz_out', np.zeros(npts), units='N*m', desc='Moment about z-axis in cylinder structure.')
        self.add_output('base_F', val=np.zeros(3), units='N', desc='Total force on cylinder')
        self.add_output('base_M', val=np.zeros(3), units='N*m', desc='Total moment on cylinder measured at base')

        self.add_output('axial_stress', np.zeros(npts), units='N/m**2', desc='Axial stress in cylinder structure')
        self.add_output('shear_stress', np.zeros(npts), units='N/m**2', desc='Shear stress in cylinder structure')
        self.add_output('hoop_stress', np.zeros(npts), units='N/m**2', desc='Hoop stress in cylinder structure calculated with simple method used in API standards')
        self.add_output('hoop_stress_euro', np.zeros(npts), units='N/m**2', desc='Hoop stress in cylinder structure calculated with Eurocode method')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

        
    def solve_nonlinear(self, params, unknowns, resids):

        # ------- node data ----------------
        z = params['z']
        n = len(z)
        node = np.arange(1, n+1)
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)

        nodes = frame3dd.NodeData(node, x, y, z, r)
        # -----------------------------------

        # ------ reaction data ------------

        # rigid base
        node = params['kidx'] + np.ones(len(params['kidx']), dtype=np.int_)  # add one because 0-based index but 1-based node numbering
        rigid = np.inf

        reactions = frame3dd.ReactionData(node, params['kx'], params['ky'], params['kz'], params['ktx'], params['kty'], params['ktz'], rigid)
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n+1)

        roll = np.zeros(n-1)

        # average across element b.c. frame3dd uses constant section elements
        # TODO: Use nodal2sectional
        Az = 0.5*(params['Az'][:-1] + params['Az'][1:])
        Asx = 0.5*(params['Asx'][:-1] + params['Asx'][1:])
        Asy = 0.5*(params['Asy'][:-1] + params['Asy'][1:])
        Jz = 0.5*(params['Jz'][:-1] + params['Jz'][1:])
        Ixx = 0.5*(params['Ixx'][:-1] + params['Ixx'][1:])
        Iyy = 0.5*(params['Iyy'][:-1] + params['Iyy'][1:])
        E = params['E']*np.ones(Az.shape)
        G = params['G']*np.ones(Az.shape)
        rho = params['rho']*np.ones(Az.shape)

        elements = frame3dd.ElementData(element, N1, N2, Az, Asx, Asy, Jz, Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------


        # ------ options ------------
        options = frame3dd.Options(params['shear'], params['geom'], params['dx'])
        # -----------------------------------

        # initialize frame3dd object
        cylinder = frame3dd.Frame(nodes, reactions, elements, options)


        # ------ add extra mass ------------

        # extra node inertia data
        N = params['midx'] + np.ones(len(params['midx']), dtype=np.int_)

        cylinder.changeExtraNodeMass(N, params['m'], params['mIxx'], params['mIyy'], params['mIzz'], params['mIxy'], params['mIxz'], params['mIyz'],
            params['mrhox'], params['mrhoy'], params['mrhoz'], params['addGravityLoadForExtraMass'])

        # ------------------------------------

        # ------- enable dynamic analysis ----------
        cylinder.enableDynamics(params['nM'], params['Mmethod'], params['lump'], params['tol'], params['shift'])
        # ----------------------------

        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -gravity

        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # point loads
        nF = params['plidx'] + np.ones(len(params['plidx']), dtype=int)
        load.changePointLoads(nF, params['Fx'], params['Fy'], params['Fz'], params['Mxx'], params['Myy'], params['Mzz'])

        # distributed loads
        Px, Py, Pz = params['Pz'], params['Py'], -params['Px']  # switch to local c.s.
        z = params['z']

        # trapezoidally distributed loads
        EL = np.arange(1, n)
        xx1 = xy1 = xz1 = np.zeros(n-1)
        xx2 = xy2 = xz2 = np.diff(z) - 1e-6  # subtract small number b.c. of precision
        wx1 = Px[:-1]
        wx2 = Px[1:]
        wy1 = Py[:-1]
        wy2 = Py[1:]
        wz1 = Pz[:-1]
        wz2 = Pz[1:]

        load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

        cylinder.addLoadCase(load)

        # -----------------------------------
        # run the analysis
        displacements, forces, reactions, internalForces, mass, modal = cylinder.run()
        iCase = 0

        # mass
        unknowns['mass'] = mass.struct_mass

        # natural frequncies
        unknowns['f1'] = modal.freq[0]
        unknowns['f2'] = modal.freq[1]

        # deflections due to loading (from cylinder top and wind/wave loads)
        unknowns['top_deflection'] = displacements.dx[iCase, n-1]  # in yaw-aligned direction

        # shear and bending (convert from local to global c.s.)
        Fz = forces.Nx[iCase, :]
        Vy = forces.Vy[iCase, :]
        Vx = -forces.Vz[iCase, :]

        Mzz = forces.Txx[iCase, :]
        Myy = forces.Myy[iCase, :]
        Mxx = -forces.Mzz[iCase, :]

        # one per element (first negative b.c. need reaction)
        Fz = np.r_[-reactions.Fz.sum(), Fz[1::2]]
        Vy = np.r_[-reactions.Fy.sum(), Vy[1::2]]
        Vx = np.r_[-reactions.Fx.sum(), Vx[1::2]]

        Mzz = np.r_[-reactions.Mzz.sum(), Mzz[1::2]]
        Myy = np.r_[-reactions.Myy.sum(), Myy[1::2]]
        Mxx = np.r_[-reactions.Mxx.sum(), Mxx[1::2]]

        # Record total forces and moments
        unknowns['base_F'] = np.array([Vx[0], Vy[0], Fz[0]])
        unknowns['base_M'] = np.array([Mxx[0], Myy[0], Mzz[0]])

        unknowns['Fz_out']  = Fz
        unknowns['Vx_out']  = Vx
        unknowns['Vy_out']  = Vy
        unknowns['Mxx_out'] = Mxx
        unknowns['Myy_out'] = Myy
        unknowns['Mzz_out'] = Mzz
        # axial and shear stress
        ##R = self.d/2.0
        ##x_stress = R*np.cos(self.theta_stress)
        ##y_stress = R*np.sin(self.theta_stress)
        ##axial_stress = Fz/self.Az + Mxx/self.Ixx*y_stress - Myy/self.Iyy*x_stress
#        V = Vy*x_stress/R - Vx*y_stress/R  # shear stress orthogonal to direction x,y
#        shear_stress = 2. * V / self.Az  # coefficient of 2 for a hollow circular section, but should be conservative for other shapes
        unknowns['axial_stress'] = Fz/params['Az'] - np.sqrt(Mxx**2+Myy**2)/params['Iyy']*params['d']/2.0  #More conservative, just use the tilted bending and add total max shear as well at the same point, if you do not like it go back to the previous lines

        unknowns['shear_stress'] = 2. * np.sqrt(Vx**2+Vy**2) / params['Az'] # coefficient of 2 for a hollow circular section, but should be conservative for other shapes

        # hoop_stress (Eurocode method)
        L_reinforced = params['L_reinforced'] * np.ones(Fz.shape)
        unknowns['hoop_stress_euro'] = hoopStressEurocode(params['z'], params['d'], params['t'], L_reinforced, params['qdyn'])

        # Simpler hoop stress used in API calculations
        unknowns['hoop_stress'] = hoopStress(params['d'], params['t'], params['qdyn'])
