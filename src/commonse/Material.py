#-------------------------------------------------------------------------------
# Name:        Material.py
# Purpose: This module contains the material class definition
#
# Author:      rdamiani
#
# Created:     04/11/2013
# Copyright:   (c) rdamiani 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    """Material Class"""
    #The main fct is written to test the class
    mat=Material(E=3.5e5,rho=8500)
    mat1=Material(name='grout')
    print 'Example Returning mat and mat1 as 2 objects'
    return mat,mat1


class Material: #isotropic for the time being
    def __init__(self,**kwargs):

        prms={'name':'ASTM992_steel','E':2.1e11,'nu':0.33,'rho':8502., 'fy':345.e6} #SI Units
        prms.update(kwargs) #update in case user put some new params in
        for key in prms:  #Initialize material object with these parameters, possibly updated by user' stuff
            setattr(self,key,prms[key])

        #Fill in the rest of the fields user may have skipped

        #Predefined materials
        steel={'name':'ASTM992_steel','E':2.1e11,'nu':0.33,'rho':7805., 'fy':345.e6} #SI Units
        grout={'name':'Grout','E':3.9e10,'nu':0.33,'rho':2500., 'fy':345.e6} #SI Units

        if ((prms['name'].lower() == 'steel') or (prms['name'].lower() == 'astm992_steel')):

            for key in prms:
                if not(key in kwargs):  #I need to operate on parameters not set by user
                    setattr(self,key,steel[key])

        elif ((prms['name'].lower() == 'concrete') or (prms['name'].lower() == 'grout')):

            for key in prms:
                if not(key in kwargs):
                    setattr(self,key,grout[key])

        self.G=self.E/(2*(1+self.nu))  #isotropic


if __name__ == '__main__':
    mat,mat1=main()
