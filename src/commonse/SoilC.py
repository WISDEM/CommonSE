#-------------------------------------------------------------------------------
# Name:        SoilC.py
# Purpose:     This module contains the basic Soil Class
#
# Author:      rdamiani
#
# Created:     26/01/2014 - Based on BuildMPtwr.py SoilC and ReadSoilInfo
# Copyright:   (c) rdamiani 2014
# Licence:     <Apache>
#-------------------------------------------------------------------------------
import warnings
from math import * #this is to allow the input file to have mathematical/trig formulas in their param expressions
import numpy as np

#______________________________________________________________________________#
def ReadSoilInfo(SoilInfoFile):
    """This function creates a soil object, filling main params and then \n
        reads in a file containing class data prepared for Matlab processing.\n
        Thus it spits out an augmented version of the object.
         """
         #Now read soil data
    if isinstance(SoilInfoFile,str) and os.path.isfile(SoilInfoFile): #it means we need to read a file
         soil=SoilC() #This creates the object with some default values
         execfile(SoilInfoFile) #This simply includes the file where I have my input deck also available to matlab
         attr1= (attr for attr in dir(soil) if not attr.startswith('_'))
        #Update MP attributes based on input parameters

    else:
         warnings.warn('SoilInfoFile was not found !')

    return soil

#______________________________________________________________________________#
class SoilC():
            #Soil Class
    def __init__(self,  **kwargs):
        """This function creates a soil class.\n
        """
    #start by setting default values in a dictionary fashion
        Pprms={'zbots':-np.array([3.,5.,7.,15.,30.,50.]), 'gammas':np.array([10000,10000,10000,10000,10000,10000]),\
        'cus':np.array([60000,60000,60000,60000,60000,60000]), 'phis':np.array([36.,33.,26.,37.,35.,37.5]),\
        'delta':25.}
        prms=Pprms.copy()
        prms.update(kwargs)
        for key in kwargs:
            if key not in Pprms:
                setattr(self,key,kwargs[key])  #In case user set something else not included in the Pprms
        for key in Pprms:
            #self.key=prms[key] This does not work instead,beats me
            setattr(self,key,prms[key]) #This takes care of updating values without overburdening the MP class with other stuff which belong to MP
#______________________________________________________________________________#


if __name__ == '__main__':
    soil=SoilC(sndflg=False)

    print 'Soil z-bottoms [m]:',soil.zbots,' ;\n','Undrained shear strength [N/m^2]:',soil.cus,' ;\n',\
          'Unit weight  [N/m^3]:', soil.gammas,' ;\n','Friction angles [deg]:', soil.phis,' ;\n',\
          'Pile-soil friction angle [deg]:', soil.delta
