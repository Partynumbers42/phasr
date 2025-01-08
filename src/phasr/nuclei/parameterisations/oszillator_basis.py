from ... import constants
from ..base import nucleus_base

import numpy as np
pi = np.pi

class nucleus_osz(nucleus_base):
    def __init__(self,name,Z,A,Cs,**args): #,R_cut=None,rho_cut=None
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "shell-model"
        self.Cs=Cs
        self.update_dependencies()

# TODO / to reimagine, also how the parametrisation is stored, is Cs the best choice/ name, etc.
# also how to store all the formfactors

# do something like this for the other multipoles 
# for key in [Fch0,Fch2,..,Fmag1,FM0p,...]: setattr(self,key,fct[key])