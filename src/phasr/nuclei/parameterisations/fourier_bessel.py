from ..base import nucleus_base

class nucleus_FB(nucleus_base):
    def __init__(self,name,Z,A,ai,R,**args): #,R_cut=None,rho_cut=None
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "Fourier-Bessel"
        self.ai=ai
        self.R=R
        self.N=len(ai)