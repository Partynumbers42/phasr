from ... import constants
from ..base import nucleus_base
from .numerical import nucleus_num

import numpy as np
pi = np.pi

from mpmath import polylog 

class nucleus_fermi(nucleus_base):
    def __init__(self,name,Z,A,c,z,**args): 
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "fermi"
        self.c=c
        self.z=z
        if "w" in args:
            self.w=args["w"]
            self.nucleus_type+="3p"
        else:
            self.w=0
            self.nucleus_type+="2p"
        self.total_charge=Z
        #
        self.nucleus_num = nucleus_num(name+self.nucleus_type,Z=Z,A=A,charge_density=self.charge_density)
        #
        self.update_dependencies()

    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        self.charge_radius_sq = charge_radius_sq_fermi(self.c,self.z,self.w)
        self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        #
        if hasattr(self,'electric_potential') and (not hasattr(self,'Vmin')):
            self.Vmin = self.nucleus_num.Vmin
    
    def set_electric_field_from_charge_density(self):
        self.nucleus_num.set_electric_field_from_charge_density()
        self.electric_field = self.nucleus_num.electric_field

    def set_electric_potential_from_electric_field(self):
        self.nucleus_num.set_electric_potential_from_electric_field()
        self.electric_potential = self.nucleus_num.electric_potential
    
    def set_form_factor_from_charge_density(self):
        self.nucleus_num.set_form_factor_from_charge_density()
        self.form_factor = self.nucleus_num.form_factor

    def fill_gaps(self):
        self.nucleus_num.fill_gaps()
        self.form_factor = self.nucleus_num.form_factor
        self.electric_field = self.nucleus_num.electric_field
        self.electric_potential = self.nucleus_num.electric_potential
        self.update_dependencies()

    def charge_density(self,r):
        return charge_density_fermi(r,self.c,self.z,self.w,self.total_charge)
    
    # def form_factor(self,r):
    #     return form_factor_fermi(r,self.c,self.z,self.w)

def charge_density_fermi(r,c,z,w,Z):
    rho0 = float(Z/(4*pi*(-2*z**3*polylog(3,-np.exp(c/z))-24*w*z**5*polylog(5,-np.exp(c/z))/c**2)))
    return rho0*(1+ w*r**2/c**2)*np.exp(-(r-c)/z)/(1+np.exp(-(r-c)/z))

def charge_radius_sq_fermi(c,z,w):
    return float(12*(c**2*z**2*polylog(5,-np.exp(c/z))+30*w*z**4*polylog(7,-np.exp(c/z)))/(c**2*polylog(3,-np.exp(c/z))+12*w*z**2*polylog(5,-np.exp(c/z))))

# from mpmath import lerchphi as lerchphi0
# def lerchphi00(z,s,a):
#     return complex(lerchphi0(z,s,a))
# lerchphi = np.vectorize(lerchphi00,excluded=[0,1])
#
# def form_factor_fermi(q,c,z,w):
# two slow to be usable
# correct Riemann sheet
#     q=q/constants.hc
#     lp2 = np.real(1j*( lerchphi(-np.exp(c/z),2,1-1j*q*z) - lerchphi(-np.exp(c/z),2,1+1j*q*z) ))
#     lp4 = np.real(1j*( lerchphi(-np.exp(c/z),4,1-1j*q*z) - lerchphi(-np.exp(c/z),4,1+1j*q*z) ))
#     return np.exp(c/z)*(c**2*lp2 + 6*w*z**2*lp4)/(4*q*z*(c**2*float(polylog(3,-np.exp(c/z)))+12*w*z**2*float(polylog(5,-np.exp(c/z)))))