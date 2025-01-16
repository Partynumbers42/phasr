from ... import constants
from ..base import nucleus_base
#from .numerical import nucleus_num

import numpy as np
pi = np.pi

from scipy.special import erf

class nucleus_gauss(nucleus_base):
    def __init__(self,name,Z,A,b,**args): 
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "gauss"
        self.b = b
        #
        self.total_charge=self.Z
        #
        self.update_dependencies()

    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        self.charge_radius_sq = charge_radius_sq_gauss(self.b)
        self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        self.Vmin = electric_potential_V0_gauss(self.b,self.total_charge)
    
    def charge_density(self,r):
        return charge_density_gauss(r,self.b,self.total_charge)
    
    def form_factor(self,r):
        return form_factor_gauss(r,self.b)
    
    def electric_field(self,r):
        return electric_field_gauss(r,self.b,self.total_charge)
    
    def electric_potential(self,r):
        return electric_potential_gauss(r,self.b,self.total_charge)

def charge_density_gauss(r,b,Z):
    return Z*np.exp(-(r/b)**2)/(pi*np.sqrt(pi)*b**3)

def charge_radius_sq_gauss(b):
    return (3./2.)*b**2

def form_factor_gauss(q,b):
    return np.exp(-b**2*q**2/4)

def electric_field_gauss(r,b,Z):
    return Z*(-2*np.exp(-r**2/b**2)*r+b*np.sqrt(pi)*erf(r/b))/(4*b*np.sqrt(pi**3)*r**2)

def electric_potential_gauss(r,b,Z):
    return -Z*erf(r/b)/(4*pi*r)

def electric_potential_V0_gauss(b,Z):
    return -Z/(2*b*np.sqrt(pi**3))