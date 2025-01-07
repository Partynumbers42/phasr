from ... import constants
from ..base import nucleus_base

import numpy as np
pi = np.pi

class nucleus_coulomb(nucleus_base):
    def __init__(self,name,Z,A,**args): 
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "Coulomb"
        self.update_dependencies()

    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        #
        self.total_charge = self.Z
        #
        self.charge_radius_sq = 0
        self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        #
        self.Vmin = -np.inf
        #
        if (self.k_barrett is not None) and (self.alpha_barrett is not None):
            self.barrett_moment = 0
    
    def charge_density(self,r):
        print('Warning: Not a practical function/implementation')
        return charge_density_coulomb(r)
    
    def electric_field(self,r):
        return electric_field_coulomb(r,self.total_charge,alpha_el=constants.alpha_el)
    
    def electric_potential(self,r):
        return electric_potential_coulomb(r,self.total_charge,alpha_el=constants.alpha_el)
    
    def form_factor(self,q):
        return form_factor_coulomb(q)
    
def charge_density_coulomb(r):
    r_arr = np.atleast_1d(r)
    rho=np.zeros(len(r_arr))
    mask_r = r_arr==0
    if np.any(mask_r):
        rho[mask_r] = np.inf
    if np.isscalar(r):
        rho=rho[0]
    return rho

def electric_field_coulomb(r,Z,alpha_el=constants.alpha_el):
    r_arr = np.atleast_1d(r)
    E=np.sqrt(alpha_el/(4*pi))*Z/r_arr**2
    if np.isscalar(r):
        E=E[0]
    return E

def electric_potential_coulomb(r,Z,alpha_el=constants.alpha_el):
    r_arr = np.atleast_1d(r)
    V=-Z*alpha_el/r_arr
    if np.isscalar(r):
        V=V[0]
    return V

def form_factor_coulomb(r):
    r_arr = np.atleast_1d(r)
    F = np.ones(len(r_arr))
    if np.isscalar(r):
        F=F[0]
    return F