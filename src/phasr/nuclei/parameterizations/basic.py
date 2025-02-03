from ... import constants
from ..base import nucleus_base
#from .numerical import nucleus_num

import numpy as np
pi = np.pi

from scipy.special import erf, spherical_jn

# gauss parameterisation
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
    return Z*np.exp(-(r/b)**2)/(np.sqrt(pi**3)*b**3)

def charge_radius_sq_gauss(b):
    return (3./2.)*b**2

def form_factor_gauss(q,b):
    q=q/constants.hc
    return np.exp(-b**2*q**2/4)

def electric_field_gauss(r,b,Z,alpha_el=constants.alpha_el):
    return Z*np.sqrt(4*pi*alpha_el)*(-2*np.exp(-r**2/b**2)*r+b*np.sqrt(pi)*erf(r/b))/(4*b*np.sqrt(pi**3)*r**2)

def electric_potential_gauss(r,b,Z,alpha_el=constants.alpha_el):
    return -Z*4*pi*alpha_el*erf(r/b)/(4*pi*r)

def electric_potential_V0_gauss(b,Z,alpha_el=constants.alpha_el):
    return -Z*4*pi*alpha_el/(2*b*np.sqrt(pi**3))

# uniform parameterisation
class nucleus_uniform(nucleus_base):
    def __init__(self,name,Z,A,rc,**args): 
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "uniform"
        self.rc = rc
        #
        self.total_charge=self.Z
        #
        self.update_dependencies()

    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        self.charge_radius_sq = charge_radius_sq_uniform(self.rc)
        self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        self.Vmin = electric_potential_V0_uniform(self.rc,self.total_charge)
    
    def charge_density(self,r):
        return charge_density_uniform(r,self.rc,self.total_charge)
    
    def form_factor(self,r):
        return form_factor_uniform(r,self.rc)
    
    def electric_field(self,r):
        return electric_field_uniform(r,self.rc,self.total_charge)
    
    def electric_potential(self,r):
        return electric_potential_uniform(r,self.rc,self.total_charge)

def charge_density_uniform(r,rc,Z):
    r_arr = np.atleast_1d(r)
    rho=np.zeros(len(r_arr))
    mask_r = r_arr<=rc
    if np.any(mask_r):
        rho[mask_r] = 3*Z/(4*pi*rc**3)
    if np.isscalar(r):
        rho=rho[0]
    return rho

def charge_radius_sq_uniform(rc):
    return (3./5.)*rc**2

def form_factor_uniform(q,rc):
    q=q/constants.hc
    return 3/(q*rc)*spherical_jn(1,q*rc)

def electric_field_uniform(r,rc,Z,alpha_el=constants.alpha_el):
    r_arr = np.atleast_1d(r)
    El=np.zeros(len(r_arr))
    mask_r = r_arr<=rc
    if np.any(mask_r):
        El[mask_r] = Z*np.sqrt(alpha_el/(4*pi))*r_arr[mask_r]/rc**3
    if np.any(~mask_r):
        El[~mask_r] = Z*np.sqrt(alpha_el/(4*pi))/r[~mask_r]**2
    if np.isscalar(r):
        El=El[0]
    return El

def electric_potential_uniform(r,rc,Z,alpha_el=constants.alpha_el):
    r_arr = np.atleast_1d(r)
    V=np.zeros(len(r_arr))
    mask_r = r_arr<=rc
    if np.any(mask_r):
        V[mask_r] = -Z*alpha_el*(3*rc**2 - r_arr[mask_r]**2)/(2*rc**3)
    if np.any(~mask_r):
        V[~mask_r] = -Z*alpha_el/r_arr[~mask_r]
    if np.isscalar(r):
        V=V[0]
    return V

def electric_potential_V0_uniform(rc,Z,alpha_el=constants.alpha_el):
    return -Z*alpha_el*3/(2*rc)