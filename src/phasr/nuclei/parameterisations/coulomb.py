from ... import constants
from ..base import nucleus_base

import numpy as np
pi = np.pi

from scipy.special import factorial, gamma
from mpmath import hyper, workdps #confluent hypergeometric function
def hyp1f1_scalar_arbitrary_precision(a,b,z,dps=15):
    with workdps(dps):
        return complex(hyper([a],[b],z))
hyp1f1=np.vectorize(hyp1f1_scalar_arbitrary_precision,excluded=[0,1,3])

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
        if hasattr(self,"k_barrett") and hasattr(self,"alpha_barrett"):
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

# this is the actual energy, E = E_bin + m
def energy_coulomb_nk(n,kappa,Z,mass,reg=+1,alpha_el=constants.alpha_el):
    rho=reg*np.sqrt(kappa**2 - (alpha_el*Z)**2)
    return mass/np.sqrt(1+(alpha_el*Z/(n-np.abs(kappa)+rho))**2)

def g_coulomb_nk(r,n,kappa,Z,mass,reg=+1,alpha_el=constants.alpha_el):
    # r in fm, mass in MeV, g in sqrt(MeV)
    r=r/constants.hc
    E = energy_coulomb_nk(n,kappa,Z,mass=mass,reg=reg,alpha_el=alpha_el)
    lam = np.sqrt(1 - E**2/mass**2)
    y=alpha_el*Z
    rho=np.sqrt(kappa**2 - y**2)
    sigma=reg*rho
    n_p = n - np.abs(kappa) 
    #
    pref=+np.sqrt(lam)**3/np.sqrt(2)
    #
    return pref*(np.abs(np.sqrt(gamma(2*sigma+1+n_p)*(mass+E)/(factorial(n_p)*y*(y-lam*kappa))+0j))/(gamma(2*sigma+1)))*((2*lam*mass*r)**sigma)*(np.exp(-lam*mass*r))*( -n_p*hyp1f1(-n_p+1,2*sigma+1,2*lam*mass*r)-(kappa-y/lam)*hyp1f1(-n_p,2*sigma+1,2*lam*mass*r) )

def f_coulomb_nk(r,n,kappa,Z,mass,reg=+1,alpha_el=constants.alpha_el):
    # r in fm, mass in MeV, g in sqrt(MeV)
    r=r/constants.hc
    E = energy_coulomb_nk(n,kappa,Z,mass=mass,reg=reg,alpha_el=alpha_el)
    lam = np.sqrt(1 - E**2/mass**2)
    y=alpha_el*Z
    rho=np.sqrt(kappa**2 - y**2)
    sigma=reg*rho
    n_p = n - np.abs(kappa) 
    #
    pref=-np.sqrt(lam)**3/np.sqrt(2)
    #
    return pref*(np.sqrt(np.abs(gamma(2*sigma+1+n_p)*(mass-E)/(factorial(n_p)*y*(y-lam*kappa))+0j))/(gamma(2*sigma+1)))*((2*lam*mass*r)**sigma)*(np.exp(-lam*mass*r))*( n_p*hyp1f1(-n_p+1,2*sigma+1,2*lam*mass*r)-(kappa-y/lam)*hyp1f1(-n_p,2*sigma+1,2*lam*mass*r) )