from ... import constants
from ..base import nucleus_base

from ...utility.math import momentum, angle_shift_mod_pi, hyper1f1_scalar_arbitrary_precision, hyper1f1_vector_z_arbitrary_precision

import numpy as np
pi = np.pi

from scipy.special import factorial, gamma

from mpmath import gamma as mp_gamma, arg as mp_arg, fabs as mp_abs, exp as mp_exp, convert as convert_to_mp, sqrt as mp_sqrt
def angle_gamma_large(z):
    return float(mp_arg(mp_gamma(z)))

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
    El=np.sqrt(alpha_el/(4*pi))*Z/r_arr**2
    if np.isscalar(r):
        El=El[0]
    return El

def electric_potential_coulomb(r,Z,alpha_el=constants.alpha_el):
    r_arr = np.atleast_1d(r)
    pot=-Z*alpha_el/r_arr
    if np.isscalar(r):
        pot=pot[0]
    return pot

def form_factor_coulomb(r):
    r_arr = np.atleast_1d(r)
    FF = np.ones(len(r_arr))
    if np.isscalar(r):
        FF=FF[0]
    return FF

# this is the actual energy, E = E_bin + m
def energy_coulomb_nk(n,kappa,Z,mass,reg=+1,alpha_el=constants.alpha_el):
    sigma=reg*np.sqrt(kappa**2 - (alpha_el*Z)**2)
    return mass/np.sqrt(1+(alpha_el*Z/(n-np.abs(kappa)+sigma))**2)

def g_coulomb_nk(r,n,kappa,Z,mass,reg=+1,alpha_el=constants.alpha_el):
    # r in fm, mass in MeV, g in sqrt(MeV)
    r=r/constants.hc
    energy = energy_coulomb_nk(n,kappa,Z,mass=mass,reg=reg,alpha_el=alpha_el)
    lam = np.sqrt(1 - energy**2/mass**2)
    y0=alpha_el*Z
    rho=rho_kappa(kappa,Z)
    sigma=reg*rho
    n_p = n - np.abs(kappa) 
    #
    pref=+np.sqrt(lam)**3/np.sqrt(2)
    #
    if np.isscalar(r):
        hyper1f1=hyper1f1_scalar_arbitrary_precision
    else:
        hyper1f1=hyper1f1_vector_z_arbitrary_precision
    #
    return pref*(np.abs(np.sqrt(gamma(2*sigma+1+n_p)*(mass+energy)/(factorial(n_p)*y0*(y0-lam*kappa))+0j))/(gamma(2*sigma+1)))*((2*lam*mass*r)**sigma)*(np.exp(-lam*mass*r))*( -n_p*hyper1f1(-n_p+1,2*sigma+1,2*lam*mass*r)-(kappa-y0/lam)*hyper1f1(-n_p,2*sigma+1,2*lam*mass*r) )

def f_coulomb_nk(r,n,kappa,Z,mass,reg=+1,alpha_el=constants.alpha_el):
    # r in fm, mass in MeV, g in sqrt(MeV)
    r=r/constants.hc
    energy = energy_coulomb_nk(n,kappa,Z,mass=mass,reg=reg,alpha_el=alpha_el)
    lam = np.sqrt(1 - energy**2/mass**2)
    y0=alpha_el*Z
    rho=rho_kappa(kappa,Z)
    sigma=reg*rho
    n_p = n - np.abs(kappa) 
    #
    pref=-np.sqrt(lam)**3/np.sqrt(2)
    #
    #
    if np.isscalar(r):
        hyper1f1=hyper1f1_scalar_arbitrary_precision
    else:
        hyper1f1=hyper1f1_vector_z_arbitrary_precision
    #
    return pref*(np.sqrt(np.abs(gamma(2*sigma+1+n_p)*(mass-energy)/(factorial(n_p)*y0*(y0-lam*kappa))+0j))/(gamma(2*sigma+1)))*((2*lam*mass*r)**sigma)*(np.exp(-lam*mass*r))*( n_p*hyper1f1(-n_p+1,2*sigma+1,2*lam*mass*r)-(kappa-y0/lam)*hyper1f1(-n_p,2*sigma+1,2*lam*mass*r) )

def g_coulomb(r,kappa,Z,energy,mass,reg,pass_eta=None,pass_hyper1f1=None,dps_hyper1f1=15,alpha_el=constants.alpha_el):
    # r in fm

    rho=rho_kappa(kappa,Z)
    sigma=rho*reg
    
    if 0<gamma(2*sigma+1)<np.inf:
        k=momentum(energy,mass)
        y=alpha_el*Z*energy/k
        
        if pass_eta is None:
            pass_eta = eta_coulomb(kappa,Z,energy,mass,reg,alpha_el=alpha_el)
        if pass_hyper1f1 is None:
            pass_hyper1f1=hyper1f1_coulomb(r,kappa,Z,energy,mass,reg,dps=dps_hyper1f1,alpha_el=alpha_el)
        
        prefactor=(-1)*np.sign(kappa)*np.sqrt(2*(energy+mass)/k)
        
        r=r/constants.hc
        
        return prefactor*((2*k*r)**sigma)*(np.exp(pi*y/2))*(np.abs(gamma(sigma+1j*y))/(gamma(2*sigma+1)))*np.real((np.exp(-1j*k*r+1j*pass_eta))*(sigma+1j*y)*pass_hyper1f1)
    else:
        if np.isscalar(r):
            return mp_g_coulomb_scalar(r,kappa,Z,energy,mass,reg,pass_eta,pass_hyper1f1,dps_hyper1f1,alpha_el)
        else:
            return mp_g_coulomb(r,kappa,Z,energy,mass,reg,pass_eta,pass_hyper1f1,dps_hyper1f1,alpha_el)
    
def f_coulomb(r,kappa,Z,energy,mass,reg,pass_eta=None,pass_hyper1f1=None,dps_hyper1f1=15,alpha_el=constants.alpha_el):
    # r in fm

    rho=rho_kappa(kappa,Z)
    sigma=rho*reg
    
    if 0<gamma(2*sigma+1)<np.inf:
        k=momentum(energy,mass)
        y=alpha_el*Z*energy/k
        
        if pass_eta is None:
            pass_eta = eta_coulomb(kappa,Z,energy,mass,reg,alpha_el=alpha_el)
        if pass_hyper1f1 is None:
            pass_hyper1f1=hyper1f1_coulomb(r,kappa,Z,energy,mass,reg,dps=dps_hyper1f1,alpha_el=alpha_el)
        
        prefactor=np.sign(kappa)*np.sqrt(2*(energy-mass)/k)
        
        r=r/constants.hc

        return prefactor*((2*k*r)**sigma)*(np.exp(pi*y/2))*(np.abs(gamma(sigma+1j*y))/(gamma(2*sigma+1)))*np.imag((np.exp(-1j*k*r+1j*pass_eta))*(sigma+1j*y)*pass_hyper1f1)
    else:
        if np.isscalar(r):
            return mp_f_coulomb_scalar(r,kappa,Z,energy,mass,reg,pass_eta,pass_hyper1f1,dps_hyper1f1,alpha_el)
        else:
            return mp_f_coulomb(r,kappa,Z,energy,mass,reg,pass_eta,pass_hyper1f1,dps_hyper1f1,alpha_el)

def mp_g_coulomb_scalar(r,kappa,Z,energy,mass,reg,pass_eta=None,pass_hyper1f1=None,dps_hyper1f1=15,alpha_el=constants.alpha_el):
    # r in fm

    if pass_eta is None:
        pass_eta = eta_coulomb(kappa,Z,energy,mass,reg,alpha_el=alpha_el)
    if pass_hyper1f1 is None:
        pass_hyper1f1=hyper1f1_coulomb(r,kappa,Z,energy,mass,reg,dps=dps_hyper1f1,alpha_el=alpha_el)
    
    energy=convert_to_mp(energy)
    mass=convert_to_mp(mass)
    pass_eta = convert_to_mp(pass_eta)
    pass_hyper1f1 = convert_to_mp(pass_hyper1f1)
    
    k=momentum(energy,mass)
    y=alpha_el*Z*energy/k
    rho=rho_kappa(kappa,Z)
    sigma=rho*reg
    
    prefactor=(-1)*np.sign(kappa)*mp_sqrt(2*(energy+mass)/k)
    
    r=r/constants.hc
    
    return float(prefactor*((2*k*r)**sigma)*(mp_exp(pi*y/2))*(mp_abs(mp_gamma(sigma+1j*y))/(mp_gamma(2*sigma+1)))*np.real((mp_exp(-1j*k*r+1j*pass_eta))*(sigma+1j*y)*pass_hyper1f1))

mp_g_coulomb =  np.vectorize(mp_g_coulomb_scalar,excluded=[1,2,3,4,5,6,7,8,9])

def mp_f_coulomb_scalar(r,kappa,Z,energy,mass,reg,pass_eta=None,pass_hyper1f1=None,dps_hyper1f1=15,alpha_el=constants.alpha_el):
    # r in fm

    if pass_eta is None:
        pass_eta = eta_coulomb(kappa,Z,energy,mass,reg,alpha_el=alpha_el)
    if pass_hyper1f1 is None:
        pass_hyper1f1=hyper1f1_coulomb(r,kappa,Z,energy,mass,reg,dps=dps_hyper1f1,alpha_el=alpha_el)
    
    energy=convert_to_mp(energy)
    mass=convert_to_mp(mass)
    pass_eta = convert_to_mp(pass_eta)
    pass_hyper1f1 = convert_to_mp(pass_hyper1f1)
    
    k=momentum(energy,mass)
    y=alpha_el*Z*energy/k
    rho=rho_kappa(kappa,Z)
    sigma=rho*reg    
    
    prefactor=np.sign(kappa)*np.sqrt(2*(energy-mass)/k)
    
    r=r/constants.hc
    
    return float(prefactor*((2*k*r)**sigma)*(mp_exp(pi*y/2))*(mp_abs(mp_gamma(sigma+1j*y))/(mp_gamma(2*sigma+1)))*np.imag((mp_exp(-1j*k*r+1j*pass_eta))*(sigma+1j*y)*pass_hyper1f1))
    
mp_f_coulomb =  np.vectorize(mp_f_coulomb_scalar,excluded=[1,2,3,4,5,6,7,8,9])
    

def hyper1f1_coulomb(r,kappa,Z,energy,mass,reg=+1,dps=15,alpha_el=constants.alpha_el):
    # r in fm
    r=r/constants.hc
    k=momentum(energy,mass)
    y=alpha_el*Z*energy/k
    rho=rho_kappa(kappa,Z)
    #
    if np.isscalar(r):
        hyper1f1=hyper1f1_scalar_arbitrary_precision
    else:
        hyper1f1=hyper1f1_vector_z_arbitrary_precision
    #
    return hyper1f1(reg*rho+1+1j*y,2*reg*rho+1,2j*k*r,dps=dps)

def delta_coulomb(kappa,Z,energy,mass,reg,pass_eta=None,alpha_el=constants.alpha_el):
    k=momentum(energy,mass)
    y=alpha_el*Z*energy/k
    rho=rho_kappa(kappa,Z)
    sigma=reg*rho
    #
    if pass_eta is None:
        pass_eta = eta_coulomb(kappa,Z,energy,mass,reg,alpha_el=alpha_el)
    #
    z=sigma+1j*y
    gamma_z=gamma(z)
    if np.abs(gamma_z) < np.inf:
        angle_gamma_z=np.angle(gamma_z)
    else:
        angle_gamma_z = angle_gamma_large(z) #only used for when scipy overflows b/c a lot slower
        #print("warning: Gamma(z) overflows, angle replaced with approximation")
        #angle_gamma_z=angle_shift_mod_pi(y*np.log(sigma))
    #
    return delta_pis(kappa) - angle_gamma_z + pass_eta - pi*sigma/2

def delta_pis(kappa):
    if kappa>0:
        pis = (1/2)*(kappa+1)*pi
    if kappa<0:
        pis = (1/2)*(-kappa)*pi
    return pis

def delta_1overr(r,kappa,Z,energy,mass,alpha_el=constants.alpha_el):
    # r in fm
    r=r/constants.hc
    k=momentum(energy,mass)
    y=alpha_el*Z*energy/k
    delta_1or = y*np.log(2*k*r)-delta_pis(kappa)
    return delta_1or

def theta_coulomb(kappa,Z,energy,mass,pass_eta_regular=None,pass_eta_irregular=None,alpha_el=constants.alpha_el): # add pass eta ?
    kE=momentum(energy,mass)
    y=alpha_el*Z*energy/kE
    rho=rho_kappa(kappa,Z)
    theta = pi*(rho-np.abs(kappa)) - np.arctan(np.tan(pi*(np.abs(kappa)-rho))/np.tanh(pi*y))
    if mass!=0:
        if pass_eta_regular is None: 
            pass_eta_regular =  eta_coulomb(kappa,Z,energy,mass,reg=+1,alpha_el=alpha_el)
        if pass_eta_irregular is None: 
            pass_eta_irregular = eta_coulomb(kappa,Z,energy,mass,reg=+1,pass_eta_regular=pass_eta_regular,alpha_el=alpha_el)
        theta += pi/2. - np.angle(rho+1j*y) - ( pass_eta_regular - pass_eta_irregular)
    return theta

def eta_coulomb(kappa,Z,energy,mass,reg=+1,pass_eta_regular=None,pass_eta_irregular=None,alpha_el=constants.alpha_el):
    
    if reg==+1:
        if pass_eta_regular is None:
            return eta_coulomb_regular(kappa,Z,energy,mass,alpha_el=alpha_el)
        else:
            return pass_eta_regular
    elif reg==-1:
        if pass_eta_irregular is None:
            if mass==0 and not (pass_eta_regular is None):
                return - pi - (pass_eta_regular + np.sign(kappa)*(pi/2))
            else:
                return - pi - eta_coulomb_regular(-kappa,Z,energy,mass,alpha_el=alpha_el)
        else:
            return pass_eta_irregular
    else:
        raise ValueError("reg=+-1 only!")

def eta_coulomb_regular(kappa,Z,energy,mass,alpha_el=constants.alpha_el):
    k=momentum(energy,mass)
    y=alpha_el*Z*energy/k
    rho=rho_kappa(kappa,Z)
    return -((1+np.sign(kappa))/2)*pi/2 - (1./2.)*np.arctan2(y*(1 +(rho * mass)/(kappa*energy)),rho - ((y**2 * mass)/(kappa*energy)))

def rho_kappa(kappa,Z,alpha_el=constants.alpha_el):
    return np.sqrt(kappa**2-(alpha_el*Z)**2) if np.abs(kappa) > alpha_el*Z else np.sqrt(kappa**2-(alpha_el*Z)**2+0j)