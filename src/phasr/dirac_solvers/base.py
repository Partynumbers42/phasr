from ..utility.math import momentum
from .. import constants

import numpy as np
pi = np.pi

from scipy.special import spherical_jn

from mpmath import besselj as mp_besselj, sqrt as mp_sqrt
def mp_spherical_jn(n,z):
    return mp_sqrt(pi/(2*z))*mp_besselj(n+0.5,z)

def radial_dirac_eq_fm(r_fm,y,potential,energy,mass,kappa,contain=False): #
    # only scalar r in fm, potential in fm^-1, mass & energy in MeV, 
    # dy/dr = A y -> [A]=[r]=fm
    #
    hc=constants.hc # MeV fm
    Ebar=energy-potential(r_fm)*hc # MeV
    
    #print(r,y,A)
    if contain:
        # use only if total norm irrelevant
        if np.any(np.abs(y)>1e100):
           y*=1e0/np.max(np.abs(y))
       
    return np.array([[-kappa/r_fm,(Ebar+mass)/hc],[-(Ebar-mass)/hc,kappa/r_fm]]) @ y

def radial_dirac_eq_norm(r_norm,y,potential,energy,mass,kappa,energy_norm,contain=False): 
    # change to units normalised to the boundstate energy of the coulomb solution
    hc=constants.hc
    return radial_dirac_eq_fm(r_norm*hc/energy_norm,y,potential,energy,mass,kappa,contain=contain)*hc/energy_norm

def initial_values_fm(beginning_radius_fm,electric_potential_V0,energy,mass,kappa,Z,nucleus_type=None,contain=False,alpha_el=constants.alpha_el): 
    
    initials = initial_values_fm_norm(beginning_radius_fm,electric_potential_V0,energy,mass,kappa,Z,nucleus_type=nucleus_type,contain=contain,alpha_el=alpha_el)
    initials = initials/np.abs(initials[0])
    
    if not nucleus_type=="coulomb":
        return beginning_radius_fm*initials
    else:
        rho_kappa = np.sqrt(kappa**2 - (alpha_el*Z)**2)
        return (beginning_radius_fm)**rho_kappa*initials
    
def initial_values_fm_norm(beginning_radius_fm,electric_potential_V0,energy,mass,kappa,Z,nucleus_type=None,contain=False,alpha_el=constants.alpha_el): 
    
    hc=constants.hc # MeV fm
    
    Ebar=energy-electric_potential_V0*hc #MeV
    k0=momentum(Ebar,mass) #MeV
    z0=k0*beginning_radius_fm/hc
    
    if not nucleus_type=="coulomb":
        
        mp_type=False
        jn_test = spherical_jn(np.abs(kappa),z0)
        if z0!=0 and np.abs(jn_test)>0:
            spherical_jn_fct=spherical_jn
        else:
            spherical_jn_fct=mp_spherical_jn
            mp_type=True
        
        if kappa>0:
            g_kappa=-np.sqrt((Ebar+mass)/(Ebar-mass))*spherical_jn_fct(kappa,z0)
            f_kappa=-spherical_jn_fct(kappa-1,z0)
        elif kappa<0:
            g_kappa=+spherical_jn_fct(-kappa-1,z0)
            f_kappa=-np.sqrt((Ebar-mass)/(Ebar+mass))*spherical_jn_fct(-kappa,z0)
        else:
            raise ValueError("kappa=0 not allowed")
        
    else:
        rho_kappa = np.sqrt(kappa**2 - (alpha_el*Z)**2)
        g_kappa=-1*(kappa-rho_kappa)/(alpha_el*Z)
        f_kappa=-1
    
    y0 = np.array([g_kappa,f_kappa])
    
    if contain or mp_type:
        min_lim=1e-200
        if np.any(np.abs(y0)<min_lim):
           y0*=min_lim/np.min(np.abs(y0))
       
    if mp_type:
        y0=np.array([float(y0[0]),float(y0[1])])
    
    return y0

def initial_values_norm(beginning_radius_norm,electric_potential_V0,energy,mass,kappa,Z,energy_norm,nucleus_type=None,contain=False,alpha_el=constants.alpha_el): 
    hc=constants.hc # MeV fm
    initials_fm = initial_values_fm(beginning_radius_norm*hc/energy_norm,electric_potential_V0,energy,mass,kappa,Z,nucleus_type=nucleus_type,contain=contain,alpha_el=alpha_el)
    if not nucleus_type=="coulomb":
        return initials_fm*(energy_norm/hc)
    else:
        rho_kappa = np.sqrt(kappa**2 - (alpha_el*Z)**2)
        return initials_fm*(energy_norm/hc)**rho_kappa

default_boundstate_settings={
    "beginning_radius_norm":1e-6, # in inverse coulomb binding energies 
    "beginning_radius":None,
    "critical_radius_norm":0.3, # in inverse coulomb binding energies
    "critical_radius":None,
    "asymptotic_radius_norm":1, # in inverse coulomb binding energies
    "asymptotic_radius":None,
    "radius_optimise_step_norm":1e-2, # in inverse coulomb binding energies
    "radius_optimise_step":None,
    "energy_precision_norm":1e-6, # in coulomb binding energies
    "energy_precision":None,
    "energy_subdivisions":100,
    "potential_precision":None,
    "atol":1e-12,
    "rtol":1e-9,
    "method":'DOP853',
    "dps_hyper1f1":None, # unused
    "verbose":False, 
    "renew":False, 
    "save":True, 
    }

default_continuumstate_settings={
    "beginning_radius_norm":None, 
    "beginning_radius":None, # set by potential vs V0
    "critical_radius_norm":None,  
    "critical_radius":None, # set by potential vs coulomb
    "asymptotic_radius_norm":None, 
    "asymptotic_radius":20, # fm
    "radius_optimise_step_norm":None, 
    "radius_optimise_step":1e-1,
    "energy_precision_norm":None, # unused
    "energy_precision":None, # unused
    "energy_subdivisions":None, # unused
    "potential_precision":1e-6,
    "atol":1e-12,
    "rtol":1e-9,
    "method":'DOP853',
    "dps_hyper1f1":15,
    "verbose":False,
    "renew":None, # unused
    "save":None, # unused
}

class solver_settings():
    def __init__(self,energy_norm,
                 beginning_radius,critical_radius,asymptotic_radius,radius_optimise_step,energy_precision,
                 beginning_radius_norm,critical_radius_norm,asymptotic_radius_norm,radius_optimise_step_norm,energy_precision_norm,
                 energy_subdivisions,potential_precision,atol,rtol,method,dps_hyper1f1,renew,save,verbose):
        self.energy_norm=energy_norm
        self.beginning_radius = beginning_radius
        self.beginning_radius_norm = beginning_radius_norm
        self.set_norm_or_unnorm("beginning_radius","beginning_radius_norm",constants.hc/self.energy_norm)
        self.critical_radius = critical_radius
        self.critical_radius_norm = critical_radius_norm
        self.set_norm_or_unnorm("critical_radius","critical_radius_norm",constants.hc/self.energy_norm)
        self.asymptotic_radius = asymptotic_radius
        self.asymptotic_radius_norm = asymptotic_radius_norm
        self.set_norm_or_unnorm("asymptotic_radius","asymptotic_radius_norm",constants.hc/self.energy_norm)
        self.radius_optimise_step = radius_optimise_step
        self.radius_optimise_step_norm = radius_optimise_step_norm
        self.set_norm_or_unnorm("radius_optimise_step","radius_optimise_step_norm",constants.hc/self.energy_norm)
        self.energy_precision = energy_precision
        self.energy_precision_norm = energy_precision_norm
        self.set_norm_or_unnorm("energy_precision","energy_precision_norm",self.energy_norm)
        self.energy_subdivisions = energy_subdivisions
        self.potential_precision = potential_precision
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.dps_hyper1f1 = dps_hyper1f1
        self.verbose = verbose
        self.renew = renew
        self.save = save

    def set_norm_or_unnorm(self,radius_str,radius_norm_str,norm):
        if not (self.energy_norm is None):
            if not (getattr(self,radius_str) is None):
                setattr(self,radius_norm_str,getattr(self,radius_str)/norm)
            elif not (getattr(self,radius_norm_str) is None):
                setattr(self,radius_str,getattr(self,radius_norm_str)*norm)
    
    def as_dict(self):
        relevant_keys = [
            "beginning_radius_norm", 
            "beginning_radius", 
            "critical_radius_norm",  
            "critical_radius", 
            "asymptotic_radius_norm", 
            "asymptotic_radius", 
            "radius_optimise_step_norm", 
            "radius_optimise_step",
            "energy_precision_norm",
            "energy_precision",
            "energy_subdivisions",
            "potential_precision",
            "atol",
            "rtol",
            "method",
            "dps_hyper1f1"]
        settings_dict={}
        for key in relevant_keys:
            if hasattr(self,key):
                setting_val = getattr(self,key)
                if setting_val is not None:
                    settings_dict[key]=setting_val
        return settings_dict

