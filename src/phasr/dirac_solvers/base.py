from ..utility.math import momentum
from .. import constants

import numpy as np
pi = np.pi

from ..nuclei.parameterisations.coulomb import energy_coulomb_nk

from scipy.special import spherical_jn

def radial_dirac_eq_fm(r_fm,y,potential,energy,mass,kappa,contain=False): #
    # only scalar r in fm, potential in fm^-1, mass & energy in MeV, 
    # dy/dr = A y -> [A]=[r]=fm
    #
    hc=constants.hc # MeV fm
    Ebar=energy-potential(r_fm)*hc # MeV
    
    #print(r,y,A)
    if contain:
       if np.any(np.abs(y)>1e100):
           y*=1e-100
           #print("downscaled at r=",r_fm,"fm")
       if np.any(np.abs(y)<1e-100):
           y*=1e100
           #print("upscaled at r=",r_fm,"fm")
    
    return np.array([[-kappa/r_fm,(Ebar+mass)/hc],[-(Ebar-mass)/hc,kappa/r_fm]]) @ y

def radial_dirac_eq_norm(r_norm,y,potential,energy,mass,kappa,energy_norm,contain=False): 
    # change to units normalised to the boundstate energy of the coulomb solution
    hc=constants.hc
    return radial_dirac_eq_fm(r_norm*hc/energy_norm,y,potential,energy,mass,kappa,contain=contain)*hc/energy_norm

def initial_values_fm(beginning_radius_fm,electric_potential_V0,energy,mass,kappa,Z,nucleus_type=None,alpha_el=constants.alpha_el): 
    
    hc=constants.hc # MeV fm
    
    Ebar=energy-electric_potential_V0*hc #MeV
    k0=momentum(Ebar,mass) #MeV
    z0=k0*beginning_radius_fm/hc
    
    if not nucleus_type=="coulomb":
        if kappa>0:
            g_kappa=-(beginning_radius_fm)*np.sqrt((Ebar+mass)/(Ebar-mass))*spherical_jn(kappa,z0)
            f_kappa=-(beginning_radius_fm)*spherical_jn(kappa-1,z0)
        elif kappa<0:
            g_kappa=+(beginning_radius_fm)*spherical_jn(-kappa-1,z0)
            f_kappa=-(beginning_radius_fm)*np.sqrt((Ebar-mass)/(Ebar+mass))*spherical_jn(-kappa,z0)
        else:
            raise ValueError("kappa=0 not allowed")
    else:
        rho_kappa = np.sqrt(kappa**2 - (alpha_el*Z)**2)
        g_kappa=-1*(kappa-rho_kappa)/(alpha_el*Z)*(beginning_radius_fm)**rho_kappa
        f_kappa=-1*(beginning_radius_fm)**rho_kappa
        
    return np.array([g_kappa,f_kappa])

def initial_values_norm(beginning_radius_norm,electric_potential_V0,energy,mass,kappa,Z,energy_norm,nucleus_type=None,alpha_el=constants.alpha_el): 
    hc=constants.hc # MeV fm
    initials_fm = initial_values_fm(beginning_radius_norm*hc/energy_norm,electric_potential_V0,energy,mass,kappa,Z,nucleus_type=nucleus_type,alpha_el=alpha_el)
    if not nucleus_type=="coulomb":
        return initials_fm*(energy_norm/hc)
    else:
        rho_kappa = np.sqrt(kappa**2 - (alpha_el*Z)**2)
        return initials_fm*(energy_norm/hc)**rho_kappa

default_boundstate_settings={
    "beginning_radius_norm":1e-6, # in inverse coulomb binding energies 
    "beginning_radius":None,
    "critical_radius_norm":0.3, # in inverse coulomb binding energies # could also be set by behaviour of coulomb solution
    "critical_radius":None,
    "asymptotic_radius_norm":1, # in inverse coulomb binding energies # could also be set by behaviour of coulomb solution
    "asymptotic_radius":None,
    "radius_optimise_step_norm":1e-2, # in inverse coulomb binding energies
    "radius_optimise_step":None,
    "energy_precision_norm":1e-6, # in coulomb binding energies
    "energy_precision":None,
    "energy_subdivisions":100,
    "atol":1e-12,
    "rtol":1e-9,
    "method":'DOP853',
    "verbose":True, 
    "renew":False, 
    "save":True, 
    }

default_continuumstate_settings={
    "beginning_radius_norm":1e-6, # in inverse coulomb binding energies 
    "beginning_radius":None,
    "critical_radius_norm":None,  
    "critical_radius":10, # set to None once set by potential
    "asymptotic_radius_norm":None, 
    "asymptotic_radius":20, # fm
    "radius_optimise_step_norm":1e-2, # in inverse coulomb binding energies
    "radius_optimise_step":None,
    "energy_precision_norm":1e-6, # in coulomb binding energies
    "energy_precision":None,
    "energy_subdivisions":None,
    "atol":1e-12,
    "rtol":1e-9,
    "method":'DOP853',
    "verbose":True, # TODO change
    "renew":True, # TODO change
    "save":False, # TODO change
}

class solver_settings():
    def __init__(self,energy_norm,
                 beginning_radius,critical_radius,asymptotic_radius,radius_optimise_step,energy_precision,
                 beginning_radius_norm,critical_radius_norm,asymptotic_radius_norm,radius_optimise_step_norm,energy_precision_norm,
                 energy_subdivisions,atol,rtol,method,renew,save,verbose):
        self.energy_norm=energy_norm
        self.beginning_radius = beginning_radius
        self.beginning_radius_norm = beginning_radius_norm
        self.set_radius("beginning_radius","beginning_radius_norm",constants.hc/self.energy_norm)
        self.critical_radius = critical_radius
        self.critical_radius_norm = critical_radius_norm
        self.set_radius("critical_radius","critical_radius_norm",constants.hc/self.energy_norm)
        self.asymptotic_radius = asymptotic_radius
        self.asymptotic_radius_norm = asymptotic_radius_norm
        self.set_radius("asymptotic_radius","asymptotic_radius_norm",constants.hc/self.energy_norm)
        self.radius_optimise_step = radius_optimise_step
        self.radius_optimise_step_norm = radius_optimise_step_norm
        self.set_radius("radius_optimise_step","radius_optimise_step_norm",constants.hc/self.energy_norm)
        self.energy_precision = energy_precision
        self.energy_precision_norm = energy_precision_norm
        self.set_radius("energy_precision","energy_precision_norm",self.energy_norm)
        self.energy_subdivisions = energy_subdivisions
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.verbose = verbose
        self.renew = renew
        self.save = save

    # TODO rename
    def set_radius(self,radius_str,radius_norm_str,norm):
        if not self.energy_norm is None:
            if not (getattr(self,radius_str) is None):
                setattr(self,radius_norm_str,getattr(self,radius_str)/norm)
            else:
                setattr(self,radius_str,getattr(self,radius_norm_str)*norm)
        
    
# adjusted by eg.  base.boundstate_settings.renew = True 

# def radial_dirac_eq_prep(atom,E,mi,kappa=-1,EisEbin=True):
#     if EisEbin:
#         E+=mi
#     Ak=np.array([[-kappa,0],[0,kappa]])
#     AE=np.array([[0,E+mi],[-(E-mi),0]])
#     AV=np.array([[0,-1],[+1,0]])
#     #
#     potential=atom.electric_potential
#     #
#     pot=lambda r: potential(r*hc/(alpha_el*mmu))*hc/(alpha_el*mmu)
#     #
#     return Ak, AE, AV, pot

# def radial_dirac_eq_eval(r,y,Ak,AE,AV,pot,vectorized=False):
#     if vectorized: # vectorized is slower and not vectorized by the solver
#         scalar=False
#         if len(np.shape(r))==0:
#             scalar=True
#             r=np.array([r])
#         A=np.einsum('ij,k->ijk',Ak,1/r)+np.repeat(AE[:,:,np.newaxis],len(r),axis=2)+np.einsum('ij,k->ijk',AE,pot(r))
#         yp=np.einsum('ijk,jk->ik',A,y)
#         if scalar:
#             yp=yp[:,0]
#         return yp
#     else:
#         A=Ak/r + AE + AV*pot(r)
#         return A @ y
