from ..utility.math import momentum
from .. import constants

import numpy as np
pi = np.pi

from scipy.special import spherical_jn

def radial_dirac_eq(r,y,potential,energy,mass,kappa,contain=False):
    # only scalar r in fm, potential in fm^-1, mass & energy in MeV, 
    # dy/dr = A y -> [A]=[r]=fm
    #
    hc=constants.hc # MeV fm
    Ebar=energy-potential(r)*hc
    
    #raise ValueError('force break')
    #print(r,y,A)
    # if contain:
    #    if np.any(np.abs(y)>1e100):
    #        y*=1e-100
    #    if np.any(np.abs(y)<1e-100):
    #        y*=1e100
    
    return np.array([[-kappa/r,(Ebar+mass)/hc],[-(Ebar-mass)/hc,kappa/r]]) @ y

def initial_values(beginning_radius,electric_potential_V0,energy,mass,kappa,Z,nucleus_type=None,alpha_el=constants.alpha_el):
    
    hc=constants.hc # MeV fm
    
    Ebar=energy-electric_potential_V0*hc #MeV
    k0=momentum(Ebar,mass) #MeV
    z0=k0*beginning_radius/hc
    
    if not nucleus_type=="coulomb":
        if kappa>0:
            g_kappa=-(beginning_radius/hc)*np.sqrt((Ebar+mass)/(Ebar-mass))*spherical_jn(kappa,z0)
            f_kappa=-(beginning_radius/hc)*spherical_jn(kappa-1,z0)
        elif kappa<0:
            g_kappa=+(beginning_radius/hc)*spherical_jn(-kappa-1,z0)
            f_kappa=-(beginning_radius/hc)*np.sqrt((Ebar-mass)/(Ebar+mass))*spherical_jn(-kappa,z0)
        else:
            raise ValueError("kappa=0 not allowed")
    else:
        rho_kappa = np.sqrt(kappa**2 - (alpha_el*Z)**2)
        g_kappa=-1*(kappa-rho_kappa)/(alpha_el*Z)*(beginning_radius/hc)**rho_kappa
        f_kappa=-1*(beginning_radius/hc)**rho_kappa
        
    return np.array([g_kappa,f_kappa])

default_boundstate_settings={
    "beginning_radius":1e-12,
    "critical_radius_perZ":500, #finetune scale with n?
    "critical_radius":None,
    "asymptotic_radius_perZ":5000,  #finetune scale with n?
    "asymptotic_radius":None,
    "radius_precision":1e-3,
    "energy_precision":1e-12,
    "energy_subdivisions":100,
    "atol":1e-12,
    "rtol":1e-9,
    "method":'DOP853',
    "verbose":True, # TODO change
    "renew":True, # TODO change
    "save":False, # TODO change
    }

class solver_settings():
    def __init__(self,Z,beginning_radius,critical_radius_perZ,asymptotic_radius_perZ,radius_precision,energy_precision,energy_subdivisions,atol,rtol,method,renew,save,verbose,critical_radius,asymptotic_radius):
        
        self.beginning_radius = beginning_radius
        self.critical_radius = critical_radius
        if self.critical_radius is None:
            self.critical_radius = critical_radius_perZ/Z
        self.asymptotic_radius = asymptotic_radius
        if self.asymptotic_radius is None:
            self.asymptotic_radius = asymptotic_radius_perZ/Z
        self.radius_precision = radius_precision
        self.energy_precision = energy_precision
        self.energy_subdivisions = energy_subdivisions
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.verbose = verbose
        self.renew = renew
        self.save = save

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
