from ..utility.math import momentum
from .. import constants

import numpy as np
pi = np.pi

from scipy.special import spherical_jn

def radial_dirac_eq(r,fct,potential,energy,mass,kappa):
    # only scalar r in fm, potential in fm^-1, mass & energy in MeV, 
    E=energy
    m=mass
    V=potential
    hc=constants.hc
    return np.array([[-kappa*hc/r,E-V(r)*hc+m],[-(E-V(r)-m),kappa*hc/r]]) @ fct

def initial_values(beginning_radius,electric_potential_V0,energy,mass,kappa,Z,nucleus_type=None,alpha_el=constants.alpha_el):
    
    r0=beginning_radius
    V0=electric_potential_V0*constants.hc
    E=energy
    m=mass
    k0=momentum(E-V0,m)
    
    if not nucleus_type=="coulomb":
        if kappa>0:
            g_kappa=-np.sqrt((E-V0+m)/(E-V0-m))*r0*spherical_jn(kappa,k0*r0)
            f_kappa=-r0*spherical_jn(kappa-1,k0*r0)
        elif kappa<0:
            g_kappa=r0*spherical_jn(-kappa-1,k0*r0)
            f_kappa=-np.sqrt((E-V0-m)/(E-V0+m))*r0*spherical_jn(-kappa,k0*r0)
        else:
            raise ValueError("kappa=0 not allowed")
    else:
        rho_kappa = np.sqrt(kappa**2 - (alpha_el*Z)**2)
        g_kappa=-1*(kappa-rho_kappa)/(alpha_el*Z)*r0**rho_kappa
        f_kappa=-1*r0**rho_kappa
        
    return np.array([g_kappa,f_kappa])

default_beginning_radius=1e-12
default_critical_radius=2.5
default_asymptotic_radius=5
default_radius_precision=1e-3
default_energy_precision=1e-12
default_energy_subdivisions=100
default_atol=1e-6
default_rtol=1e-3
default_method='DOP853'
default_verbose=True # TODO change
default_renew=True # TODO change
default_save=False # TODO change

class solver_settings():
    def __init__(self,beginning_radius,critical_radius,asymptotic_radius,radius_precision,energy_precision,energy_subdivisions,atol,rtol,method,renew,save,verbose):
        
        self.beginning_radius = beginning_radius
        self.critical_radius = critical_radius
        self.asymptotic_radius = asymptotic_radius
        self.radius_precision = radius_precision
        self.energy_precision = energy_precision
        self.energy_subdivisions = energy_subdivisions
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.verbose = verbose
        self.renew = renew
        self.save = save

boundstate_settings = solver_settings(default_beginning_radius,default_critical_radius,default_asymptotic_radius,default_radius_precision,default_energy_precision,default_energy_subdivisions,default_atol,default_rtol,default_method,default_renew,default_save,default_verbose)

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
