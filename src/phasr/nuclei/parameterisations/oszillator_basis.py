from ... import constants
from ..base import nucleus_base

import numpy as np
pi = np.pi

from scipy.special import spherical_jn, gamma, hyp1f1

class nucleus_osz(nucleus_base):
    def __init__(self,name,Z,A,Ci_dict,**args): #,R_cut=None,rho_cut=None
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "oszillator-basis"
        self.multipoles = list(Ci_dict.keys()) 
        for multipole in Ci_dict:
            setattr(self,'Ci_'+multipole,Ci_dict[multipole])
        if "b_osz" in args:
            self.b_osz = args["b_osz"]
        else:
            self.b_osz = b_osz_shell_model(self.A)
        #
        self.update_dependencies()

    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        for multipole in self.multipoles:
            def struc(q,multipole=multipole): return structure_function_osz(q,getattr(self,'Ci_'+multipole),self.b_osz)
            setattr(self,'F'+multipole,struc)    
        nucleus_base.update_dependencies(self)
        for multipole in self.multipoles:
            #
            if multipole in [S+str(L)+nuc for L in np.arange(0,2*self.spin+1,2,dtype=int) for S in ['M','Phipp'] for nuc in ['0','1']]:
                #
                def rho(q,multipole=multipole): return density_osz(q,getattr(self,'Ci_'+multipole),self.b_osz)
                setattr(self,'rho'+multipole,rho)
                def rho2(q,multipole=multipole): return density_osz(q,getattr(self,'Ci_'+multipole),self.b_osz,q_order=2)
                setattr(self,'rho2'+multipole,rho2)
                #
                def El(q,multipole=multipole): return field_osz(q,getattr(self,'Ci_'+multipole),self.b_osz)
                setattr(self,'El'+multipole,El)
                def El2(q,multipole=multipole): return field_osz(q,getattr(self,'Ci_'+multipole),self.b_osz,q_order=2)
                setattr(self,'El2'+multipole,El2)
                #
                def V(q,multipole=multipole): return potential_osz(q,getattr(self,'Ci_'+multipole),self.b_osz)
                setattr(self,'V'+multipole,V)
                def V2(q,multipole=multipole): return potential_osz(q,getattr(self,'Ci_'+multipole),self.b_osz,q_order=2)
                setattr(self,'V2'+multipole,V2) 
            #
        nucleus_base.update_dependencies(self)
        #
        # add radius, total charge, etc.
        # maybe remove some attributes (e.g. isospin basis) to declutter
        
def IsoToNP(Pl,Mi,PN):
    #PN=+-1
    return (Pl + PN*Mi)/2

def b_osz_shell_model(A): #oszillation length
    return 197.327/np.sqrt(938.919*(45.*A**(-1./3.)-25.*A**(-2./3.)))

def structure_function_osz(q,Ci,b):
    #
    q=q/constants.hc
    #
    q_arr = np.atleast_1d(q)
    #
    N_i=len(Ci)
    u=(q_arr**2)*(b**2)/2
    N_u=len(u)
    #
    k=np.arange(N_i)
    k_grid=np.tile(k,(N_u,1)).transpose()
    u_grid=np.tile(u,(N_i,1))
    upk=np.power(u_grid,k_grid)
    Fstructure = np.einsum('i,ij->j',Ci,upk)*np.exp(-u/2)
    #
    if np.isscalar(q):
        Fstructure = Fstructure[0]
    return Fstructure

def density_osz(r,Ci,b,q_order=0):
    #
    r_arr = np.atleast_1d(r)
    #
    N_i=len(Ci)
    z=r_arr**2/b**2
    N_z=len(z)
    #
    k=np.arange(N_i)
    k_grid=np.tile(k,(N_z,1)).transpose()
    z_grid=np.tile(z,(N_i,1))
    hyp1f1_grid= 2**k_grid*gamma(3./2.+q_order/2.+k_grid)*hyp1f1(3./2.+q_order/2.+k_grid,3./2.,-z_grid)
    density = 2**(2+q_order)*np.einsum('i,ij->j',Ci,hyp1f1_grid)/b**(3+q_order)
    #
    if np.isscalar(r):
        density=density[0]
    #
    return density/(2*pi**2)

def field_osz(r,Ci,b,q_order=0,alpha_el=constants.alpha_el):
    #
    r_arr = np.atleast_1d(r)
    #
    N_i=len(Ci)
    z=r_arr**2/b**2
    N_z=len(z)
    #
    k=np.arange(N_i)
    k_grid=np.tile(k,(N_z,1)).transpose()
    z_grid=np.tile(z,(N_i,1))
    hyp1f1_grid= 2**k_grid*gamma(3./2.+q_order/2.+k_grid)*hyp1f1(3./2.+q_order/2.+k_grid,5./2.,-z_grid)
    field = np.sqrt(4*pi*alpha_el)*(r/3.)*2**(2+q_order)*np.einsum('i,ij->j',Ci,hyp1f1_grid)/b**(3+q_order)
    #
    if np.isscalar(r):
        field=field[0]
    #
    return field/(2*pi**2)

def potential_osz(r,Ci,b,q_order=0,alpha_el=constants.alpha_el):
    #
    r_arr = np.atleast_1d(r)
    #
    N_i=len(Ci)
    z=r_arr**2/b**2
    N_z=len(z)
    #
    k=np.arange(N_i)
    k_grid=np.tile(k,(N_z,1)).transpose()
    z_grid=np.tile(z,(N_i,1))
    hyp1f1_grid= 2**k_grid*gamma(1./2.+q_order/2.+k_grid)*hyp1f1(1./2.+q_order/2.+k_grid,3./2.,-z_grid)
    potential = -4*pi*alpha_el*2**q_order*np.einsum('i,ij->j',Ci,hyp1f1_grid)/b**(1+q_order)
    #
    if np.isscalar(r):
        potential=potential[0]
    return potential/(2*pi**2)

def potential0_osz(Ci,b,q_order=0,alpha_el=constants.alpha_el):
    N_i=len(Ci)
    k=np.arange(N_i)
    potential0 = -4*pi*alpha_el*2**q_order*np.sum(Ci*2**k*gamma(1./2.+q_order/2.+k))/b**(1+q_order)
    #
    return potential0/(2*pi**2)

def r_sq_osz(Ci,b,q_order=0):
    if q_order==0:
        rsq=3.*(Ci[0]-2*Ci[1])*pi**2*b**2
    elif q_order==1:
        raise ValueError("q_order=1 does not converge")
    elif q_order==2:
        rsq=-12*Ci[0]*pi**2
    elif q_order>=2:
        rsq=0
    else:
        raise ValueError("invalid value for q_order")
    return rsq/(2*pi**2)

def total_charge_osz(Ci,q_order=0):
    if q_order==0:
        Q=Ci[0]
    elif q_order>=1:
        Q=0
    else:
        raise ValueError("invalid value for q_order")
    return Q