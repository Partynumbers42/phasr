from ... import constants
from ..base import nucleus_base

import numpy as np
pi = np.pi

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
            self.b_osz = b_osz(self.A)
        self.update_dependencies()

    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        for multipole in self.multipoles:
            def struc(q,multipole=multipole): return structure_function_osz(q,getattr(self,'Ci_'+multipole),self.b_osz)
            setattr(self,'F'+multipole,struc)
        nucleus_base.update_dependencies(self)

def IsoToNP(Pl,Mi,PN):
    #PN=+-1
    return (Pl + PN*Mi)/2

def b_osz(A): #oszillation length
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

#rework TODO
def rho_shell(r,Cs,b,order=0):
    scalar=False
    if len(np.shape(r))==0:
        scalar=True
        r=np.array([r])
    #
    N=len(Cs)
    z=r**2/b**2
    N_z=len(z)
    #
    k=np.arange(N)
    k_grid=np.tile(k,(N_z,1)).transpose()
    z_grid=np.tile(z,(N,1))
    hyp1f1_grid= 2**k_grid*gamma_fct(3./2.+order+k_grid)*hyp1f1(3./2.+order+k_grid,3./2.,-z_grid)
    S=np.einsum('i,ij->j',Cs,hyp1f1_grid)
    out = 2**(2+2*order)*S/b**(3+2*order)
    #
    if scalar:
        out=out[0]
    return out/(2*pi**2)

#
def E_shell(r,Cs,b,order=0):
    scalar=False
    if len(np.shape(r))==0:
        scalar=True
        r=np.array([r])
    #
    N=len(Cs)
    z=r**2/b**2
    N_z=len(z)
    #
    k=np.arange(N)
    k_grid=np.tile(k,(N_z,1)).transpose()
    z_grid=np.tile(z,(N,1))
    hyp1f1_grid= 2**k_grid*gamma_fct(3./2.+order+k_grid)*hyp1f1(3./2.+order+k_grid,5./2.,-z_grid)
    S=np.einsum('i,ij->j',Cs,hyp1f1_grid)
    out = np.sqrt(4*pi*alpha_el)*(r/3.)*2**(2+2*order)*S/b**(3+2*order)
    #
    if scalar:
        out=out[0]
    return out
#
def V_shell(r,Cs,b,order=0):
    scalar=False
    if len(np.shape(r))==0:
        scalar=True
        r=np.array([r])
    #
    N=len(Cs)
    z=r**2/b**2
    N_z=len(z)
    #
    k=np.arange(N)
    k_grid=np.tile(k,(N_z,1)).transpose()
    z_grid=np.tile(z,(N,1))
    hyp1f1_grid= 2**k_grid*gamma_fct(1./2.+order+k_grid)*hyp1f1(1./2.+order+k_grid,3./2.,-z_grid)
    S=np.einsum('i,ij->j',Cs,hyp1f1_grid)
    out = -4*pi*alpha_el*2**(2*order)*S/b**(1+2*order)
    #
    if scalar:
        out=out[0]
    return out
#
def V0_shell(_,Cs,b,order=0):
    N=len(Cs)
    k=np.arange(N)
    S=np.sum(Cs*2**k*gamma_fct(1./2.+order+k))
    out = -4*pi*alpha_el*2**(2*order)*S/b**(1+2*order)
    #
    return out
#
def r_sq_shell(_,Cs,b,order=0):
    if order==0:
        S=3.*(Cs[0]-2*Cs[1])*pi**2*b**2
    if order==1:
        S=-12*Cs[0]*pi**2
    if order>1:
        S=0
    return S

