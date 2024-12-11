from ... import constants
from ..base import nucleus_base

import numpy as np
pi = np.pi

from scipy.special import spherical_jn

class nucleus_FB(nucleus_base):
    def __init__(self,name,Z,A,ai,R,**args): #,R_cut=None,rho_cut=None
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "Fourier-Bessel"
        self.ai=ai
        self.R=R
        self.N=len(ai)
        #
        self.qi=np.arange(1,self.N+1)*pi/self.R
        self.total_charge = total_charge_FB(self.ai,self.qi,self.N)
        self.charge_radius_sq = charge_radius_sq_FB(self.ai,self.Z,self.qi,self.N)
        self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        #self.charge_density = charge_density_FB_gen(self.ai,self.R,self.qi)
        #self.form_factor = formfactor_FB_gen(self.ai,self.R,self.Z,self.qi,self.N)
        self.electric_potential = electric_potential_FB_gen(self.ai,self.R,self.Z,self.qi,alpha_el=constants.alpha_el)


# FB
def total_charge_FB(ai,qi,N):
    nu=np.arange(1,N+1)
    Qi = -(-1)**nu*nu*pi*ai/qi**3
    return 4*pi*np.sum(Qi)
# #
# def total_charge_FB_uncertainty(ai,qi,N,cov): # indep. of ai # no include_R
#     nu=np.arange(1,N+1)
#     dQi_dai = -(-1)**nu*nu*pi/qi**3
#     J=dQi_dai
#     dQi = np.einsum('i,ik,k->',J,cov,J)
#     return 4*pi*dQi
#
def charge_radius_FB(ai,Z,qi,N):
    nu=np.arange(1,N+1)
    Qi = (-1)**nu*nu*pi*(6-(nu*pi)**2)*ai/qi**5
    return np.sqrt(4*pi*np.sum(Qi)/Z) # +0j np.real?
#
def charge_radius_sq_FB(ai,Z,qi,N):
    nu=np.arange(1,N+1)
    Qi = (-1)**nu*nu*pi*(6-(nu*pi)**2)*ai/qi**5
    return 4*pi*np.sum(Qi)/Z # +0j np.real?
#
# def charge_radius_sq_FB_deriv(ai,Z,qi,N,R,cov):
#     nu=np.arange(1,N+1)
#     dQi_dai = (-1)**nu*nu*pi*(6-(nu*pi)**2)/qi**5
#     drsq_dai = 4*pi*dQi_dai/Z
#     return drsq_dai
# #
# def charge_radius_FB_uncertainty(ai,Z,qi,N,R,cov,include_R=False,return_deriv=False):
#     r=charge_radius_FB(ai,Z,qi,N)
#     nu=np.arange(1,N+1)
#     dQi_dai = (-1)**nu*nu*pi*(6-(nu*pi)**2)/qi**5
#     if include_R:
#         dQi_dR = (-1)**nu*nu*pi*(6-(nu*pi)**2)*ai*(5/R)/qi**5
#     drsq_dai = 4*pi*dQi_dai/Z
#     if include_R:
#         drsq_dR = 4*pi*np.sum(dQi_dR)/Z
#     if include_R:
#         Jsq=np.insert(drsq_dai,0,drsq_dR) #--------------------
#     else:
#         Jsq=drsq_dai
#     J=Jsq/(2*r)
#     dr=np.sqrt(np.einsum('i,ik,k->',J,cov,J))
#     return dr
#
# def Bi0(qi,R,k,alpha):
#     return (1./qi)*np.imag(complex(gammainc(k+2,0,R*(alpha-1j*qi))/(alpha-1j*qi)**(k+2)))
# Bi=np.vectorize(Bi0,excluded=[1,2,3])
# #
# def Barrett_moment_FB(ai,Z,qi,R,k,alpha):
#     return 4*pi*np.sum(ai*Bi(qi,R,k,alpha))/Z
# #
# def Barrett_moment_FB_uncertainty(ai,Z,qi,R,k,alpha,cov): # no include_R
#     dB_dai=4*pi*Bi(qi,R,k,alpha)/Z    
#     J=dB_dai 
#     dB = np.sqrt(np.einsum('i,ik,k->',J,cov,J)) 
#     return dB
#
def charge_density_FB(r,ai,R,qi):
    scalar=False
    if len(np.shape(r))==0:
        scalar=True
        r=np.array([r])
    rho=np.array([0])
    if np.any(r<=R):
        qr=np.einsum('i,j->ij',qi,r)
        rho = np.einsum('i,ij->j',ai,spherical_jn(0,qr))
    if np.any(r>R):
        if np.size(rho)>1:
            rho[np.where(r>R)]=np.array([0])
        else:
            rho=np.array([0])
    if scalar:
        rho=rho[0]
    return rho
#
def charge_density_FB_gen(ai,R,qi):
    def charge_density(r):
        return charge_density_FB(r,ai,R,qi)
    return charge_density
# #
# def dcharge_density_dr_FB(r,ai,R,qi):
#     scalar=False
#     if len(np.shape(r))==0:
#         scalar=True
#         r=np.array([r])
#     drho_dr=np.array([0])
#     if np.any(r<=R):
#         qr=np.einsum('i,j->ij',qi,r)
#         drho_dr = np.einsum('i,ij->j',-ai*qi,spherical_jn(1,qr))
#     if np.any(r>R):
#         if np.size(drho_dr)>1:
#             drho_dr[np.where(r>R)]=np.array([0])
#         else:
#             drho_dr=np.array([0])
#     if scalar:
#         drho_dr=drho_dr[0]
#     return drho_dr
# #
# def dcharge_density_dr_FB_gen(ai,R,qi):
#     def dcharge_density_dr(r):
#         return dcharge_density_dr_FB(r,ai,R,qi)
#     return dcharge_density_dr
# #
# def charge_density_FB_uncertainty(r,ai,R,qi,cov,include_R=False,R_cut=None):
#     if R_cut is None:
#         R_cut=R
#     scalar=False
#     if len(np.shape(r))==0:
#         scalar=True
#         r=np.array([r])
#     drho=np.array([0])
#     if np.any(r<=R_cut):
#         qr=np.einsum('i,j->ij',qi,r)
#         drho_dai = spherical_jn(0,qr)
#         if not include_R:
#             J=drho_dai
#         else:
#             drho_dR = np.einsum('i,ij->j',ai,spherical_jn(1,qr)*qr/R)
#             J=np.concatenate((drho_dR[np.newaxis,:],drho_dai),axis=0)
#         drho=np.sqrt(np.einsum('ij,ik,kj->j',J,cov,J))
#     if np.any(r>R_cut):
#         if np.size(drho)>1:
#             drho[np.where(r>R_cut)]=np.array([0])
#         else:
#             drho=np.array([0])
#     if scalar:
#         drho=drho[0]
    
#     # add alternative cov_rho return
        
#     return drho#, J
# #
# def charge_density_FB_uncertainty_gen(ai,R,qi,cov):
#     def charge_density_uncertainty(r):
#         return charge_density_FB_uncertainty(r,ai,R,qi,cov)
#     return charge_density_uncertainty
# #
# def electric_field_FB(r,ai,R,Z,qi,alpha_el=alpha_el):
#     scalar=False
#     if len(np.shape(r))==0:
#         scalar=True
#         r=np.array([r])
#     E0=np.array([0])
#     if np.any(r<=R):
#         qr=np.einsum('i,j->ij',qi,r)
#         E0 = np.einsum('i,ij->j',ai/qi,spherical_jn(1,qr)) #-spherical_jn(0,qr,derivative=True) # faster with (np.sin(qr) - qr*np.cos(qr))/qr**2 but more precise like this
#     if np.any(r>R):
#         if np.size(E0)>1:
#             E0[np.where(r>R)]=(1/(4*pi))*Z/r[np.where(r>R)]**2
#         else:
#             E0=(1/(4*pi))*Z/r**2
#     if scalar:
#         E0=E0[0]
#     return np.sqrt(4*pi*alpha_el)*E0
# #
# def electric_field_FB_gen(ai,R,Z,qi,alpha_el=alpha_el):
#     def E_field(r):
#         return electric_field_FB(r,ai,R,Z,qi,alpha_el=alpha_el)
#     return E_field
# #
# def electric_field_FB_uncertainty(r,ai,R,Z,qi,cov,alpha_el=alpha_el):
#     scalar=False
#     if len(np.shape(r))==0:
#         scalar=True
#         r=np.array([r])
#     dE0=np.array([0])
#     if np.any(r<=R):
#         qr=np.einsum('i,j->ij',qi,r)
#         J = np.einsum('i,ij->j',1/qi,spherical_jn(1,qr)) #-spherical_jn(0,qr,derivative=True) # faster with (np.sin(qr) - qr*np.cos(qr))/qr**2 but more precise like this    
#         dE0=np.sqrt(np.einsum('ij,ik,kj->j',J,cov,J))
#     if np.any(r>R):
#         if np.size(dE0)>1:
#             dE0[np.where(r>R)]=np.array([0])
#         else:
#             dE0=np.array([0])
#     if scalar:
#         dE0=dE0[0]
#     return np.sqrt(4*pi*alpha_el)*dE0
# #
# def delectric_field_dai_FB(r,R,Z,qi,alpha_el=alpha_el):
#     scalar=False
#     if len(np.shape(r))==0:
#         scalar=True
#         r=np.array([r])
#     dE0_dai=np.array([0])
#     if np.any(r<=R):
#         qr=np.einsum('i,j->ij',qi,r)
#         dE0_dai = np.einsum('i,ij->j',1/qi,spherical_jn(1,qr))
#     if np.any(r>R):
#         if np.size(dE0_dai)>1:
#             dE0_dai[np.where(r>R)]=np.array([0])
#         else:
#             dE0_dai=np.array([0])
#     if scalar:
#         dE0_dai=dE0_dai[0]
#     return np.sqrt(4*pi*alpha_el)*dE0_dai
# #
def electric_potential_FB(r,ai,R,Z,qi,alpha_el=constants.alpha_el):
    scalar=False
    if len(np.shape(r))==0:
        scalar=True
        r=np.array([r])
    V=np.array([0])
    if np.any(r<=R):
        qr = np.einsum('i,j->ij',qi,r)
        V0 = -alpha_el*Z/R
        V = V0 - 4*pi*alpha_el*np.einsum('i,ij->j',ai/qi**2,spherical_jn(0,qr))
    if np.any(r>R):
        if np.size(V)>1:
            V[np.where(r>R)]=-alpha_el*Z/r[np.where(r>R)]
        else:
            V=-alpha_el*Z/r
    if scalar:
        V=V[0]
    return V
#
def electric_potential_FB_gen(ai,R,Z,qi,alpha_el=constants.alpha_el):
    def V_pot(r):
        return electric_potential_FB(r,ai,R,Z,qi,alpha_el=alpha_el)
    return V_pot
# #
# def electric_potential_FB_V0(ai,R,Z,qi,alpha_el=alpha_el):
#     V0 = -alpha_el*Z/R - 4*pi*alpha_el*np.sum(ai/qi**2)
#     return V0
# #
# def formfactor_FB_uncertainty(q,ai,R,Z,qi,N,cov,return_cov=False): # include_R not included. R uncertainty not propagated
#     scalar=False
#     if len(np.shape(q))==0:
#         scalar=True
#         q=np.array([q])
#     #
#     dF=np.array([0])
#     nu=np.arange(1,N+1)
#     #
#     N_z=len(q)
#     q_grid=np.tile(q,(N,1))
#     qi_grid=np.tile(qi,(N_z,1)).transpose()
#     denom=q_grid**2-qi_grid**2
#     #
#     dF_dai = 4*pi*R*spherical_jn(0,q*R)*np.einsum('i,ij->ij',(-1)**nu,1./denom)
#     J=dF_dai
#     if not return_cov:
#         dF=np.sqrt(np.einsum('ij,ik,kj->j',J,cov,J))
#         #
#         if scalar:
#             dF=dF[0]
#         return dF/Z
#     else:
#         cov_F=np.einsum('ij,ik,kl->jl',J,cov,J)
#         if scalar:
#             dF=dF[0,0]
#         return cov_F/Z**2
# #
# def formfactor_FB_uncertainty_gen(ai,R,Z,qi,N,cov):
#     def formfactor_uncertainty(q):
#         return formfactor_FB_uncertainty(q,ai,R,Z,qi,N,cov)
#     return formfactor_uncertainty
# #
def formfactor_FB(q,ai,R,Z,qi,N):
    scalar=False
    if len(np.shape(q))==0:
        scalar=True
        q=np.array([q])
    F=np.array([0])
    nu=np.arange(1,N+1)
    N_z=len(q)
    q_grid=np.tile(q,(N,1))
    qi_grid=np.tile(qi,(N_z,1)).transpose()
    denom=q_grid**2-qi_grid**2
    F= 4*pi*R*spherical_jn(0,q*R)*np.einsum('i,ij->j',ai*(-1)**nu,1./denom)
    if scalar:
        F=F[0]
    return F/Z
#
def formfactor_FB_gen(ai,R,Z,qi,N):
    def Fch(q):
        # unit conversion happening here
        return formfactor_FB(q/constants.hc,ai,R,Z,qi,N)
    return Fch