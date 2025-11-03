from ... import constants
from ..base import nucleus_base

import numpy as np
pi = np.pi

from functools import partial

from scipy.special import spherical_jn
from mpmath import gammainc

class nucleus_FB(nucleus_base):
    def __init__(self,name,Z,A,ai,R,**args): 
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "fourier-bessel"
        self.ai=ai
        self.R=R
        #
        if ('ai_proton' in args) and ('R_proton' in args) :
            self.ai_proton=args['ai_proton']
            self.R_proton=args['R_proton']
        
        if ('ai_neutron' in args) and ('R_neutron' in args) :
            self.ai_neutron=args['ai_neutron']
            self.R_neutron=args['R_neutron']
        
        if ('ai_weak' in args) and ('R_weak' in args) :
            self.ai_weak=args['ai_weak']
            self.R_weak=args['R_weak']
        #
        self.update_dependencies()

    def update_dependencies(self):
        
        self.N_a=len(self.ai)
        self.qi=np.arange(1,self.N_a+1)*pi/self.R
        #
        #self.rrange[1]=self.R 
        #self.qrange[1]=self.qi[-1]*constants.hc 
        #
        self.total_charge = total_charge_FB(self.ai,self.qi,self.N_a)
        self.total_charge_jacobian = total_charge_FB_jacob(self.qi,self.N_a)
        #
        if np.abs(self.total_charge - self.Z)/self.Z>1e-3:
            print('Warning total charge for '+self.name+' deviates more than 1e-3: Z='+str(self.Z)+', Q='+str(self.total_charge))
        #
        self.charge_radius_sq = charge_radius_sq_FB(self.ai,self.qi,self.total_charge,self.N_a)
        self.charge_radius_sq_jacobian = charge_radius_sq_FB_jacob(self.qi,self.total_charge,self.N_a)
        self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        self.charge_radius_jacobian = self.charge_radius_sq_jacobian/(2*self.charge_radius)
        #
        self.Vmin = electric_potential_V0_FB(self.ai,self.R,self.qi,self.total_charge,alpha_el=constants.alpha_el)
        self.Vmin_jacobian = electric_potential_V0_FB_jacob(self.qi,alpha_el=constants.alpha_el)
        #
        #if hasattr(self,'k_barrett') and hasattr(self,'alpha_barrett'):
        #    self.barrett_moment = Barrett_moment_FB(self.ai,self.R,self.qi,self.total_charge,self.k_barrett,self.alpha_barrett)
        #    self.barrett_moment_jacobian = Barrett_moment_FB_jacob(self.R,self.qi,self.total_charge,self.k_barrett,self.alpha_barrett)
        #
        if hasattr(self,'ai_proton') and hasattr(self,'R_proton'):
            self.N_a_proton=len(self.ai_proton)
            self.qi_proton=np.arange(1,self.N_a_proton+1)*pi/self.R_proton
            #def rho_p(r): return charge_density_FB(r,self.ai_proton,self.R_proton,self.qi_proton)
            self.proton_density = partial(charge_density_FB,ai=self.ai_proton,R=self.R_proton,qi=self.qi_proton)
        #
        if hasattr(self,'ai_neutron') and hasattr(self,'R_neutron'):
            self.N_a_neutron=len(self.ai_neutron)
            self.qi_neutron=np.arange(1,self.N_a_neutron+1)*pi/self.R_neutron
            #def rho_n(r): return charge_density_FB(r,self.ai_neutron,self.R_neutron,self.qi_neutron)
            self.neutron_density = partial(charge_density_FB,ai=self.ai_neutron,R=self.R_neutron,qi=self.qi_neutron)
        #
        if hasattr(self,'ai_weak') and hasattr(self,'R_weak'):
            self.N_a_weak=len(self.ai_weak)
            self.qi_weak=np.arange(1,self.N_a_weak+1)*pi/self.R_weak
            #def rho_w(r): return charge_density_FB(r,self.ai_weak,self.R_weak,self.qi_weak)
            self.weak_density = partial(charge_density_FB,ai=self.ai_weak,R=self.R_weak,qi=self.qi_weak)
        
        nucleus_base.update_dependencies(self)

    def update_R(self,R):
        self.R=R
        self.update_dependencies()
    
    def update_ai(self,ai):
        self.ai=ai
        self.update_dependencies()
    
    def update_R_and_ai(self,R,ai):
        self.R=R
        self.ai=ai
        self.update_dependencies()
    
    def barrett_moment(self,k_barrett:float,alpha_barrett:float):
        return Barrett_moment_FB(self.ai,self.R,self.qi,self.total_charge,k_barrett,alpha_barrett)
    
    def barrett_moment_jacobian(self,k_barrett:float,alpha_barrett:float):
        return Barrett_moment_FB_jacob(self.R,self.qi,self.total_charge,k_barrett,alpha_barrett)    
    
    def charge_density(self,r):
        return charge_density_FB(r,self.ai,self.R,self.qi)
    
    def charge_density_jacobian(self,r):
        return charge_density_FB_jacob(r,self.R,self.qi,self.N_a)
    
    def dcharge_density_dr(self,r):
        return dcharge_density_dr_FB(r,self.ai,self.R,self.qi)

    def electric_field(self,r):
        return electric_field_FB(r,self.ai,self.R,self.qi,self.total_charge,alpha_el=constants.alpha_el)
    
    def electric_field_jacobian(self,r):
        return electric_field_FB_jacob(r,self.R,self.qi,self.N_a,alpha_el=constants.alpha_el)
    
    def electric_potential(self,r):
        return electric_potential_FB(r,self.ai,self.R,self.qi,self.total_charge,alpha_el=constants.alpha_el)
    
    def electric_potential_jacobian(self,r):
        return electric_potential_FB_jacob(r,self.R,self.qi,self.N_a,alpha_el=constants.alpha_el)
    
    def form_factor(self,q):
        return form_factor_FB(q,self.ai,self.R,self.qi,self.total_charge,self.N_a)
    
    def form_factor_jacobian(self,q):
        return form_factor_FB_jacob(q,self.R,self.qi,self.total_charge,self.N_a)

def total_charge_FB(ai,qi,N):
    nu=np.arange(1,N+1)
    Qi = -(-1)**nu*nu*pi*ai/qi**3
    return 4*pi*np.sum(Qi)

def total_charge_FB_jacob(qi,N):
    nu=np.arange(1,N+1)
    dQi_dai = -(-1)**nu*nu*pi/qi**3
    dQ_dai=4*pi*dQi_dai
    return dQ_dai

def charge_radius_sq_FB(ai,qi,Z,N):
    nu=np.arange(1,N+1)
    Qi = (-1)**nu*nu*pi*(6-(nu*pi)**2)*ai/qi**5
    return 4*pi*np.sum(Qi)/Z

def charge_radius_sq_FB_jacob(qi,Z,N):
    nu=np.arange(1,N+1)
    dQi_dai = (-1)**nu*nu*pi*(6-(nu*pi)**2)/qi**5
    drsq_dai = 4*pi*dQi_dai/Z
    return drsq_dai

def Bi0(R,qi,k,alpha):
    return (1./qi)*np.imag(complex(gammainc(k+2,0,R*(alpha-1j*qi))/(alpha-1j*qi)**(k+2)))
Bi=np.vectorize(Bi0,excluded=[0,2,3])

def Barrett_moment_FB(ai,R,qi,Z,k_barrett,alpha_barrett):
    return 4*pi*np.sum(ai*Bi(R,qi,k_barrett,alpha_barrett))/Z

def Barrett_moment_FB_jacob(R,qi,Z,k_barrett,alpha_barrett):
    dB_dai=4*pi*Bi(R,qi,k_barrett,alpha_barrett)/Z
    return dB_dai

def electric_potential_V0_FB(ai,R,qi,Z,alpha_el=constants.alpha_el):
    V0 = -alpha_el*Z/R - 4*pi*alpha_el*np.sum(ai/qi**2)
    return V0

def electric_potential_V0_FB_jacob(qi,alpha_el=constants.alpha_el):
    V0 = -4*pi*alpha_el/qi**2
    return V0

def charge_density_FB(r,ai,R,qi):
    r_arr = np.atleast_1d(r)
    rho=np.zeros(len(r_arr))
    mask_r = r_arr<=R
    if np.any(mask_r):
        qr=np.einsum('i,j->ij',qi,r_arr[mask_r])
        rho[mask_r] = np.einsum('i,ij->j',ai,spherical_jn(0,qr))
    if np.isscalar(r):
        rho=rho[0]
    return rho

def charge_density_FB_jacob(r,R,qi,N):
    r_arr = np.atleast_1d(r)
    drho_dai=np.zeros((N,len(r_arr)))
    mask_r = r_arr<=R
    if np.any(mask_r):
        qr=np.einsum('i,j->ij',qi,r_arr[mask_r])
        drho_dai[:,mask_r] = spherical_jn(0,qr)
    if np.isscalar(r):
        drho_dai=drho_dai[:,0]
    return drho_dai

def electric_field_FB(r,ai,R,qi,Z,alpha_el=constants.alpha_el):
    r_arr = np.atleast_1d(r)
    El = np.zeros(len(r_arr))
    mask_r = r_arr<=R
    if np.any(mask_r):
        qr=np.einsum('i,j->ij',qi,r_arr[mask_r])
        El[mask_r] = np.einsum('i,ij->j',ai/qi,spherical_jn(1,qr)) 
    if np.any(~mask_r):
        El[~mask_r]=(1/(4*pi))*Z/r_arr[~mask_r]**2
    if np.isscalar(r):
        El = El[0]
    return np.sqrt(4*pi*alpha_el)*El

def electric_field_FB_jacob(r,R,qi,N,alpha_el=constants.alpha_el):
    r_arr = np.atleast_1d(r)
    dEl_dai = np.zeros((N,len(r_arr)))
    mask_r = r_arr<=R
    if np.any(mask_r):
        qr=np.einsum('i,j->ij',qi,r_arr[mask_r])
        dEl_dai[:,mask_r] = np.einsum('i,ij->ij',1/qi,spherical_jn(1,qr)) 
    if np.isscalar(r):
        dEl_dai = dEl_dai[:,0]
    return np.sqrt(4*pi*alpha_el)*dEl_dai

def electric_potential_FB(r,ai,R,qi,Z,alpha_el=constants.alpha_el):
    r_arr = np.atleast_1d(r)
    V = np.zeros(len(r_arr))
    mask_r = r_arr<=R
    if np.any(mask_r):
        qr=np.einsum('i,j->ij',qi,r_arr[mask_r])
        V0 = -alpha_el*Z/R
        V[mask_r] =  V0 - 4*pi*alpha_el*np.einsum('i,ij->j',ai/qi**2,spherical_jn(0,qr))
    if np.any(~mask_r):
        V[~mask_r]=-alpha_el*Z/r_arr[~mask_r]
    if np.isscalar(r):
        V = V[0]
    return V

def electric_potential_FB_jacob(r,R,qi,N,alpha_el=constants.alpha_el):
    r_arr = np.atleast_1d(r)
    dV_dai = np.zeros((N,len(r_arr)))
    mask_r = r_arr<=R
    if np.any(mask_r):
        qr=np.einsum('i,j->ij',qi,r_arr[mask_r])
        dV_dai[:,mask_r] = -4*pi*alpha_el*np.einsum('i,ij->ij',1/qi**2,spherical_jn(0,qr))
    if np.isscalar(r):
        dV_dai = dV_dai[:,0]
    return dV_dai

def form_factor_FB(q,ai,R,qi,Z,N): 
    q=q/constants.hc
    q_arr = np.atleast_1d(q)
    N_q=len(q_arr)
    nu=np.arange(1,N+1)
    q_grid=np.tile(q_arr,(N,1))
    qi_grid=np.tile(qi,(N_q,1)).transpose()
    denom=q_grid**2-qi_grid**2
    F= 4*pi*R*spherical_jn(0,q_arr*R)*np.einsum('i,ij->j',ai*(-1)**nu,1./denom)
    # overwrite singularities numerically
    for q_c in qi:
        eps = 1e-6
        mask_q = np.abs(q_arr-q_c)/q_c < eps
        if np.any(mask_q):
            F[mask_q]= (Z*form_factor_FB((q_arr[mask_q]-4*eps*q_c)*constants.hc,ai,R,qi,Z,N) + Z*form_factor_FB((q_arr[mask_q]+4*eps*q_c)*constants.hc,ai,R,qi,Z,N))/2
    if np.isscalar(q):
        F = F[0]
    return F/Z

def form_factor_FB_jacob(q,R,qi,Z,N):
    q=q/constants.hc
    q_arr = np.atleast_1d(q)
    N_q=len(q_arr)
    nu=np.arange(1,N+1)
    q_grid=np.tile(q_arr,(N,1))
    qi_grid=np.tile(qi,(N_q,1)).transpose()
    denom=q_grid**2-qi_grid**2
    dF_dai = 4*pi*R*spherical_jn(0,q_arr*R)*np.einsum('i,ij->ij',(-1)**nu,1./denom)
    if np.isscalar(q):
        dF_dai = dF_dai[:,0]
    return dF_dai/Z

def dcharge_density_dr_FB(r,ai,R,qi):
    r_arr = np.atleast_1d(r)
    drho_dr=np.zeros(len(r_arr))
    mask_r = r_arr<=R
    if np.any(mask_r):
        qr=np.einsum('i,j->ij',qi,r_arr[mask_r])
        drho_dr[mask_r] = np.einsum('i,ij->j',-ai*qi,spherical_jn(1,qr))
    if np.isscalar(r):
        drho_dr=drho_dr[0]
    return drho_dr
