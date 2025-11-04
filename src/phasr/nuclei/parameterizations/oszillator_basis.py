from ... import constants, masses
from ..base import nucleus_base
from ...utility.basis_change import Isospin_basis_to_nucleon_basis, Nucleon_basis_to_isospin_basis
from ...utility.math import hyper1f1_vector_arbitrary_precision as hyp1f1

import numpy as np
pi = np.pi

from scipy.special import gamma

class nucleus_osz(nucleus_base):
    def __init__(self,name,Z,A,Ci_dict,**args): #,R_cut=None,rho_cut=None
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "oszillator-basis"
        self.multipoles = list(Ci_dict.keys())
        for multipole in Ci_dict:
            setattr(self,'Ci_'+multipole,Ci_dict[multipole])
        #
        self.update_Ci_basis() # nuc <-> iso
        #
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
            #
            if multipole in [S+str(L)+nuc for L in np.arange(0,2*self.spin+1,2,dtype=int) for S in ['M','Phipp'] for nuc in ['p','n']]:
                L=int(multipole[-2])
                def rho(r,multipole=multipole,L=L): return density_osz(r,getattr(self,'Ci_'+multipole),self.b_osz,L=L)
                setattr(self,'rho'+multipole,rho)
                #
            if multipole in [S+'0'+nuc for S in ['M','Phipp'] for nuc in ['p','n']]:
                # only for L=0
                #
                def El(r,multipole=multipole): return field_osz(r,getattr(self,'Ci_'+multipole),self.b_osz)
                setattr(self,'El'+multipole,El)
                #
                def V(r,multipole=multipole): return potential_osz(r,getattr(self,'Ci_'+multipole),self.b_osz)
                setattr(self,'V'+multipole,V)
            #
        # rewrite directly as fcts?
        if hasattr(self,'rhoM0p') and hasattr(self,'rhoM0n') and hasattr(self,'rhoPhipp0p') and hasattr(self,'rhoPhipp0n'):
            rhoMLp=getattr(self,'rhoM0p')
            rhoMLn=getattr(self,'rhoM0n')
            def rho2MLp(r): return density_osz(r,getattr(self,'Ci_M0p'),self.b_osz,L=0,q_order=2)
            setattr(self, "rho2MLp", rho2MLp)
            def rho2MLn(r): return density_osz(r,getattr(self,'Ci_M0n'),self.b_osz,L=0,q_order=2)
            setattr(self, "rho2MLn", rho2MLn)
            def rho2PhippLp(r): return density_osz(r,getattr(self,'Ci_Phipp0p'),self.b_osz,L=0,q_order=2)
            setattr(self, "rho2PhippLp", rho2PhippLp)
            def rho2PhippLn(r): return density_osz(r,getattr(self,'Ci_Phipp0n'),self.b_osz,L=0,q_order=2)
            setattr(self, "rho2PhippLn", rho2PhippLn)
            def rhoch(r): return rhoch_composition_osz(r,rhoMLp,rho2MLp,rho2MLn,rho2PhippLp,rho2PhippLn)
            setattr(self,'charge_density',rhoch)
            def rhow(r): return rhow_composition_osz(r,rhoMLp,rhoMLn,rho2MLp,rho2MLn,rho2PhippLp,rho2PhippLn)
            setattr(self,'weak_density',rhow)
        
        if hasattr(self,'ElM0p') and hasattr(self,'ElM0n') and hasattr(self,'ElPhipp0p') and hasattr(self,'ElPhipp0n'):
            ElMLp=getattr(self,'ElM0p')
            def El2MLp(r): return field_osz(r,getattr(self,'Ci_M0p'),self.b_osz,q_order=2)
            def El2MLn(r): return field_osz(r,getattr(self,'Ci_M0n'),self.b_osz,q_order=2)
            def El2PhippLp(r): return field_osz(r,getattr(self,'Ci_Phipp0p'),self.b_osz,q_order=2)
            def El2PhippLn(r): return field_osz(r,getattr(self,'Ci_Phipp0n'),self.b_osz,q_order=2)
            def Elch(r): return rhoch_composition_osz(r,ElMLp,El2MLp,El2MLn,El2PhippLp,El2PhippLn)
            setattr(self,'electric_field',Elch)
        
        if hasattr(self,'VM0p') and hasattr(self,'VM0n') and hasattr(self,'VPhipp0p') and hasattr(self,'VPhipp0n'):
            VMLp=getattr(self,'VM0p')
            def V2MLp(r): return potential_osz(r,getattr(self,'Ci_M0p'),self.b_osz,q_order=2)
            def V2MLn(r): return potential_osz(r,getattr(self,'Ci_M0n'),self.b_osz,q_order=2)
            def V2PhippLp(r): return potential_osz(r,getattr(self,'Ci_Phipp0p'),self.b_osz,q_order=2)
            def V2PhippLn(r): return potential_osz(r,getattr(self,'Ci_Phipp0n'),self.b_osz,q_order=2)
            def Vch(r): return rhoch_composition_osz(r,VMLp,V2MLp,V2MLn,V2PhippLp,V2PhippLn)
            setattr(self,'electric_potential',Vch)
            setattr(self,'Vmin',Vch(0)) # <--- maybe include in a cleaner easier/quicker to evaluate way?
        
        if hasattr(self,'rhoM0p'):
            self.proton_radius_sq=r_sq_osz(getattr(self,'Ci_M0p'),self.b_osz)
            self.proton_radius = np.sqrt(self.proton_radius_sq) if self.proton_radius_sq>=0 else np.sqrt(self.proton_radius_sq+0j)
        if hasattr(self,'rhoM0n'):
            self.neutron_radius_sq=r_sq_osz(getattr(self,'Ci_M0n'),self.b_osz)
            self.neutron_radius = np.sqrt(self.neutron_radius_sq) if self.neutron_radius_sq>=0 else np.sqrt(self.neutron_radius_sq+0j)
        if hasattr(self,'charge_density'):
            self.total_charge=total_charge_osz(getattr(self,'Ci_M0p'))
            self.set_charge_radius_sq_osz()
            self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        if hasattr(self,'weak_density'):
            self.weak_charge=constants.Qw_p*total_charge_osz(getattr(self,'Ci_M0p')) + constants.Qw_n*total_charge_osz(getattr(self,'Ci_M0n'))
            self.set_weak_radius_sq_osz()
            self.weak_radius = np.sqrt(self.weak_radius_sq) if self.weak_radius_sq>=0 else np.sqrt(self.weak_radius_sq+0j)
        
        nucleus_base.update_dependencies(self)

    def set_charge_radius_sq_osz(self,rsqp=constants.rsq_p,rqsn=constants.rsq_n,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
        # only valid for L=0
        mN/=constants.hc 
        QM_p=total_charge_osz(getattr(self,'Ci_M0p'))
        QM_n=total_charge_osz(getattr(self,'Ci_M0n'))
        QPhipp_p=total_charge_osz(getattr(self,'Ci_Phipp0p'))
        QPhipp_n=total_charge_osz(getattr(self,'Ci_Phipp0n'))
        rsqM_p=r_sq_osz(getattr(self,'Ci_M0p'),self.b_osz)
        rsq2M_p=r_sq_osz(getattr(self,'Ci_M0p'),self.b_osz,q_order=2)
        rsq2M_n=r_sq_osz(getattr(self,'Ci_M0n'),self.b_osz,q_order=2)
        rsq2Phipp_p=r_sq_osz(getattr(self,'Ci_Phipp0p'),self.b_osz,q_order=2)
        rsq2Phipp_n=r_sq_osz(getattr(self,'Ci_Phipp0n'),self.b_osz,q_order=2)
        self.charge_radius_sq = 1/self.total_charge * \
        ( QM_p*rsqM_p - ((rsqp/6)+(1./(8*mN**2)))*QM_p*rsq2M_p \
        - (rqsn/6)*QM_n*rsq2M_n \
        + ((1+2*kp)/(4*mN**2))*QPhipp_p*rsq2Phipp_p \
        + ((2*kn)/(4*mN**2))*QPhipp_n*rsq2Phipp_n )
    
    def set_weak_radius_sq_osz(self,Qw_p=constants.Qw_p,Qw_n=constants.Qw_n,rsqp=constants.rsq_p,rsqn=constants.rsq_n,rsqsN=constants.rsq_sN,kp=constants.kappa_p,kn=constants.kappa_n,ksN=constants.kappa_sN,mN=masses.mN):
        # only valid for L=0
        mN/=constants.hc 
        QM_p=total_charge_osz(getattr(self,'Ci_M0p'))
        QM_n=total_charge_osz(getattr(self,'Ci_M0n'))
        QPhipp_p=total_charge_osz(getattr(self,'Ci_Phipp0p'))
        QPhipp_n=total_charge_osz(getattr(self,'Ci_Phipp0n'))
        rsqM_p=r_sq_osz(getattr(self,'Ci_M0p'),self.b_osz)
        rsqM_n=r_sq_osz(getattr(self,'Ci_M0n'),self.b_osz)
        rsq2M_p=r_sq_osz(getattr(self,'Ci_M0p'),self.b_osz,q_order=2)
        rsq2M_n=r_sq_osz(getattr(self,'Ci_M0n'),self.b_osz,q_order=2)
        rsq2Phipp_p=r_sq_osz(getattr(self,'Ci_Phipp0p'),self.b_osz,q_order=2)
        rsq2Phipp_n=r_sq_osz(getattr(self,'Ci_Phipp0n'),self.b_osz,q_order=2)
        self.weak_radius_sq = 1/self.weak_charge * \
        ( Qw_p*QM_p*rsqM_p - (Qw_p*((rsqp/6)+(1./(8*mN**2))) + Qw_n*((rsqn/6)+(rsqsN/6)))*QM_p*rsq2M_p \
        + Qw_n*QM_n*rsqM_n - (Qw_n*((rsqp/6)+(rsqsN/6)+(1./(8*mN**2))) + Qw_p*(rsqn/6))*QM_n*rsq2M_n \
        + ((Qw_p*(1+2*kp)+Qw_n*(2*kn+2*ksN))/(4*mN**2))*QPhipp_p*rsq2Phipp_p \
        + ((Qw_n*(1+2*kp+2*ksN)+Qw_p*(2*kn))/(4*mN**2))*QPhipp_n*rsq2Phipp_n )
    
    def update_Ci_basis(self):
        for multipole in np.unique([key[:-1] for key in self.multipoles]):
            if (hasattr(self,'Ci_'+multipole+'0') and hasattr(self,'Ci_'+multipole+'1')):
                for nuc in ['p','n']:
                    if not hasattr(self,'Ci_'+multipole+nuc):
                        Ci0 = getattr(self,'Ci_'+multipole+'0')
                        Ci1 = getattr(self,'Ci_'+multipole+'1')
                        Cinuc = Isospin_basis_to_nucleon_basis(Ci0,Ci1,nuc)
                        setattr(self,'Ci_'+multipole+nuc,Cinuc)
                        self.multipoles = list(np.unique(self.multipoles+[multipole+nuc]))
            if (hasattr(self,'Ci_'+multipole+'p') and hasattr(self,'Ci_'+multipole+'n')):
                for iso in ['0','1']:
                    if not hasattr(self,'Ci_'+multipole+iso):
                        Cip = getattr(self,'Ci_'+multipole+'p')
                        Cin = getattr(self,'Ci_'+multipole+'n')
                        Ciiso = Nucleon_basis_to_isospin_basis(Cip,Cin,iso)
                        setattr(self,'Ci_'+multipole+iso,Ciiso)
                        self.multipoles = list(np.unique(self.multipoles+[multipole+iso]))

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

def density_osz(r,Ci,b,L=0,q_order=0):
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
    #hyp1f1_grid= 2**k_grid*gamma(3./2.+q_order/2.+k_grid)*hyp1f1(3./2.+q_order/2.+k_grid,3./2.,-z_grid)
    #density = 2**(2+q_order)*np.einsum('i,ij->j',Ci,hyp1f1_grid)/b**(3+q_order)
    hyp1f1_grid= 2**k_grid*gamma(3./2.+q_order/2.+k_grid+L/2)*hyp1f1(3./2.+q_order/2.+k_grid+L/2,3./2.+L,-z_grid)
    density = 2**(2+q_order)*(r**L)*((np.sqrt(pi)/2)/gamma(3./2.+L))*np.einsum('i,ij->j',Ci,hyp1f1_grid)/b**(3+q_order+L)
    #
    if np.all(np.isreal(density)):
        density=np.real(density)
    #
    if np.isscalar(r):
        density=density[0]
    #
    return density/(2*pi**2)

def field_osz(r,Ci,b,q_order=0,alpha_el=constants.alpha_el):
    # only valid for L=0
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
    if np.all(np.isreal(field)):
        field=np.real(field)
    #
    if np.isscalar(r):
        field=field[0]
    #
    return field/(2*pi**2)

def potential_osz(r,Ci,b,q_order=0,alpha_el=constants.alpha_el):
    # only valid for L=0
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
    if np.all(np.isreal(potential)):
        potential=np.real(potential)
    #
    if np.isscalar(r):
        potential=potential[0]
    #
    return potential/(2*pi**2)

def potential0_osz(Ci,b,q_order=0,alpha_el=constants.alpha_el):
    # only valid for L=0
    N_i=len(Ci)
    k=np.arange(N_i)
    potential0 = -4*pi*alpha_el*2**q_order*np.sum(Ci*2**k*gamma(1./2.+q_order/2.+k))/b**(1+q_order)
    #
    return potential0/(2*pi**2)

def r_sq_osz(Ci,b,q_order=0):
    # only valid for L=0
    if q_order==0:
        rsq=3./2.*(Ci[0]-2*Ci[1])*b**2
    elif q_order==1:
        raise ValueError("q_order=1 does not converge")
    elif q_order==2:
        rsq=-6*Ci[0]
    elif q_order>=2:
        rsq=0
    else:
        raise ValueError("invalid value for q_order")
    return rsq/total_charge_osz(Ci)

def total_charge_osz(Ci,q_order=0):
    # only valid for L=0
    if q_order==0:
        Q=Ci[0]
    elif q_order>=1:
        Q=0
    else:
        raise ValueError("invalid value for q_order")
    return Q

def rhoch_composition_osz(r,rhoM_p,rho2M_p,rho2M_n,rho2Phipp_p,rho2Phipp_n,rsqp=constants.rsq_p,rqsn=constants.rsq_n,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
    # rho are "F.T." (using the correct L in j_L) of F(q), rho2 are F.T. of q^2 F(q), ...
    mN/=constants.hc
    return 1 * \
    ( rhoM_p(r) - ((rsqp/6)+(1./(8*mN**2)))*rho2M_p(r) \
     - (rqsn/6)*rho2M_n(r) \
     + ((1+2*kp)/(4*mN**2))*rho2Phipp_p(r) \
     + ((2*kn)/(4*mN**2))*rho2Phipp_n(r) )

def rhow_composition_osz(r,rhoM_p,rhoM_n,rho2M_p,rho2M_n,rho2Phipp_p,rho2Phipp_n,Qw_p=constants.Qw_p,Qw_n=constants.Qw_n,rsqp=constants.rsq_p,rsqn=constants.rsq_n,rsqsN=constants.rsq_sN,kp=constants.kappa_p,kn=constants.kappa_n,ksN=constants.kappa_sN,mN=masses.mN):
    # rho are "F.T." (using the correct L in j_L) of F(q), rho2 are F.T. of q^2 F(q), ...
    mN/=constants.hc
    return 1 * \
    ( Qw_p*rhoM_p(r) - (Qw_p*((rsqp/6)+(1./(8*mN**2))) + Qw_n*((rsqn/6)+(rsqsN/6)))*rho2M_p(r) \
     + Qw_n*rhoM_n(r) - (Qw_n*((rsqp/6)+(rsqsN/6)+(1./(8*mN**2))) + Qw_p*(rsqn/6))*rho2M_n(r) \
     + ((Qw_p*(1+2*kp)+Qw_n*(2*kn+2*ksN))/(4*mN**2))*rho2Phipp_p(r) \
     + ((Qw_n*(1+2*kp+2*ksN)+Qw_p*(2*kn))/(4*mN**2))*rho2Phipp_n(r) )

# currently unused
# def jmag_composition_osz(r,j1Delta_p,j1Sigmap_p,j1Sigmap_n,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
#     # j1 are "F.T." (using the correct L in j_L) of q^1 F(q)
#     mN/=constants.hc
#     return 1*(-1j/mN)*( j1Delta_p(r) \
#      - ((1+kp)/2)*j1Sigmap_p(r) \
#      - (kn/2)*j1Sigmap_n(r) )