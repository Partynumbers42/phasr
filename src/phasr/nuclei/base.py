from .. import constants, masses
from ..physical_constants.iaea_nds import massofnucleusZN, abundanceofnucleusZN, JPofnucleusZN
from ..utility.math import radial_laplace

import numpy as np
pi = np.pi

from functools import partial

class nucleus_base:
    def __init__(self,name,Z, A, mass=None, abundance=None, spin=None, parity=None, #spline_hyp1f1=None, fp=False, ap_dps=15, 
                 **args):
        #
        self.nucleus_type = "base"
        self.name = name
        self.Z = Z
        self.A = A
        #
        self.mass = mass
        if self.mass is None:
            self.lookup_nucleus_mass()
        self.abundance=abundance
        if self.abundance is None:
            self.lookup_nucleus_abundance()
        self.spin=spin
        self.parity=parity
        if (self.spin is None) or (self.parity is None):
            self.lookup_nucleus_JP()
        Qw_p=constants.Qw_p
        Qw_n=constants.Qw_n
        self.Qw = self.Z*Qw_p + (self.A-self.Z)*Qw_n
        #
        # move to numerical?
        if 'form_factor_dict' in args:
            form_factor_dict=args['form_factor_dict']
            multipoles_form_factor = [key[1:] for key in form_factor_dict]
            if hasattr(self,"multipoles"):
                self.multipoles = list(np.unique(self.multipoles+multipoles_form_factor))
            else:
                self.multipoles = multipoles_form_factor
            # Expected for numerical inputs keys: FM0p, FM0n, FM2p, FM2n, ... , FDelta1p, ... , FSigmap1n, ...
            for key in form_factor_dict:
                setattr(self,key,form_factor_dict[key])
        #
        if 'density_dict' in args:
            density_dict=args['density_dict']
            multipoles_charge_density = [key[3:] for key in density_dict]
            if hasattr(self,"multipoles"):
                self.multipoles = list(np.unique(self.multipoles+multipoles_charge_density))
            else:
                self.multipoles = multipoles_charge_density
            
            # Expected keys: rhoM0p, rhoM0n, rhoPhipp0p, rhoPhipp0n, ...
            for key in density_dict:
                setattr(self,key,density_dict[key])
        #
        nucleus_base.update_dependencies(self)
        #
    
    def update_dependencies(self):

        if (not hasattr(self,'rho_M0p')) and hasattr(self,'proton_density'):
            self.rhoM0p = self.proton_density
            self.multipoles = list(np.unique(self.multipoles+["M0p"]))

        if (not hasattr(self,'rho_M0n')) and hasattr(self,'neutron_density'):
            self.rhoM0n = self.neutron_density
            self.multipoles = list(np.unique(self.multipoles+["M0n"]))
        
        #if hasattr(self,"multipoles"):
        #    self.update_basis_representations()

        if (not hasattr(self,'proton_density')) and hasattr(self,'rhoM0p'):
            self.proton_density = self.rhoM0p
        
        if (not hasattr(self,'neutron_density')) and hasattr(self,'rhoM0n'):
            self.neutron_density = self.rhoM0n
        
        if (not hasattr(self,'form_factor')) and (hasattr(self,'FM0p') and hasattr(self,'FM0n') and hasattr(self,'FPhipp0p') and hasattr(self,'FPhipp0n')):
            #def F0ch(q): return self.Fch(q,0)
            self.form_factor = partial(self.Fch,L=0)
        
        if (not hasattr(self,'charge_density')) and (hasattr(self,'rhoM0p') and hasattr(self,'rhoM0n') and hasattr(self,'rhoPhipp0p') and hasattr(self,'rhoPhipp0n')):
            #def rhoch(r): return self.rhoch(r)
            self.charge_density = self.rhoch
        
        if (not hasattr(self,'weak_density')) and (hasattr(self,'rhoM0p') and hasattr(self,'rhoM0n') and hasattr(self,'rhoPhipp0p') and hasattr(self,'rhoPhipp0n')):
            #def rhow(r): return self.rhow(r)
            self.weak_density = self.rhow
        
        if (not hasattr(self,'weak_potential')) and hasattr(self,'weak_density'):
            #def Vweak(r): return constants.fermi_constant*constants.hc**2/(2**(3./2.))*self.weak_density(r)
            self.weak_potential = self.Vweak
        
        if (not hasattr(self,'electric_field')) and (hasattr(self,'ElM0p') and hasattr(self,'ElM0n') and hasattr(self,'ElPhipp0p') and hasattr(self,'ElPhipp0n')):
            #def Elch(r): return self.Elch(r)
            self.electric_field = self.Elch
        
        if (not hasattr(self,'electric_potential')) and (hasattr(self,'VM0p') and hasattr(self,'VM0n') and hasattr(self,'VPhipp0p') and hasattr(self,'VPhipp0n')):
            #def Vch(r): return self.Vch(r)
            self.electric_potential = self.Vch
    
    # introduce this option again?
    # def update_basis_representations(self):
    #     for structure in ['F']:
    #         for multipole in [key[:-1] for key in self.multipoles]:
    #                 if (hasattr(self,structure+multipole+'0') and hasattr(self,structure+multipole+'1')):
    #                     for nuc in ['p','n']:
    #                         if not hasattr(self,structure+multipole+nuc):
    #                             F0 = getattr(self,structure+multipole+'0')
    #                             F1 = getattr(self,structure+multipole+'1')
    #                             def Fnuc(q,F0=F0,F1=F1,nuc=nuc): return Isospin_basis_to_nucleon_basis(F0(q),F1(q),nuc)
    #                             setattr(self,structure+multipole+nuc,Fnuc)
    #                             self.multipoles = list(np.unique(self.multipoles+[multipole+nuc]))
    #                 if (hasattr(self,structure+multipole+'p') and hasattr(self,structure+multipole+'n')):
    #                     for iso in ['0','1']:
    #                         if not hasattr(self,structure+multipole+iso):
    #                             Fp = getattr(self,structure+multipole+'p')
    #                             Fn = getattr(self,structure+multipole+'n')
    #                             def Fiso(q,Fp=Fp,Fn=Fn,iso=iso): return Nucleon_basis_to_isospin_basis(Fp(q),Fn(q),iso)
    #                             setattr(self,structure+multipole+iso,Fiso)
    #                             self.multipoles = list(np.unique(self.multipoles+[multipole+iso]))

    def update_name(self,name):
        self.name=name
    
    def update_Z(self,Z):
        self.Z=Z
        self.update_dependencies()
    
    def update_A(self,A):
        self.A=A
        self.update_dependencies()
    
    def update_m(self,m):
        self.mass=m
        self.update_dependencies()
    
    def update_spin(self,spin):
        self.spin=spin
        self.update_dependencies()

    def update_parity(self,parity):
        self.parity=parity
        self.update_dependencies()

    def update_abundance(self,abundance):
        self.abundance=abundance
        self.update_dependencies()

    def lookup_nucleus_mass(self):
        self.mass = massofnucleusZN(self.Z,self.A-self.Z)

    def lookup_nucleus_abundance(self):
        self.abundance = abundanceofnucleusZN(self.Z,self.A-self.Z)

    def lookup_nucleus_JP(self):
        JP = JPofnucleusZN(self.Z,self.A-self.Z)
        if type(JP) is tuple:
            J , P = JP
            if self.spin is not None and J!=self.spin:
                raise ValueError('looked up spin J='+str(J)+' different to present one J='+str(self.spin))
            if self.parity is not None and P!=self.parity:
                raise ValueError('looked up parity P='+str(P)+' different to present one P='+str(self.parity))
            self.spin, self.parity = J, P

    def Fch(self,q,L=0):
        
        if L>=2*self.spin+1:
            raise ValueError('This nucleus has a maximum L of '+str(2*self.spin))

        if L%2==1:
            raise ValueError('Fch only nonzero for even L')
        
        if not (hasattr(self,'FM'+str(L)+'p') and hasattr(self,'FM'+str(L)+'n') and hasattr(self,'FPhipp'+str(L)+'p') and hasattr(self,'FPhipp'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate Fch'+str(L))
        
        FMLp=getattr(self,'FM'+str(L)+'p')
        FMLn=getattr(self,'FM'+str(L)+'n')
        FPhippLp=getattr(self,'FPhipp'+str(L)+'p')
        FPhippLn=getattr(self,'FPhipp'+str(L)+'n')
        return Fch_composition(q,FMLp,FMLn,FPhippLp,FPhippLn,self.Z)
        
    def Fmag(self,q,L=1):
        
        if L>=2*self.spin+1:
            raise ValueError('This nucleus has a maximum L of '+str(2*self.spin))

        if L%2==0:
            raise ValueError('Fmag only nonzero for odd L')

        if not (hasattr(self,'FDelta'+str(L)+'p') and hasattr(self,'FSigmap'+str(L)+'p') and hasattr(self,'FSigmap'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate Fmag'+str(L))
        
        FDeltaLp=getattr(self,'FDelta'+str(L)+'p')
        FSigmapLp=getattr(self,'FSigmap'+str(L)+'p')
        FSigmapLn=getattr(self,'FSigmap'+str(L)+'n')
        return Fmag_composition(q,FDeltaLp,FSigmapLp,FSigmapLn)
    
    def Fw(self,q,L=0):
        
        if L>=2*self.spin+1:
            raise ValueError('This nucleus has a maximum L of '+str(2*self.spin))

        if L%2==1:
            raise ValueError('Fch only nonzero for even L')
        
        if not (hasattr(self,'FM'+str(L)+'p') and hasattr(self,'FM'+str(L)+'n') and hasattr(self,'FPhipp'+str(L)+'p') and hasattr(self,'FPhipp'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate Fch'+str(L))
        
        FMLp=getattr(self,'FM'+str(L)+'p')
        FMLn=getattr(self,'FM'+str(L)+'n')
        FPhippLp=getattr(self,'FPhipp'+str(L)+'p')
        FPhippLn=getattr(self,'FPhipp'+str(L)+'n')
        return Fw_composition(q,FMLp,FMLn,FPhippLp,FPhippLn,self.weak_charge)

    def rhoch(self,r):
        L=0
        # For L>0 the "F.T." (with j_L) with and q^2 is non-trivial to write in terms of rho_L (for L=0 - laplace of rho) 
        if not (hasattr(self,'rhoM'+str(L)+'p') and hasattr(self,'rhoM'+str(L)+'n') and hasattr(self,'rhoPhipp'+str(L)+'p') and hasattr(self,'rhoPhipp'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate rhoch'+str(L))
        
        rhoMLp=getattr(self,'rhoM'+str(L)+'p')
        rhoMLn=getattr(self,'rhoM'+str(L)+'n')
        rhoPhippLp=getattr(self,'rhoPhipp'+str(L)+'p')
        rhoPhippLn=getattr(self,'rhoPhipp'+str(L)+'n')
        
        if hasattr(self,'rho2M'+str(L)+'p') and hasattr(self,'rho2M'+str(L)+'n') and hasattr(self,'rho2Phipp'+str(L)+'p') and hasattr(self,'rho2Phipp'+str(L)+'n'):   
            rho2MLp=getattr(self,'rho2M'+str(L)+'p')
            rho2MLn=getattr(self,'rho2M'+str(L)+'n')
            rho2PhippLp=getattr(self,'rho2Phipp'+str(L)+'p')
            rho2PhippLn=getattr(self,'rho2Phipp'+str(L)+'n')
            return rho0ch_composition(r,rhoMLp,rhoMLn,rhoPhippLp,rhoPhippLn,rho2MLp,rho2MLn,rho2PhippLp,rho2PhippLn)
        else:
            return rho0ch_composition(r,rhoMLp,rhoMLn,rhoPhippLp,rhoPhippLn)

    def rhow(self,r):
        L=0
        # For L>0 the "F.T." (with j_L) with and q^2 is non-trivial to write in terms of rho_L (for L=0 - laplace of rho) 
        if not (hasattr(self,'rhoM'+str(L)+'p') and hasattr(self,'rhoM'+str(L)+'n') and hasattr(self,'rhoPhipp'+str(L)+'p') and hasattr(self,'rhoPhipp'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate rhow'+str(L))
        
        rhoMLp=getattr(self,'rhoM'+str(L)+'p')
        rhoMLn=getattr(self,'rhoM'+str(L)+'n')
        rhoPhippLp=getattr(self,'rhoPhipp'+str(L)+'p')
        rhoPhippLn=getattr(self,'rhoPhipp'+str(L)+'n')
        
        if hasattr(self,'rho2M'+str(L)+'p') and hasattr(self,'rho2M'+str(L)+'n') and hasattr(self,'rho2Phipp'+str(L)+'p') and hasattr(self,'rho2Phipp'+str(L)+'n'):   
            rho2MLp=getattr(self,'rho2M'+str(L)+'p')
            rho2MLn=getattr(self,'rho2M'+str(L)+'n')
            rho2PhippLp=getattr(self,'rho2Phipp'+str(L)+'p')
            rho2PhippLn=getattr(self,'rho2Phipp'+str(L)+'n')
            return rho0w_composition(r,rhoMLp,rhoMLn,rhoPhippLp,rhoPhippLn,rho2MLp,rho2MLn,rho2PhippLp,rho2PhippLn)
        else:
            return rho0w_composition(r,rhoMLp,rhoMLn,rhoPhippLp,rhoPhippLn)
    
    def Elch(self,r):
        
        # might lead to sign errors for r->0
        
        L=0
        if not (hasattr(self,'ElM'+str(L)+'p') and hasattr(self,'ElM'+str(L)+'n') and hasattr(self,'ElPhipp'+str(L)+'p') and hasattr(self,'ElPhipp'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate Elch'+str(L))
        
        ElMLp=getattr(self,'ElM'+str(L)+'p')
        ElMLn=getattr(self,'ElM'+str(L)+'n')
        ElPhippLp=getattr(self,'ElPhipp'+str(L)+'p')
        ElPhippLn=getattr(self,'ElPhipp'+str(L)+'n')
        return rho0ch_composition(r,ElMLp,ElMLn,ElPhippLp,ElPhippLn)

    def Vch(self,r):
        L=0
        if not (hasattr(self,'VM'+str(L)+'p') and hasattr(self,'VM'+str(L)+'n') and hasattr(self,'VPhipp'+str(L)+'p') and hasattr(self,'VPhipp'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate Vch'+str(L))
        
        VMLp=getattr(self,'VM'+str(L)+'p')
        VMLn=getattr(self,'VM'+str(L)+'n')
        VPhippLp=getattr(self,'VPhipp'+str(L)+'p')
        VPhippLn=getattr(self,'VPhipp'+str(L)+'n')
        return rho0ch_composition(r,VMLp,VMLn,VPhippLp,VPhippLn)
    
    def Vweak(self,r):
        return constants.fermi_constant*constants.hc**2/(2**(3./2.))*self.weak_density(r)
            
def Fch_composition(q,FM_p,FM_n,FPhipp_p,FPhipp_n,Z,rsqp=constants.rsq_p,rsqn=constants.rsq_n,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
    rsqp/=constants.hc**2
    rsqn/=constants.hc**2
    return 1/Z * \
    ( (1-(rsqp/6)*q**2-((q**2)/(8*mN**2)))*FM_p(q) \
     - (rsqn/6)*(q**2)*FM_n(q) \
     + ((1+2*kp)/(4*mN**2))*(q**2)*FPhipp_p(q) \
     + ((2*kn)/(4*mN**2))*(q**2)*FPhipp_n(q) )

def Fmag_composition(q,FDelta_p,FSigmap_p,FSigmap_n,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
    return (-1j*q/mN)*( FDelta_p(q) \
     - ((1+kp)/2)*FSigmap_p(q) \
     - (kn/2)*FSigmap_n(q) )

def Fw_composition(q,FM_p,FM_n,FPhipp_p,FPhipp_n,Qw,Qw_p=constants.Qw_p,Qw_n=constants.Qw_n,rsqp=constants.rsq_p,rsqn=constants.rsq_n,rsqsN=constants.rsq_sN,kp=constants.kappa_p,kn=constants.kappa_n,ksN=constants.kappa_sN,mN=masses.mN):
    rsqp/=constants.hc**2
    rsqn/=constants.hc**2
    rsqsN/=constants.hc**2
    #Qw = Z*Qw_p + (A-Z)*Qw_n
    return 1/Qw * \
    ( (Qw_p*(1-(rsqp/6)*q**2-((q**2)/(8*mN**2))) + Qw_n*(-(rsqn/6)*q**2-(rsqsN/6)*q**2))*FM_p(q) \
     + (Qw_n*(1-(rsqp/6)*q**2-(rsqsN/6)*q**2-((q**2)/(8*mN**2))) + Qw_p*(-(rsqn/6)*q**2))*FM_n(q) \
     + ((Qw_p*(1+2*kp)+Qw_n*(2*kn+2*ksN))/(4*mN**2))*(q**2)*FPhipp_p(q) \
     + ((Qw_n*(1+2*kp+2*ksN)+Qw_p*(2*kn))/(4*mN**2))*(q**2)*FPhipp_n(q) )

def rho0ch_composition(r,rhoM_p,rhoM_n,rhoPhipp_p,rhoPhipp_n,rho2M_p=None,rho2M_n=None,rho2Phipp_p=None,rho2Phipp_n=None,rsqp=constants.rsq_p,rqsn=constants.rsq_n,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
    # only valid for L=0
    mN/=constants.hc 
    if rho2M_p is None:
        def rho2M_p(r): return -radial_laplace(rhoM_p)(r)
    if rho2M_n is None:
        def rho2M_n(r): return -radial_laplace(rhoM_n)(r)
    if rho2Phipp_p is None:
        def rho2Phipp_p(r): return -radial_laplace(rhoPhipp_p)(r)
    if rho2Phipp_n is None:
        def rho2Phipp_n(r): return -radial_laplace(rhoPhipp_n)(r)
    return 1 * \
    ( rhoM_p(r) - ((rsqp/6)+(1./(8*mN**2)))*rho2M_p(r) \
     - (rqsn/6)*rho2M_n(r) \
     + ((1+2*kp)/(4*mN**2))*rho2Phipp_p(r) \
     + ((2*kn)/(4*mN**2))*rho2Phipp_n(r) )

def rho0w_composition(r,rhoM_p,rhoM_n,rhoPhipp_p,rhoPhipp_n,rho2M_p=None,rho2M_n=None,rho2Phipp_p=None,rho2Phipp_n=None,Qw_p=constants.Qw_p,Qw_n=constants.Qw_n,rsqp=constants.rsq_p,rsqn=constants.rsq_n,rsqsN=constants.rsq_sN,kp=constants.kappa_p,kn=constants.kappa_n,ksN=constants.kappa_sN,mN=masses.mN):
    # only valid for L=0
    mN/=constants.hc
    if rho2M_p is None:
        def rho2M_p(r): return -radial_laplace(rhoM_p)(r)
    if rho2M_n is None:
        def rho2M_n(r): return -radial_laplace(rhoM_n)(r)
    if rho2Phipp_p is None:
        def rho2Phipp_p(r): return -radial_laplace(rhoPhipp_p)(r)
    if rho2Phipp_n is None:
        def rho2Phipp_n(r): return -radial_laplace(rhoPhipp_n)(r)
    return 1 * \
    ( Qw_p*rhoM_p(r) - (Qw_p*((rsqp/6)+(1./(8*mN**2))) + Qw_n*((rsqn/6)+(rsqsN/6)))*rho2M_p(r) \
     + Qw_n*rhoM_n(r) - (Qw_n*((rsqp/6)+(rsqsN/6)+(1./(8*mN**2))) + Qw_p*(rsqn/6))*rho2M_n(r) \
     + ((Qw_p*(1+2*kp)+Qw_n*(2*kn+2*ksN))/(4*mN**2))*rho2Phipp_p(r) \
     + ((Qw_n*(1+2*kp+2*ksN)+Qw_p*(2*kn))/(4*mN**2))*rho2Phipp_n(r) )
