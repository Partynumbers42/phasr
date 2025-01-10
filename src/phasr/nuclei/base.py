from .. import constants, masses
from ..physical_constants.iaea_nds import massofnucleusZN, abundanceofnucleusZN, JPofnucleusZN

import numpy as np
pi = np.pi

class nucleus_base:
    def __init__(self,name,Z, A, m=None, abundance=None, spin=None, parity=None,# weak_charge=None,
                 #spline_hyp1f1=None, fp=False, ap_dps=15, 
                 **args):
        #
        self.nucleus_type = "base"
        self.name = name
        self.Z = Z
        self.A = A
        #
        self.m = m
        if self.m is None:
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
        if ('k_barrett' in args) and ('alpha_barrett' in args):
            self.k_barrett = args['k_barrett']
            self.alpha_barrett = args['alpha_barrett']
        else:
            self.k_barrett = None
            self.alpha_barrett = None
        #
        if 'form_factor_dict' in args:
            form_factor_dict=args['form_factor_dict']
            #self.multipoles_form_factor = [key[1:] for key in form_factor_dict]
            # Expected keys: FM0p, FM0n, FM2p, FM2n, ... , FDelta1p, ... , FSigmap1n, ...
            for key in form_factor_dict:
                setattr(self,key,form_factor_dict[key])
        #
        if 'density_dict' in args:
            density_dict=args['density_dict']
            #self.multipoles_charge_density = [key[3:] for key in density_dict]
            # Expected keys: rhoM0p, rhoM0n, rhoM2p, rhoM2n, ... , rho2M0p, rho2M0n, ... (rho are F.T. of F(q), rho2 are F.T. of q^2 F(q), ...)
            for key in density_dict:
                setattr(self,key,density_dict[key])
        #
        # remaining keywords are made attributes, do i want this???
        #for key in args:
        #    if not hasattr(self,key,args[key]):
        #        setattr(self,key,args[key])
        #
        nucleus_base.update_dependencies(self)
        #
    
    def update_dependencies(self):
        
        if (not hasattr(self,'proton_density')) and hasattr(self,'rhoM0p'):
            self.proton_density = self.rhoM0p

        if (not hasattr(self,'rho_M0p')) and hasattr(self,'proton_density'):
            self.rhoM0p = self.proton_density

        if (not hasattr(self,'neutron_density')) and hasattr(self,'rhoM0n'):
            self.neutron_density = self.rhoM0n

        if (not hasattr(self,'rho_M0n')) and hasattr(self,'neutron_density'):
            self.rhoM0n = self.neutron_density
        
        if (not hasattr(self,'form_factor')) and (hasattr(self,'FM0p') and hasattr(self,'FM0n') and hasattr(self,'FPhipp0p') and hasattr(self,'FPhipp0n')):
            def Fch(q): return self.Fch(q,0)
            self.form_factor = Fch
        
        if (not hasattr(self,'charge_density')) and (hasattr(self,'rhoM0p') and hasattr(self,'rho2M0p') and hasattr(self,'rho2M0n') and hasattr(self,'rho2Phipp0p') and hasattr(self,'rho2Phipp0n')):
            def rhoch(r): return self.rhoch(r,0)
            self.charge_density = rhoch
        
        if (not hasattr(self,'weak_density')) and (hasattr(self,'rhoM0p') and hasattr(self,'rhoM0n') and hasattr(self,'rho2M0p') and hasattr(self,'rho2M0n') and hasattr(self,'rho2Phipp0p') and hasattr(self,'rho2Phipp0n')):
            def rhow(r): return self.rhow(r,0)
            self.weak_density = rhow

    def update_name(self,name):
        self.name=name
    
    def update_Z(self,Z):
        self.Z=Z
        self.update_dependencies()
    
    def update_A(self,A):
        self.A=A
        self.update_dependencies()
    
    def update_m(self,m):
        self.m=m
        self.update_dependencies()
    
    def update_spin(self,spin):
        self.spin=spin
        self.update_dependencies()

    def update_parity(self,parity):
        self.parity=parity
        self.update_dependencies()

    def update_k_barrett(self,k_barrett):
        self.k_barrett=k_barrett
        self.update_dependencies()
    
    def update_alpha_barrett(self,alpha_barrett):
        self.alpha_barrett=alpha_barrett
        self.update_dependencies()

    def update_abundance(self,abundance):
        self.abundance=abundance
        self.update_dependencies()

    def lookup_nucleus_mass(self):
        self.m = massofnucleusZN(self.Z,self.A-self.Z)

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
        
    def Fmag(self,q,L=0):
        
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
    
    def rhoch(self,r,L=0):
        
        if L>=2*self.spin+1:
            raise ValueError('This nucleus has a maximum L of '+str(2*self.spin))

        if L%2==1:
            raise ValueError('rhoch only nonzero for even L')
        
        if not (hasattr(self,'rhoM'+str(L)+'p') and hasattr(self,'rho2M'+str(L)+'n') and hasattr(self,'rho2M'+str(L)+'n') and hasattr(self,'rho2Phipp'+str(L)+'p') and hasattr(self,'rho2Phipp'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate rhoch'+str(L))
        
        rhoMLp=getattr(self,'rhoM'+str(L)+'p')
        rho2MLp=getattr(self,'rho2M'+str(L)+'p')
        rho2MLn=getattr(self,'rho2M'+str(L)+'n')
        rho2PhippLp=getattr(self,'rho2Phipp'+str(L)+'p')
        rho2PhippLn=getattr(self,'rho2Phipp'+str(L)+'n')
        return rhoch_composition(r,rhoMLp,rho2MLp,rho2MLn,rho2PhippLp,rho2PhippLn)

    def jmag(self,r,L,Lp=None):
        #weak current j_LL' for L=L'
        
        if Lp is None:
            Lp=L

        if L!=Lp:
            raise ValueError("No implementation for j_LL' for L!=L'")
        
        if L>=2*self.spin+1:
            raise ValueError('This nucleus has a maximum L of '+str(2*self.spin))

        if L%2==0:
            raise ValueError('jmag only nonzero for odd L')

        if not (hasattr(self,'j1Delta'+str(L)+'p') and hasattr(self,'j1Sigmap'+str(L)+'p') and hasattr(self,'j1Sigmap'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate jmag'+str(L))
        
        j1DeltaLp=getattr(self,'j1Delta'+str(L)+'p')
        j1SigmapLp=getattr(self,'j1Sigmap'+str(L)+'p')
        j1SigmapLn=getattr(self,'j1Sigmap'+str(L)+'n')
        return jmag_composition(r,j1DeltaLp,j1SigmapLp,j1SigmapLn)

    def rhow(self,r,L=0):
        
        if L>=2*self.spin+1:
            raise ValueError('This nucleus has a maximum L of '+str(2*self.spin))

        if L%2==1:
            raise ValueError('rhow only nonzero for even L')
        
        if not (hasattr(self,'rhoM'+str(L)+'p') and hasattr(self,'rhoM'+str(L)+'n') and hasattr(self,'rho2M'+str(L)+'p') and hasattr(self,'rho2M'+str(L)+'n') and hasattr(self,'rho2Phipp'+str(L)+'p') and hasattr(self,'rho2Phipp'+str(L)+'n')):
            raise ValueError('Missing multipoles to evaluate Fch'+str(L))
        
        rhoMLp=getattr(self,'rhoM'+str(L)+'p')
        rhoMLn=getattr(self,'rhoM'+str(L)+'n')
        rho2MLp=getattr(self,'rho2M'+str(L)+'p')
        rho2MLn=getattr(self,'rho2M'+str(L)+'n')
        rho2PhippLp=getattr(self,'rho2Phipp'+str(L)+'p')
        rho2PhippLn=getattr(self,'rho2Phipp'+str(L)+'n')
        return rhow_composition(r,rhoMLp,rhoMLn,rho2MLp,rho2MLn,rho2PhippLp,rho2PhippLn)

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

def rhoch_composition(r,rhoM_p,rho2M_p,rho2M_n,rho2Phipp_p,rho2Phipp_n,rsqp=constants.rsq_p,rqsn=constants.rsq_n,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
    # rho are F.T. of F(q), rho2 are F.T. of q^2 F(q), ...
    mN/=constants.hc
    return 1/(2*pi**2) * \
    ( rhoM_p(r) - ((rsqp/6)+(1./(8*mN**2)))*rho2M_p(r) \
     - (rqsn/6)*rho2M_n(r) \
     + ((1+2*kp)/(4*mN**2))*rho2Phipp_p(r) \
     + ((2*kn)/(4*mN**2))*rho2Phipp_n(r) )

def jmag_composition(r,j1Delta_p,j1Sigmap_p,j1Sigmap_n,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
    # j1 are F.T. of q^1 F(q)
    mN/=constants.hc
    return 1/(2*pi**2)*(-1j/mN)*( j1Delta_p(r) \
     - ((1+kp)/2)*j1Sigmap_p(r) \
     - (kn/2)*j1Sigmap_n(r) )

def rhow_composition(r,rhoM_p,rhoM_n,rho2M_p,rho2M_n,rho2Phipp_p,rho2Phipp_n,Qw_p=constants.Qw_p,Qw_n=constants.Qw_n,rsqp=constants.rsq_p,rsqn=constants.rsq_n,rsqsN=constants.rsq_sN,kp=constants.kappa_p,kn=constants.kappa_n,ksN=constants.kappa_sN,mN=masses.mN):
    # rho are F.T. of F(q), rho2 are F.T. of q^2 F(q), ...
    mN/=constants.hc
    return 1/(2*pi**2) * \
    ( Qw_p*rhoM_p(r) - (Qw_p*((rsqp/6)+(1./(8*mN**2))) + Qw_n*((rsqn/6)+(rsqsN/6)))*rho2M_p(r) \
     + Qw_n*rhoM_n(r) - (Qw_n*((rsqp/6)+(rsqsN/6)+(1./(8*mN**2))) + Qw_p*(rsqn/6))*rho2M_n(r) \
     + ((Qw_p*(1+2*kp)+Qw_n*(2*kn+2*ksN))/(4*mN**2))*rho2Phipp_p(r) \
     + ((Qw_n*(1+2*kp+2*ksN)+Qw_p*(2*kn))/(4*mN**2))*rho2Phipp_n(r) )