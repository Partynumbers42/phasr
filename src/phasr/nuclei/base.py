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
        #self.weak_charge = weak_charge # remove option to provide weak_charge?
        #if (self.weak_charge is None):
        #    self.weak_charge = self.Qw 
        #
        if ('k_barrett' in args) and ('alpha_barrett' in args):
            self.k_barrett = args['k_barrett']
            self.alpha_barrett = args['alpha_barrett']
        else:
            self.k_barrett = None
            self.alpha_barrett = None
        # Add lookup mechanism/file where the k, alpha values are saved?
        #
        if 'form_factor_dict' in args:
            form_factor_dict=args['form_factor_dict']
            # Expected keys: FM0p, FM0n, FM2p, FM2n, ... , FDelta1p, ...
            for key in form_factor_dict:
                setattr(self,key,form_factor_dict[key])
        #
        if 'density_dict' in args:
            density_dict=args['density_dict']
            # Expected keys: rhoM0p, rhoM0n, rhoM2p, rhoM2n, ... 
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
        pass
    
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
    
    def Fch(self,q,L):
        
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
        
    def Fmag(self,q,L):
        
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
    
    def Fw(self,q,L):
        
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

def Fch_composition(q,FM_p,FM_n,FPhipp_p,FPhipp_n,Z,rsqp=constants.rsq_p/constants.hc**2,rqsn=constants.rsq_n/constants.hc**2,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
    return 1/Z * \
    ( (1-(rsqp/6)*q**2-((q**2)/(8*mN**2)))*FM_p(q) \
     - (rqsn/6)*(q**2)*FM_n(q) \
     + ((1+2*kp)/(4*mN**2))*(q**2)*FPhipp_p(q) \
     + ((2*kn)/(4*mN**2))*(q**2)*FPhipp_n(q) )

def Fmag_composition(q,FDelta_p,FSigmap_p,FSigmap_n,kp=constants.kappa_p,kn=constants.kappa_n,mN=masses.mN):
    return (-1j*q/mN)*( FDelta_p(q) \
     - ((1+kp)/2)*FSigmap_p(q) \
     - (kn/2)*FSigmap_n(q) )

def Fw_composition(q,FM_p,FM_n,FPhipp_p,FPhipp_n,Qw,Qw_p=constants.Qw_p,Qw_n=constants.Qw_n,rsqp=constants.rsq_p/constants.hc**2,rsqn=constants.rsq_n/constants.hc**2,rsqsN=constants.rsq_sN/constants.hc**2,kp=constants.kappa_p,kn=constants.kappa_n,ksN=constants.kappa_sN,mN=masses.mN):
    #Z*Qw_p + (A-Z)*Qw_n
    return 1/Qw * \
    ( (Qw_p*(1-(rsqp/6)*q**2-((q**2)/(8*mN**2))) + Qw_n*(-(rsqn/6)*q**2-(rsqsN/6)*q**2))*FM_p(q) \
     + (Qw_n*(1-(rsqp/6)*q**2-(rsqsN/6)*q**2-((q**2)/(8*mN**2))) + Qw_p*(-(rsqn/6)*q**2))*FM_n(q) \
     + ((Qw_p*(1+2*kp)+Qw_n*(2*kn+2*ksN))/(4*mN**2))*(q**2)*FPhipp_p(q) \
     + ((Qw_n*(1+2*kp+2*ksN)+Qw_p*(2*kn))/(4*mN**2))*(q**2)*FPhipp_n(q) )